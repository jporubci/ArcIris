from argparse import ArgumentParser
from dataset import ImageDataset
from functools import partial
import madgrad
import numpy as np
from online_triplet_loss.losses import batch_all_triplet_loss, batch_hard_triplet_loss
import os
from PIL import Image
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision import models
from torchvision.models.convnext import LayerNorm2d
from torchvision.transforms import Compose, Normalize, Resize, ToTensor

torch.set_printoptions(precision=4, sci_mode=False)


def train(args, model):

    if args.polar:
        input_transform = Compose(
            [Resize((64, 512)), ToTensor(), Normalize(mean=(0.5,), std=(0.5,))]
        )
    else:
        input_transform = Compose(
            [Resize((256, 256)), ToTensor(), Normalize(mean=(0.5,), std=(0.5,))]
        )

    if args.log_txt:
        sys.stdout = open(
            "./"
            + args.tag
            + "_"
            + args.model_type.lower()
            + "_"
            + str(args.stem_width)
            + "_"
            + args.distance_type
            + "_trainer_output.txt",
            "a",
        )

    directory = os.path.dirname(
        args.tag
        + "_"
        + args.model_type.lower()
        + "_"
        + str(args.stem_width)
        + "_"
        + args.distance_type
        + "_checkpoint/"
    )
    if not os.path.exists(directory):
        os.makedirs(directory)

    # weight = torch.ones(NUM_CLASSES)
    print(f"image_dir: {args.image_dir}", flush=True)
    print(f"input_transform: {input_transform}", flush=True)
    print("------------loader beginning-----------")

    dataset = ImageDataset(
        args.image_dir,
        args.img_uid_map,
        args.polar,
        input_transform,
        True,
        args.flip,
        True,
    )
    if (args.val_image_dir is None) or (args.val_img_uid_map is None):
        val_length = int(0.05 * len(dataset))
        train_length = len(dataset) - val_length
        val_dataset, train_dataset = random_split(
            dataset,
            [val_length, train_length],
            generator=torch.Generator().manual_seed(42),
        )
        print(
            "Train Dataset Length:",
            len(train_dataset),
            ", Val Dataset Length:",
            len(val_dataset),
            flush=True,
        )
        loader = DataLoader(
            train_dataset,
            num_workers=args.num_workers,
            batch_size=args.batch_size,
            shuffle=True,
        )
        print("loader complete", flush=True)
        print("-----------val loader beginning--------", flush=True)
        val_loader = DataLoader(
            val_dataset,
            num_workers=args.num_workers,
            batch_size=args.batch_size,
            shuffle=False,
        )
        print("val_loader complete", flush=True)
    else:
        loader = DataLoader(
            dataset,
            num_workers=args.num_workers,
            batch_size=args.batch_size,
            shuffle=True,
        )
        val_dataset = ImageDataset(
            args.val_image_dir,
            args.val_img_uid_map,
            args.polar,
            input_transform,
            False,
            False,
            True,
        )
        val_loader = DataLoader(
            val_dataset,
            num_workers=args.num_workers,
            batch_size=args.batch_size,
            shuffle=False,
        )

    print("training length:", len(loader), flush=True)
    print("validation length:", len(val_loader), flush=True)

    # optimizer = SGD(model.parameters(), lr = args.lr, momentum=0.9, weight_decay=0.001)
    # optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3, eps=1e-4)
    # optimizer = NAdam(model.parameters(), lr=args.lr)
    # if args.finetune and args.model_type.lower() == 'resnet50':
    #    optimizer = madgrad.MADGRAD(model.fc.parameters(), lr=0.0001, weight_decay=0.01)
    # else:
    #    optimizer = madgrad.MADGRAD(model.parameters(), lr=args.lr, weight_decay=0.01)

    optimizer = madgrad.MADGRAD(model.parameters(), lr=args.lr)
    best_val_loss_average = float("inf")

    if args.cuda:
        scaler = torch.cuda.amp.GradScaler()

    if args.margin is None:
        if args.distance_type == "euclidean":
            args.margin = 1.0
        else:
            args.margin = 0.05

    for epoch in range(1, args.num_epochs + 1):

        model.train()

        # if (epoch - 1) % 10 == 0 and epoch != 1:
        #    for param_group in optimizer.param_groups:
        #        param_group["lr"] *= 0.1

        epoch_loss = []

        # if args.finetune and epoch > 15 and args.model_type.lower() == 'resnet50':
        #    optimizer = madgrad.MADGRAD(model.parameters(), lr=args.lr, weight_decay=0.01)
        #    args.finetune = False

        # loader.dataset.dataset.set_val(False)
        for batch, data in enumerate(loader):
            # setup input
            images = data["images"]
            images = images.repeat(1, 3, 1, 1)

            if args.debug:
                if not os.path.exists("./debug/"):
                    os.mkdir("./debug/")
                for i in range(images.shape[0]):
                    image = images[i][0].clone().detach().cpu().numpy()
                    # print(image.min(), image.max())
                    image = (image * 0.5) + 0.5
                    # print(image.min(), image.max())
                    image = np.clip(image, 0, 1)
                    image = (image * 255).astype(np.uint8)
                    image_pil = Image.fromarray(image, "L")
                    image_pil.save("./debug/" + str(i) + ".png")
                args.debug = False

            labels = data["labels"]
            if args.cuda:
                images = images.cuda()
                labels = labels.cuda()
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    outputs = model(images)
                    loss = batch_hard_triplet_loss(
                        labels,
                        outputs,
                        margin=args.margin,
                        distance_type=args.distance_type,
                    )
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            else:
                outputs = model(images)
                loss = batch_hard_triplet_loss(labels, outputs, margin=0.05)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            # compute IoU
            epoch_loss.append(loss.item())

            if batch % args.log_batch == 0:
                train_loss_average = sum(epoch_loss) / len(epoch_loss)
                print(
                    "Train loss: {aver} (epoch: {epoch}, batch: {batch}/{total})".format(
                        aver=train_loss_average,
                        epoch=epoch,
                        batch=batch,
                        total=len(loader),
                    ),
                    flush=True,
                )

            if args.log_txt:
                sys.stdout.close()
                sys.stdout = open(
                    "./"
                    + args.tag
                    + "_"
                    + args.model_type.lower()
                    + "_"
                    + str(args.stem_width)
                    + "_"
                    + args.distance_type
                    + "_trainer_output.txt",
                    "a",
                )

        # Evaluate test images
        # if epoch % args.eval_epoch == 0:
        #    evaluate(args, model, epoch)

        # Validation set
        if len(val_loader) > 0:
            # val_loader.dataset.dataset.set_val(True)
            val_epoch_loss = []
            model.eval()
            with torch.inference_mode():
                for batch, data in enumerate(val_loader):
                    # setup input
                    images = data["images"]
                    images = images.repeat(1, 3, 1, 1)
                    labels = data["labels"]
                    if args.cuda:
                        images = images.cuda()
                        labels = labels.cuda()

                    outputs = model(images)

                    val_loss, _ = batch_all_triplet_loss(
                        labels,
                        outputs,
                        margin=args.margin,
                        distance_type=args.distance_type,
                    )

                    val_epoch_loss.append(val_loss.item())

                val_loss_average = sum(val_epoch_loss) / len(val_epoch_loss)
                print(
                    "Val loss: {aver} (epoch: {epoch})".format(
                        aver=val_loss_average, epoch=epoch
                    ),
                    flush=True,
                )
                if args.log_txt:
                    sys.stdout.close()
                    sys.stdout = open(
                        "./"
                        + args.tag
                        + "_"
                        + args.model_type.lower()
                        + "_"
                        + str(args.stem_width)
                        + "_"
                        + args.distance_type
                        + "_trainer_output.txt",
                        "a",
                    )
                if val_loss_average < best_val_loss_average:
                    # Save checkpoint
                    best_val_loss_average = val_loss_average
                    filename = os.path.join(
                        directory,
                        "{model}-{epoch:03}-{val}.pth".format(
                            model=args.model_type,
                            epoch=epoch,
                            val=round(val_loss_average, 6),
                        ),
                    )
                    if args.multi_gpu:
                        torch.save(model.module.state_dict(), filename)
                    else:
                        torch.save(model.state_dict(), filename)
            if args.log_txt:
                sys.stdout.close()
                sys.stdout = open(
                    "./"
                    + args.tag
                    + "_"
                    + args.model_type.lower()
                    + "_"
                    + str(args.stem_width)
                    + "_"
                    + args.distance_type
                    + "_trainer_output.txt",
                    "a",
                )
    if args.log_txt:
        sys.stdout.close()


def main(args):
    model_type = args.model_type.lower()
    if model_type.startswith("convnext"):
        lastconv_output_channels = {
            "convnext_tiny": 768,
            "convnext_small": 768,
            "convnext_base": 1024,
            "convnext_large": 1536,
        }

        norm_layer = partial(LayerNorm2d, eps=1e-6)

        if model_type == "convnext_tiny":
            model = models.convnext_tiny(weights="DEFAULT")
        elif model_type == "convnext_small":
            model = models.convnext_small(weights="DEFAULT")
        elif model_type == "convnext_base":
            model = models.convnext_base(weights="DEFAULT")
        elif model_type == "convnext_large":
            model = models.convnext_large(weights="DEFAULT")

        model.classifier = nn.Sequential(
            norm_layer(lastconv_output_channels[model_type]),
            nn.Flatten(),
        )
    elif model_type == "resnet50":
        model = models.resnet50(pretrained=True)
        model.fc = nn.Identity()
        args.stem_width = 50

    if args.state:
        try:
            if args.cuda:
                model.load_state_dict(torch.load(args.state))
            else:
                model.load_state_dict(
                    torch.load(args.state, map_location=torch.device("cpu"))
                )
            print("model state loaded")
        except AssertionError:
            print("assertion error")
            model.load_state_dict(
                torch.load(args.state, map_location=lambda storage, loc: storage)
            )

    if args.cuda:
        if torch.cuda.device_count() > 1 and args.multi_gpu:
            model = nn.DataParallel(model.cuda())
        else:
            if args.gpu is not None:
                torch.cuda.set_device(args.gpu)
            model = model.cuda()
        if args.cudnn:
            print("Using CUDNN")
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True

    # if args.mode == 'eval':
    #    evaluate(args, model)

    if args.mode == "train":
        train(args, model)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--state")
    parser.add_argument("--mode", default="train")
    parser.add_argument(
        "--image_dir",
        type=str,
        default="/afs/crc/group/cvrl/czajka/gbir2/aczajka/BXGRID/iris_segmented_SegNet",
    )
    parser.add_argument("--img_uid_map", type=str, default="./img_to_uid_map.json")
    parser.add_argument("--val_image_dir", default=None)
    parser.add_argument("--val_img_uid_map", default=None)
    parser.add_argument("--model_type", type=str, default="convnext_tiny")
    parser.add_argument("--lr", type=float, default=0.00001)
    parser.add_argument("--num_epochs", type=int, default=2001)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--log_batch", type=int, default=10)
    parser.add_argument("--multi_gpu", action="store_true")
    parser.add_argument("--log_txt", action="store_true")
    parser.add_argument("--stem_width", type=int, default=64)
    parser.add_argument("--cudnn", action="store_true")
    parser.add_argument("--tag", type=str, default="")
    parser.add_argument("--polar", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--flip", action="store_true")
    parser.add_argument("--finetune", action="store_true")
    parser.add_argument("--distance_type", type=str, default="euclidean")
    parser.add_argument("--margin", type=float, default=None)

    main(parser.parse_args())
