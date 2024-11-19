from PIL import Image
from arcface_torch import ArcFace
from dataset import ImageDataset
from functools import partial
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision.models.convnext import LayerNorm2d
from torchvision.transforms import Compose, Normalize, Resize, ToTensor
import numpy as np
import os
import sys
import torch
import torch.nn as nn


def train(args, model):
    input_shape = (64, 512) if args.polar else (256, 256)
    input_transform = Compose([Resize(input_shape), ToTensor(), Normalize(mean=(0.5,), std=(0.5,))])

    if args.log_txt:
        sys.stdout = open(f"{args.tag}_{args.model_type.lower()}_{args.stem_width}_{args.distance_type}_trainer_output.txt", "a")

    directory = os.path.dirname(f"./{args.tag}_{args.model_type.lower()}_{args.stem_width}_{args.distance_type}_checkpoint")

    if not os.path.exists(directory):
        os.makedirs(directory)

    print(f"args.image_dir: {args.image_dir}")
    print(f"input_transform: {input_transform}")
    print("------------loader beginning-----------", flush=True)

    dataset = ImageDataset(args.image_dir, args.img_uid_map, args.polar, input_transform, True, args.flip, True)

    if any(v is None for v in {args.val_image_dir, args.val_img_uid_map}):
        val_length = len(dataset) // 20
        train_length = len(dataset) - val_length
        train_dataset, val_dataset = random_split(dataset, [train_length, val_length], generator=torch.Generator().manual_seed(42))

        print(f"len(train_dataset): {len(train_dataset)}")
        print(f"len(val_dataset): {len(val_dataset)}")

        loader = DataLoader(train_dataset, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=True)

        print("loader complete")
        print("-----------val loader beginning--------", flush=True)

        val_loader = DataLoader(val_dataset, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False)

        print("val_loader complete", flush=True)

    else:
        loader = DataLoader(dataset, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=True)
        val_dataset = ImageDataset(args.val_image_dir, args.val_img_uid_map, args.polar, input_transform, False, False, True)
        val_loader = DataLoader(val_dataset, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False)

    print("training length:", len(loader))
    print("validation length:", len(val_loader), flush=True)

    ################ Set model classifier ################
    lastconv_output_channels = {
        "convnext_tiny" :  768,
        "convnext_small":  768,
        "convnext_base" : 1024,
        "convnext_large": 1536,
    }

    norm_layer = partial(LayerNorm2d, eps=1e-6)
    model_type = args.model_type.lower()
    output_channels = lastconv_output_channels[model_type]

    print("Getting num_classes...", flush=True)
    num_classes = len(set(label for _, label in dataset))
    print(f"num_classes: {num_classes}", flush=True)

    model.classifier = nn.Sequential(norm_layer(output_channels), nn.Flatten(), nn.Linear(output_channels, num_classes))

    optimizer = torch.optim.SGD(model.parameters(), lr = args.lr, momentum=0.9, weight_decay=0.0005)
    ######################################################

    best_val_loss_average = float("inf")
    margin_loss = ArcFace()

    if args.cuda:
        scaler = torch.GradScaler()
        margin_loss = margin_loss.cuda()

    if args.margin is None:
        args.margin = 1.0 if args.distance_type == "euclidean" else 0.05

    for epoch in range(1, args.num_epochs + 1):
        model.train()
        epoch_loss = []

        for batch, data in enumerate(loader):
            # Setup input
            images = data["images"]
            images = images.repeat(1, 3, 1, 1)

            if args.debug:
                if not os.path.exists("debug"):
                    os.mkdir("debug")

                for i in range(images.shape[0]):
                    image = images[i][0].clone().detach().cpu().numpy()
                    image = (image * 0.5) + 0.5
                    image = np.clip(image, 0, 1)
                    image = (image * 255).astype(np.uint8)
                    image_pil = Image.fromarray(image, "L")
                    image_pil.save(os.path.join("debug", f"{i}.png"))

                args.debug = False

            labels = data["labels"]

            if args.cuda:
                images = images.cuda()
                labels = labels.cuda()

                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    embeddings = model(images)
                    loss = margin_loss(embeddings, labels)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            else:
                embeddings = model(images)
                loss = margin_loss(embeddings, labels)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            # compute IoU
            epoch_loss.append(loss.item())

            if batch % args.log_batch == 0:
                print(f"Train loss: {sum(epoch_loss) / len(epoch_loss)} (epoch: {epoch}, batch: {batch}/{len(loader)})", flush=True)

            if args.log_txt:
                sys.stdout.close()
                sys.stdout = open(f"{args.tag}_{args.model_type.lower()}_{args.stem_width}_{args.distance_type}_trainer_output.txt", "a")

        # Validation set
        if len(val_loader) > 0:
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
                    val_loss = margin_loss(outputs, labels)
                    val_epoch_loss.append(val_loss.item())

                val_loss_average = sum(val_epoch_loss) / len(val_epoch_loss)

                print(f"Val loss: {val_loss_average} (epoch: {epoch})", flush=True)

                if args.log_txt:
                    sys.stdout.close()
                    sys.stdout = open(f"{args.tag}_{args.model_type.lower()}_{args.stem_width}_{args.distance_type}_trainer_output.txt", "a")

                if val_loss_average < best_val_loss_average:
                    # Save checkpoint
                    best_val_loss_average = val_loss_average
                    filename = os.path.join(directory, f"{args.model_type}-{epoch:03}-{round(val_loss_average, 6)}.pth")
                    torch.save(model.module.state_dict() if args.multi_gpu else model.state_dict(), filename)

            if args.log_txt:
                sys.stdout.close()
                sys.stdout = open(f"{args.tag}_{args.model_type.lower()}_{args.stem_width}_{args.distance_type}_trainer_output.txt", "a")

    if args.log_txt:
        sys.stdout.close()
