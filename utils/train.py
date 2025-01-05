from arcface_torch import CombinedMarginLoss
from dataset import ImageDataset
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision.transforms import Compose, Normalize, Resize, ToTensor
import numpy as np
import os
import sys
import torch


def train(args, model):
    input_shape = (64, 512) if args.polar else (256, 256)
    input_transform = Compose([Resize(input_shape), ToTensor(), Normalize(mean=(0.5,), std=(0.5,))])

    if args.log_txt:
        stdout_file = os.path.join(args.output_dir, f"{args.tag}_{args.model_type.lower()}_{args.stem_width}_{args.distance_type}_trainer_output.txt")
        if os.path.exists(stdout_file):
            os.remove(stdout_file)
        sys.stdout = open(stdout_file, "a")

    directory = os.path.dirname(os.path.join(args.output_dir, f"{args.tag}_{args.model_type.lower()}_{args.stem_width}_{args.distance_type}_checkpoint"))
    debug_dir = os.path.join(args.output_dir, "debug")

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


    best_val_loss_average = float("inf")

    # Classifier based on the number of identities in the dataset
    if args.cuda and torch.cuda.device_count() > 1 and args.multi_gpu:
        model.module.classifier[-1] = torch.nn.Linear(model.module.classifier[-1].in_features, dataset.num_classes).to("cuda")
    else:
        model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features, dataset.num_classes)
    optimizer = torch.optim.SGD(model.parameters(), lr = args.lr, momentum=0.9, weight_decay=0.0005)
    criterion = torch.nn.CrossEntropyLoss()
    combined_margin_loss = CombinedMarginLoss(64.0, 1.0, 0.5, 0.0)

    if args.cuda:
        scaler = torch.GradScaler()
        combined_margin_loss = combined_margin_loss.cuda()

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
                if not os.path.exists(debug_dir):
                    os.mkdir(debug_dir)

                for i in range(images.shape[0]):
                    image = images[i][0].clone().detach().cpu().numpy()
                    image = (image * 0.5) + 0.5
                    image = np.clip(image, 0, 1)
                    image = (image * 255).astype(np.uint8)
                    image_pil = Image.fromarray(image, "L")
                    image_pil.save(os.path.join(debug_dir, f"{i}.png"))

                args.debug = False

            labels = data["labels"]

            optimizer.zero_grad()
            if args.cuda:
                images = images.cuda()
                labels = labels.cuda()

                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    embeddings = model(images)
                    logits = combined_margin_loss(embeddings, labels)
                    loss = criterion(logits, labels)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            else:
                embeddings = model(images)
                logits = combined_margin_loss(embeddings, labels)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

            # compute IoU
            print(f"Batch: {batch}\nLoss: {loss.item()}")
            epoch_loss.append(loss.item())

            if batch % args.log_batch == 0:
                print(f"Train loss: {sum(epoch_loss)} / {len(epoch_loss)} = {sum(epoch_loss) / len(epoch_loss)} (epoch: {epoch}, batch: {batch}/{len(loader)})", flush=True)

            if args.log_txt:
                sys.stdout.close()
                sys.stdout = open(stdout_file, "a")

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

                    embeddings = model(images)
                    logits = combined_margin_loss(embeddings, labels)
                    loss = criterion(logits, labels)
                    val_epoch_loss.append(loss.item())

                val_loss_average = sum(val_epoch_loss) / len(val_epoch_loss)

                print(f"Val loss: {val_loss_average} (epoch: {epoch})", flush=True)

                if args.log_txt:
                    sys.stdout.close()
                    sys.stdout = open(stdout_file, "a")

                if val_loss_average < best_val_loss_average:
                    # Save checkpoint
                    best_val_loss_average = val_loss_average
                    filename = os.path.join(directory, f"{args.model_type}-{epoch:03}-{round(val_loss_average, 6)}.pth")
                    torch.save(model.module.state_dict() if args.multi_gpu else model.state_dict(), filename)

            if args.log_txt:
                sys.stdout.close()
                sys.stdout = open(stdout_file, "a")

    if args.log_txt:
        sys.stdout.close()
