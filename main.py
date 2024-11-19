from torchvision import models
from utils import parse_args, train
import torch
import torch.nn as nn


def main(args):
    model_type = args.model_type.lower()
    if model_type.startswith("convnext"):
        if model_type == "convnext_tiny":
            model = models.convnext_tiny(weights="DEFAULT")
        elif model_type == "convnext_small":
            model = models.convnext_small(weights="DEFAULT")
        elif model_type == "convnext_base":
            model = models.convnext_base(weights="DEFAULT")
        elif model_type == "convnext_large":
            model = models.convnext_large(weights="DEFAULT")

    elif model_type == "resnet50":
        model = models.resnet50(pretrained=True)
        model.fc = nn.Identity()
        args.stem_width = 50

    if args.state:
        try:
            model.load_state_dict(torch.load(args.state, map_location=None if args.cuda else torch.device("cpu")))
            print("model state loaded")

        except AssertionError:
            print("assertion error")
            model.load_state_dict(torch.load(args.state, map_location=lambda storage, loc: storage))

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

    if args.mode == "train":
        train(args, model)


if __name__ == "__main__":
    args = parse_args()
    main(args)
