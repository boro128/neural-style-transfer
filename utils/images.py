import torch
import torchvision

from pathlib import Path

# values below come from:
# https://pytorch.org/vision/main/models/generated/torchvision.models.vgg19.html
IMAGENET1K_V1_MEAN = torch.tensor([0.485, 0.456, 0.406])
IMAGENET1K_V1_STD = torch.tensor([0.229, 0.224, 0.225])


def load_img(path, height=None, width=None):
    img = torchvision.io.read_image(path)
    img = img.float()

    if height is None or width is None:
        height, width = img.shape[1:]

    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((height, width)),
        torchvision.transforms.Normalize(0, 255),  # scaling to [0.0, 1.0]
        torchvision.transforms.Normalize(IMAGENET1K_V1_MEAN, IMAGENET1K_V1_STD)
    ])

    return transform(img)


def save_img(img, path):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Normalize(0, 1/IMAGENET1K_V1_STD),
        torchvision.transforms.Normalize(-IMAGENET1K_V1_MEAN, 1),
    ])

    img = transform(img)

    # create dir if not exists
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    torchvision.utils.save_image(img, path)


def load_resize_save_img(path, new_height, new_width=None):
    img = torchvision.io.read_image(path)
    img = img.float()

    if new_width is None:
        _, old_height, old_width = img.shape
        new_width = old_width * new_height / old_height

    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((new_height, new_width)),
        torchvision.transforms.Normalize(0, 255),  # scaling to [0.0, 1.0]
    ])
    img = transform(img)

    path_obj = Path(path)
    new_img_path = f"{path_obj.parent}/{path_obj.stem}_{new_height}p{path_obj.suffix}"

    torchvision.utils.save_image(img, new_img_path)
