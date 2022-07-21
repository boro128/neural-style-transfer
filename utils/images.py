import torchvision

# values below come from:
# https://pytorch.org/vision/main/models/generated/torchvision.models.vgg19.html
IMAGENET1K_V1_MEAN = [0.485, 0.456, 0.406]
IMAGENET1K_V1_STD = [0.229, 0.224, 0.225]


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

    torchvision.utils.save_image(img, path)
