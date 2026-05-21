from PIL import Image
from torchvision import transforms

from .transforms_custom import *


def augment(image):
    raise NotImplementedError("Training-time augmentation is not needed in this prototype.")


def convert_img_to_tensor(image, force_one_channel=False):
    transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Grayscale(),
            transforms.ToTensor(),
        ]
    )
    return transform(image)
