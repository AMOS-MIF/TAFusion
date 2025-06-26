import os
from torchvision import transforms as transform

import utils


def mkdirs(paths):
    if isinstance(paths, str):
        os.makedirs(paths, exist_ok=True)
    else:
        for path in paths:
            os.makedirs(path, exist_ok=True)

def mri_transform():
    return transform.Compose([
        transform.Resize(256),
        transform.ToTensor(),
    ])


def val(name, img_fusion):
    output_path = './output/result/' + str(name)
    utils.save_image_test(img_fusion, output_path)
