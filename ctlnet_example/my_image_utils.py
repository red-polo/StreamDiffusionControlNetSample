from typing import List, Optional, Tuple, Union

import numpy as np
import PIL.Image
import torch
import torchvision


def process_image(
    image_pil: PIL.Image.Image, range: Tuple[int, int] = (-1, 1)
) -> Tuple[torch.Tensor, PIL.Image.Image]:
    # image = torchvision.transforms.ToTensor()(image_pil).cuda()
    #print(image.shape)
    image_pil = np.array(image_pil)
    #print(image_pil.shape)
    image = torch.from_numpy(image_pil).cuda()
    image = image.permute(2,0,1)
    image = image /255
    #print(image.shape)
    
    r_min, r_max = range[0], range[1]
    image = image * (r_max - r_min) + r_min
    return image[None, ...], image_pil


def pil2tensor(image_pil: PIL.Image.Image) -> torch.Tensor:
    height = image_pil.height
    width = image_pil.width
    imgs = []
    img, _ = process_image(image_pil,[0,1]) ####  ctlnetは0-1
    imgs.append(img)
    imgs = torch.vstack(imgs)
    images = torch.nn.functional.interpolate(
        imgs, size=(height, width), mode="bilinear"
    )
    image_tensors = images.to(torch.float16)
    return image_tensors
