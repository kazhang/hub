---
layout: hub_detail
background-class: hub-background
body-class: hub
title: RegNet
summary: A convolutional network design space with simple, regular models.
category: researchers
image: regnet.png
author: Pytorch Team
tags: [vision, scriptable]
github-link: https://github.com/pytorch/vision/blob/main/torchvision/models/regnet.py
github-id: pytorch/vision
featured_image_1: regnet.png
featured_image_2: no-image
accelerator: cuda-optional
order: 10
---

```python
import torch
model = torch.hub.load('pytorch/vision:v0.11.0', 'regnet_x_400mf', pretrained=True)
model.eval()
```

All pre-trained models expect input images normalized in the same way,
i.e. mini-batches of 3-channel RGB images of shape `(3 x H x W)`, where `H` and `W` are expected to be at least `224`.
The images have to be loaded in to a range of `[0, 1]` and then normalized using `mean = [0.485, 0.456, 0.406]`
and `std = [0.229, 0.224, 0.225]`.

Here's a sample execution.

```python
# Download an example image from the pytorch website
import urllib
url, filename = ("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
try: urllib.URLopener().retrieve(url, filename)
except: urllib.request.urlretrieve(url, filename)
```

```python
# sample execution (requires torchvision)
from PIL import Image
from torchvision import transforms
input_image = Image.open(filename)
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

# move the input and model to GPU for speed if available
if torch.cuda.is_available():
    input_batch = input_batch.to('cuda')
    model.to('cuda')

with torch.no_grad():
    output = model(input_batch)
# Tensor of shape 1000, with confidence scores over Imagenet's 1000 classes
print(output[0])
# The output has unnormalized scores. To get probabilities, you can run a softmax on it.
probabilities = torch.nn.functional.softmax(output[0], dim=0)
print(probabilities)
```

```
# Download ImageNet labels
!wget https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt
```

```
# Read the categories
with open("imagenet_classes.txt", "r") as f:
    categories = [s.strip() for s in f.readlines()]

# Show top categories per image
top5_prob, top5_catid = torch.topk(probabilities, 5)

for i in range(top5_prob.size(0)):
    print(categories[top5_catid[i]], top5_prob[i].item())
```

### Model Description

RegNet models were proposed in [Designing Network Design Spaces](https://arxiv.org/abs/2003.13678v1).
Here we have the two families of the RegNet models: RegNetX and RegNetY. There are 7 models in each
family. The RegNet design space provides simple and fast networks that work well across a wide 
range of flop regimes.
Their 1-crop error rates on imagenet dataset with pretrained models are listed below.

|  Model structure  | Top-1 error | Top-5 error |
| ----------------- | ----------- | ----------- |
| regnet_x_400mf | 72.834 | 90.950 | 
| regnet_x_800mf | 75.212 | 92.348 |
| regnet_x_1_6gf | 77.040 | 93.440 |
| regnet_x_3_2gf | 78.364 | 93.992 |
| regnet_x_8gf   | 79.344 | 94.686 | 
| regnet_x_16gf  | 80.058 | 94.944 |
| regnet_x_32gf  | 80.622 | 95.248 |
| regnet_y_400mf | 74.046 | 91.716 |
| regnet_y_800mf | 76.420 | 93.136 |
| regnet_y_1_6gf | 77.950 | 93.966 |
| regnet_y_3_2gf | 78.948 | 94.576 |
| regnet_y_8gf   | 80.032 | 95.048 |
| regnet_y_16gf  | 80.424 | 95.240 |
| regnet_y_32gf  | 80.878 | 95.340 |

### References

 - [Designing Network Design Spaces](https://arxiv.org/abs/2003.13678v1)
