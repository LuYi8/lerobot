---
language: en
license: apache-2.0
tags:
- pytorch
- jax-conversion
- transformers
- resnet
- hil-serl
- Lerobot
- vision
- image-classification
library_name: pytorch
---

# JAX to PyTorch Converted Model (ResNet-10)

It's done in context of porting `HIL-SERL` paper code (https://hil-serl.github.io/) to `Lerobot` (https://github.com/Lerobot/lerobot).
The HF doesn't have ResNet-10 model, which could be pretty usefult for robotics tasks because of it's small size.
This model is converted from JAX to PyTorch, and the weights are preserved.
## Model Description

[Brief description of the original model and its purpose]

This model is a PyTorch port of the original JAX implementation. The conversion maintains
the original model's architecture and weights while making it accessible to PyTorch users.
The original model is from https://github.com/rail-berkeley/hil-serl/blob/7d17d13560d85abffbd45facec17c4f9189c29c0/serl_launcher/serl_launcher/utils/train_utils.py#L103.

## Model Details

- **Original Framework:** JAX
- **Target Framework:** PyTorch
- **Model Architecture:** ResNet-10 (4-stage ResNet with basic blocks)
- **Original Model:** HIL-SERL ResNet-10
- **Total Parameters:** 4,905,792 (~4.9M parameters)
- **Hidden Sizes:** [64, 128, 256, 512]
- **Input:** 3-channel RGB images (128x128)
- **Embedding Size:** 64

## Conversion Process

This model was converted using an automated JAX to PyTorch conversion pipeline, ensuring:
- Weight preservation
- Architecture matching
- Numerical stability


## Code

https://github.com/helper2424/resnet10


## Usage
```python
from transformers import AutoModel, AutoTokenizer
model = AutoModel.from_pretrained("helper2424/resnet10")
```

## Citation
```bibtex

@misc{resnet10,
   title = "Resnet10",
   author = "Eugene Mironov and Khalil Meftah and Adil Zouitine and Michel Aractingi and Ke Wang",
   month = jan,
   year = "2025",
   address = "Online",
   publisher = "Hugging Face",
   url = "https://huggingface.co/helper2424/resnet10",
}

```
