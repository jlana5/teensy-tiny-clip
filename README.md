# teensy-tiny-clip

![readme](https://user-images.githubusercontent.com/113657678/190564375-64be5212-a168-4e16-98e6-5dfb5ade70f5.png)

# Usage

These are two 4 million and 5 million parameter [MLP-Mixer]([http://](https://arxiv.org/abs/2105.01601)) models trained on the [CLIP](https://arxiv.org/abs/2103.00020) objective on a very small subset of the [COCO 2017 dataset](https://arxiv.org/abs/1405.0312). The teensy-tiny CLIP model uses a pretrained `sentence-transformers/all-MiniLM-L6-v2` model for its text encoder, from the [`sentence-transformers`](https://www.sbert.net/) library. As such, you would need to install the `sentence-transformers` library.

```python
!pip install sentence-transformers
```
The model takes in an image-text_embedding pair, so I recommend the following function for encoding text:

```python
from sentence_transformers import SentenceTransformer

text_encoder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def encode_text(text, device):
    embeddings = text_encoder.encode(text, show_progress_bar=False, convert_to_tensor=True, device=device)
    return embeddings.squeeze()
```

To load the model:
```python
import torch

saved_model_path = 'clip-mixer-5M.bin'
model = torch.jit.load(saved_model_path)
model.eval()
```

Resize and normalize the image first, as the model was trained on size 224 and normalized to ImageNet mean and std:

```python
preprocess = trnsforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)),
])

image = preprocess(image)
```


To get image-text pair logits:

```python
images = torch.randn(3,3,224,224)

text = ['Cat', 'A person flying a kite', 'River in a forest']
encoded_text = encode_text(text, 'cpu')

model(images, encoded_text) # shape (3,3)

```

You can also use the model for feature extraction. The model maps the image to a 384-dimensional vector:
```python
image = torch.randn(1,3,224,224)
features = model.image_encoder(image) # shape (1, 384)
```

# Zero-shot performance

| | Model | CIFAR-10 | [ImageNette](https://github.com/fastai/imagenette) | [Natural Images](https://arxiv.org/abs/1807.10108) | CIFAR-100 | Caltech-256 |
| --- | --- | --- | --- | --- | --- | --- |
| Top-1 accuracy | `clip-mixer-4M` | 20.02% | 16.88% | 22.58% | 1.85% | 0.47% |
| Top-5 accuracy | `clip-mixer-4M` | 73.11% | 59.35% | 81.19% | 7.86% | 3.39% |
| Top-1 accuracy | `clip-mixer-5M` | 19.83% | 16.78% | 24.48% | 2.00% | 0.47% |
| Top-5 accuracy | `clip-mixer-5M` | 71.96% | 59.68% | 79.66% | 7.83% | 3.37% |

Inferring from the zero-shot performance, the model seems to perform better on datasets with smaller number of classes and more distinct classes.

# Training details

`clip-mixer-4M` was trained on ~51k images with a batch size of `512`, while `clip-mixer-5M` was trained on ~57K images with a batch size of `384`.
