# teensy-tiny-clip!

[readme](https://user-images.githubusercontent.com/113657678/190564375-64be5212-a168-4e16-98e6-5dfb5ade70f5.png)

## Usage

This teensy-tiny CLIP model uses the `sentence-transformers/all-MiniLM-L6-v2` model for its text encoder, from the `sentence-transformers` library. As such, you would need to install the `sentence-transformers` library.

`!pip install sentence-transformers`

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

saved_model_path = '../input/teensytinyclip/4M/pytorch_model.bin'
model = torch.jit.load(saved_model_path)
```

To get image-text pair logits:

```python
images = torch.randn(3,3,224,224)
text = ['Cat', 'A person flying a kite', 'River in a forest']
encoded_text = encode_text(text, 'cpu')

model(images, captions) # shape (3,3)

```

# Note

The model
