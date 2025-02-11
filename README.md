# BobVLM

BobVLM is a friendly 1.5 billion parameter Vision Language Model that combines powerful image understanding with natural language capabilities. Named with a touch of humor (partly from the author's initials and partly because it sounds like your friendly neighborhood AI), BobVLM excels at detailed image captioning and visual question answering.

## Model Architecture
![Bob VLM diagram](https://github.com/user-attachments/assets/e212fb6b-d8c5-4c33-8170-753ec05979a3)

BobVLM consists of three main components:

1. **Vision Encoder**: Uses CLIP (clip-vit-large-patch14) for processing images into meaningful embeddings
2. **Language Model**: Employs LLaMA (llama-3.2-1.3b) for generating coherent and contextually relevant text
3. **Adapter Layer**: A 1.5M parameter MLP projection layer that acts as a "bilingual translator" between CLIP's embeddings and LLaMA's token space
   - 3 layers with 500 nodes each
   - 257 adapted image embeddings (following CLIP's final hidden state)


### Training Approach

To maintain efficiency and accessibility:
- Vision and language components are frozen
- Only the adapter layer is trained
- Supervised training approach, treating adapter training as finetuning
- Inspired by Houlsby et al. (2019)'s work on MLP adapters for transfer learning
- Can be trained on accessible hardware (T4 or P100 GPUs)

## Features

- Supports multiple image input formats:
  - Local image files
  - Image URLs
  - PIL Image objects
- Detailed image captioning
- Visual question answering
- Flexible chat interface
- Built on the Hugging Face Transformers library
- Resource-efficient design

## Installation

You can install the package directly from GitHub:

```bash
pip install git+https://github.com/selfDotOsman/BobVLM.git
```

Or directly from Hugging Face:

```python
from transformers import AutoModelForVision2Seq
model = AutoModelForVision2Seq.from_pretrained("selfDotOsman/BobVLM-1.5b", trust_remote_code=True)
```

## Usage

### Basic Usage

```python
from BobVLM import pipeline

# Create pipeline
vlm = pipeline()

# You can use different types of image inputs:
# 1. Local file
response = vlm(
    chat=[{"role": "user", "content": "What's in this image?"}],
    images="path/to/your/image.jpg"
)

# 2. Image URL
response = vlm(
    chat=[{"role": "user", "content": "Describe this image"}],
    images="https://example.com/image.jpg"
)

# 3. PIL Image
from PIL import Image
image = Image.open("your_image.jpg")
response = vlm(
    chat=[{"role": "user", "content": "What do you see?"}],
    images=image
)

print(response)
```

### Multiple Images

```python
# You can pass multiple images
response = vlm(
    chat=[{"role": "user", "content": "Compare these images"}],
    images=["image1.jpg", "https://example.com/image2.jpg"]
)
```

### Chat with Context

```python
# Chat with context
messages = [
    {"role": "user", "content": "What's in this image?"},
    {"role": "assistant", "content": "I see a dog playing in a park."},
    {"role": "user", "content": "What breed is it?"}
]

response = vlm(
    chat=messages,
    images="dog.jpg"
)
```

## Requirements

- Python 3.7+
- transformers
- torch
- Pillow
- requests

## Model Card

For more detailed information about the model, visit the [Hugging Face model page](https://huggingface.co/selfDotOsman/BobVLM-1.5b).

## Citation

If you use BobVLM in your research, please cite:

```bibtex
@misc{bobvlm2024,
  author = {selfDotOsman},
  title = {BobVLM: A Lightweight Vision Language Model with Efficient Adapter Architecture},
  year = {2024},
  publisher = {Hugging Face},
  howpublished = {\url{https://huggingface.co/selfDotOsman/BobVLM-1.5b}}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
