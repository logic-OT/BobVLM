# BobVLM ✨👀
[[Article on Medium🔥](https://medium.com/@uthmant14/building-a-custom-vision-language-model-with-hugging-face-7d51099bfbcb)] [[Model on huggingface🤗](https://huggingface.co/selfDotOsman/BobVLM-1.5b)] 

BobVLM is an ambitious passion project that experiments pre-training a good multimodal language model on limited resources and hardware and still achieve impressive performance. The result is a 1.5b model pre-trained on P100 GPU that is capable of detailed image description and moderate question answering.

***Don't forget to star the repo if you find this useful*** 🌟

## Model Architecture 🔧
![Bob VLM diagram](https://github.com/user-attachments/assets/e212fb6b-d8c5-4c33-8170-753ec05979a3)


### Training Approach 📖

To maintain efficiency and accessibility:
- Vision and language components are frozen
- **Only the adapter layer is trained**
- **Supervised training approach, treating adapter training as model finetuning**(Houlsby et al. (2019)'s work on MLP adapters for transfer learning)
- Can be trained on accessible hardware (T4 or P100 GPUs)

## Model Shortcomings 🥲
1. Training Data (coco mainly). The model was train on mainly coco images. Therefore the model thrives with scenes rather than close up images or animations
2. Question Answering: Question answering is just okay. Its not the most accurate


## Installation 🔧
Make sure you run with GPU/cuda. Works on Colab or any other service.
<br><br>You can install the package directly from GitHub:

```bash
pip install git+https://github.com/logic-ot/BobVLM.git
```

or in a notebook

```
!pip install git+https://github.com/logic-ot/BobVLM.git
```

## Usage

### Basic Usage

```python
from BobVLM import BobVLMProcessor, load_model, pipeline

# Load model and processor
model = load_model()
processor = BobVLMProcessor()

# Create pipeline
pipe = pipeline(model, processor)

# Example with URL image and system prompt
response = pipe(
    chat=[
        {"role": "system", "content": "You are an image understanding assistant. You can see and interpret images in fine detail"},
        {"role": "user", "content": "What's in this image?"},
    ],
    images="http://images.cocodataset.org/train2017/000000436349.jpg"
)

print(response)
```
**Model Output**
```
The image shows a large group of trucks parked in a parking lot, with a variety of vehicles, including semi-trucks, buses, and vans, all lined up in a neat and organized manner. The trucks are parked in a row, with some of them having their doors open, while others are closed. The vehicles are all yellow, with some having white or black stripes.<|eot_id|>'
```
### Different Input Types

```python
# 1. Local file
response = pipe(
    chat=[{"role": "user", "content": "Describe this image"}],
    images="path/to/your/image.jpg"
)

# 2. PIL Image
from PIL import Image
image = Image.open("your_image.jpg")
response = pipe(
    chat=[{"role": "user", "content": "What do you see?"}],
    images=image
)
```

### Multiple Images

```python
# You can pass multiple images
response = pipe(
    chat=[{"role": "user", "content": "Compare these images"}],
    images=["image1.jpg", "https://example.com/image2.jpg"]
)
```

### Chat with Context

```python
# Chat with context
messages = [
    {"role": "system", "content": "You are an expert at analyzing images in detail."},
    {"role": "user", "content": "What's in this image?"},
    {"role": "assistant", "content": "I see a dog playing in a park."},
    {"role": "user", "content": "What breed is it?"}
]

response = pipe(
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


## License

This project is licensed under the MIT License - see the LICENSE file for details.
