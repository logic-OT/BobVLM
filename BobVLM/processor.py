
from typing import Optional, Union, Dict, List
from PIL import Image
import torch
from transformers import ProcessorMixin, AutoTokenizer
import json
from transformers import CLIPImageProcessor
import os

class BobVLMProcessor(ProcessorMixin):
    attributes = ["image_processor", "tokenizer"]

    def __init__(self):

        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        #Initialize image processor
        self.image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")
        # Required for ProcessorMixin
        self.current_processor = self

    def save_pretrained(self, save_directory: str, **kwargs):
        """Save processor configuration and components."""
        # Save tokenizer
        self.tokenizer.save_pretrained(save_directory)

        # Save image processor
        self.image_processor.save_pretrained(save_directory)

        # Save processor config
        config = {
            "tokenizer_name": "meta-llama/Llama-3.2-1B-Instruct",
            "image_processor_name": "openai/clip-vit-large-patch14"
        }

        config_path = os.path.join(save_directory, "processor_config.json")
        with open(config_path, 'w') as f:
            json.dump(config, f)

    @classmethod
    def from_dict(cls, config_dict, **kwargs):
        """Create config from dictionary."""
        config = cls()
        for key, value in config_dict.items():
            setattr(config, key, value)
        return config, kwargs

    def __call__(
            self,
            text: Optional[Union[str, List[str]]] = None,
            images: Optional[Union[Image.Image, List[Image.Image]]] = None,
            return_tensors: Optional[str] = 'pt',
            **kwargs
        ) -> Dict[str, torch.Tensor]:
            """Process images and text.

            Args:
                text: Single string or list of strings to process
                images: Single image or list of images to process
                return_tensors: Type of tensors to return ('pt' for PyTorch)
                **kwargs: Additional arguments passed to the tokenizer

            Raises:
                ValueError: If both images and text are provided but their lengths don't match
            """
            if images is None and text is None:
                raise ValueError(
                        "Please provide inputs for images and text"
                    )

            encoding = {}
            special_image_token = None
            if images is not None:
                if isinstance(images, Image.Image):
                    images = [images]
                special_image_symbol = "※"
                special_image_token = self.tokenizer(
                    special_image_symbol,
                    return_tensors=return_tensors,
                    add_special_tokens = False,
                ).input_ids[0]
                pixel_values = self.image_processor(images,return_tensors=return_tensors)['pixel_values']
                encoding["pixel_values"] = pixel_values


            if text is not None:
                # If text is a list containing a dictionary, convert it to a string using apply_chat_template
                if isinstance(text, list) and any(isinstance(item, dict) for item in text):
                    text = self.tokenizer.decode(self.tokenizer.apply_chat_template(text, add_generation_prompt=True))

                if isinstance(text, list) and any(isinstance(item, list) for item in text):
                    text = [self.tokenizer.decode(self.tokenizer.apply_chat_template(text, add_generation_prompt=True)) for text in text]

                if isinstance(text, str):
                    text = [text]

                if images is not None and len(images) != len(text):
                    raise ValueError(
                        f"Number of images ({len(images)}) must match number of text entries ({len(text)}) "
                        "when both are provided."
                    )


                text_encoding = self.tokenizer(
                    text,
                    return_tensors=return_tensors,
                    padding = "max_length",
                    max_length = 249,
                    truncation=True,
                    **kwargs
                )
                if special_image_token:
                    #adding special token to each array
                    special_tokens_column = torch.full(
                        (text_encoding['input_ids'].size(0), 1),
                        float(special_image_token[0]),
                        device=text_encoding['input_ids'].device,
                        dtype=text_encoding['input_ids'].dtype
                    )
                    attention_mask = text_encoding['attention_mask']

                    # Concatenate along the sequence dimension
                    encoding['input_ids'] = torch.cat([special_tokens_column,text_encoding['input_ids']], dim=1)
                    # Create attention mask that includes the special token but not padding
                    attention_mask_special = torch.zeros(attention_mask.size()[0],attention_mask.size()[1]+1)
                    seq_lengths = attention_mask.sum(dim=1)

                    # Set attention to original content and special token
                    for i in range(len(seq_lengths)):
                        attention_mask_special[i, :seq_lengths[i]] = 1  # Original content

                    encoding['attention_mask'] = attention_mask_special
                else:
                    encoding['input_ids'] = text_encoding['input_ids']
                    encoding['attention_mask'] = text_encoding['attention_mask']
            return encoding

    def batch_decode(self, *args, **kwargs):
        """Decode token ids to text."""
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """Decode token ids to text."""
        return self.tokenizer.decode(*args, **kwargs)

    @property
    def model_input_names(self):
        """Names of the inputs expected by the model."""
        return ["input_ids", "attention_mask", "pixel_values"]

    def apply_chat_template(self, messages:Optional[List[str]]):
        '''
        Apply chat template to test
        returns: str
        '''
        return self.tokenizer.apply_chat_template(messages)