from PIL import Image
import os
from transformers.image_utils import load_image
from typing import Union, List

def pipeline(model,processor, device=None):
    """Create a pipeline for easy inference with BobVLM.
    
    Args:
        model_name (str): Name or path of the model
        device (str, optional): Device to load model on
        
    Returns:
        callable: A function that processes inputs and returns generated text
    """
    
    def process_image(image_input: Union[str, Image.Image]) -> Image.Image:
        """Process different types of image inputs into PIL Image.
        
        Args:
            image_input: Can be PIL Image, local path, or URL
            
        Returns:
            PIL.Image: Processed image
        """
        # If already PIL Image, return as is
        if isinstance(image_input, Image.Image):
            return image_input
        
        # Handle string inputs (paths or URLs)
        if isinstance(image_input, str):
            # Check if URL (simple check for http/https)
            if image_input.startswith(('http://', 'https://')):
                return load_image(image_input)
            
            # Try to load as local file
            if os.path.exists(image_input):
                return Image.open(image_input)
            
            raise ValueError(f"Image path not found: {image_input}")
            
        raise ValueError(f"Unsupported image input type: {type(image_input)}")
    
    def __call__(chat, images=None, max_new_tokens=100, temperature=0.05, attend_to_img_tokens=True, **kwargs):
        """Run inference with the BobVLM pipeline.
        
        Args:
            chat (list): List of chat messages
            images (Union[str, Image.Image, List[Union[str, Image.Image]]]): Image input(s)
            max_new_tokens (int): Maximum number of tokens to generate
            temperature (float): Sampling temperature
            attend_to_img_tokens (bool): Whether to attend to image tokens
            **kwargs: Additional arguments passed to model.generate()
            
        Returns:
            list: Generated text responses
        """
        
        # Process images if provided
        processed_images = None
        if images is not None:
            # Convert single image to list
            if not isinstance(images, list):
                images = [images]
            
            # Process each image
            processed_images = [process_image(img) for img in images]
        
        inputs = processor(text=chat, images=processed_images)
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            attend_to_img_tokens=attend_to_img_tokens,
            **kwargs
        )
        return processor.batch_decode(outputs)
    
    return __call__
