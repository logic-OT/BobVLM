from transformers import AutoModelForVision2Seq
import torch

def load_model(model_name="selfDotOsman/BobVLM-1.5b", device=None):
    """Load the BobVLM model.
    
    Args:
        model_name (str): Name or path of the model
        device (str, optional): Device to load model on. If None, will use CUDA if available
        
    Returns:
        AutoModelForCausalLM: Loaded model
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = AutoModelForVision2Seq.from_pretrained(model_name, trust_remote_code=True).to(device)
    
    return model
