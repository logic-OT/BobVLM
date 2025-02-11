import torch
from torch.utils.data import Dataset, DataLoader
from transformers.image_utils import load_image
import os
from tqdm import tqdm
import logging
from datetime import datetime
import json
from PIL import Image
import numpy as np
#from BobVLM.model import BobVLMProcessor,BobVLMConfig,BobVLM
import functools
from IPython.display import Javascript
from threading import Thread
#from peft import get_peft_config,get_peft_model,LoraConfig,TaskType, PeftModel


    
def load_adapter_weights(model, checkpoint_path):
    """
    Load adapter weights from a checkpoint file.
    
    Args:
        model: The model containing the adapter
        checkpoint_path (str): Path to the checkpoint file
    """
    try:
        # Load checkpoint with CPU/GPU handling
        if torch.cuda.is_available():
            checkpoint = torch.load(checkpoint_path)
        else:
            checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        
        # Load adapter state dict
        model.adapter.load_state_dict(checkpoint['adapter_state_dict'])
        
        # Log information about the checkpoint
        print(f"Successfully loaded checkpoint from step {checkpoint['step']}")
        print(f"Checkpoint loss: {checkpoint['loss']:.4f}")
        
        # Verify adapter parameters are loaded
        trainable_params = sum(p.numel() for p in model.adapter.parameters())
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Adapter parameters loaded: {trainable_params:,}")
        print(f"Total parameters: {total_params:,}")
        
        return model
        
    except Exception as e:
        logger.error(f"Error loading checkpoint: {str(e)}")
        return False
        

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class VLMDataset(Dataset):
    def __init__(self, data_dir, processor):
        """
        Args:
            data_dir (str): Directory containing the dataset
            processor: BobVLMProcessor instance
        """
        self.processor = processor
        with open(data_dir, 'r') as f:
            self.data = json.load(f)[5000*4:]
        # Load your dataset here
        # Example structure:
        # self.data = [
        #     {"image_path": "path/to/image.jpg", "role": "description"}
        # ]

    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):
        item = self.data[idx]
        image = load_image(item['image_path'])
        inputs = self.processor(
            text=item["conversations"],
            images=image,
            return_tensors="pt",
        )
        inputs = prepare_labels(inputs)

        for k in inputs:
            if isinstance(inputs[k], torch.Tensor):
                inputs[k] = inputs[k].squeeze(0)
        return inputs

def prepare_labels(inputs):
    #Edit the labels such that the image token id is appended to the beginning of the original ids so they match for loss calc
    special_image_token_id = 64780 # the id corresponds to
    num_image_tokens = 257

    # Clone the input_ids to avoid modifying tensors in-place
    input_ids = inputs['input_ids'].clone()

    special_tokens_column = torch.full(
        (input_ids.size(0), num_image_tokens),
        float(special_image_token_id),
        device=input_ids.device,
        dtype=input_ids.dtype
    )

    # Create new tensor for labels
    inputs['labels'] = torch.cat([special_tokens_column, input_ids], dim=1).detach()

    return inputs

class AdapterTrainer:
    def __init__(
        self,
        model,
        train_dataloader,
        val_dataloader=None,
        learning_rate=1e-4,
        save_steps=100,
        log_steps=10,
        checkpoint_dir="/kaggle/working/",
        device="cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.device = device
        self.save_steps = save_steps
        self.log_steps = log_steps
        self.checkpoint_dir = checkpoint_dir

        # Freeze vision and language models
        for param in self.model.vit.parameters():
            param.requires_grad = False
        for param in self.model.language_model.parameters():
            param.requires_grad = True
        

        # Only optimize adapter parameters
        self.optimizer = torch.optim.AdamW(
            self.model.adapter.parameters(),
            lr=learning_rate
        )

        # Setup TensorBoard
        # self.writer = SummaryWriter(
        #     log_dir=os.path.join("runs", datetime.now().strftime("%Y%m%d-%H%M%S"))
        # )

        # Create checkpoint directory
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )

    # In your trainer:
    
    def save_checkpoint(self, step, loss):
        file = f'LLM_checkpoint_step_{step}.pt'
        adapter_file = f'adapter_checkpoint_step_{step}.pt'
        path1 = os.path.join(self.checkpoint_dir, file)
        path2 = os.path.join(self.checkpoint_dir, adapter_file)
        
        checkpoint1 = {
            'step': step,
            'adapter_state_dict': self.model.language_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
        }

        checkpoint2 = {
            'step': step,
            'adapter_state_dict': self.model.adapter.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
        }
        
        
        torch.save(checkpoint1, path1)
        torch.save(checkpoint2, path2)
        
    def train(self, num_epochs):
        global_step = 0
        best_val_loss = float('inf')

        for epoch in range(num_epochs):
            self.model.train()
            epoch_loss = 0
            progress_bar = tqdm(self.train_dataloader, desc=f'Epoch {epoch+1}/{num_epochs}')

            for batch in progress_bar:
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}

                # Forward pass
                outputs = self.model(**batch)
                outputs['logits'].shape
                loss = outputs['loss']

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.adapter.parameters(), max_norm=1.0)
                self.optimizer.step()

                # Update metrics
                epoch_loss += loss.item()
                global_step += 1

                # Log metrics
                if global_step % self.log_steps == 0:
                    #self.writer.add_scalar('Loss/train', loss.item(), global_step)
                    progress_bar.set_postfix({'loss': loss.item()})

                # Save checkpoint
                if global_step % self.save_steps == 0:
                    self.save_checkpoint(global_step, loss.item())

            avg_epoch_loss = epoch_loss / len(self.train_dataloader)
            logger.info(f'Epoch {epoch+1} average loss: {avg_epoch_loss:.4f}')

            # Validation
            if self.val_dataloader is not None:
                val_loss = self.validate()
                #self.writer.add_scalar('Loss/val', val_loss, global_step)
                self.scheduler.step(val_loss)

                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.save_checkpoint(global_step, val_loss)
                    logger.info(f'New best validation loss: {val_loss:.4f}')
            break

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        total_loss = 0

        for batch in tqdm(self.val_dataloader, desc='Validating'):
            batch = {k: v.to(self.device) for k, v in batch.items()}
            # print(batch.shape)

            outputs = self.model(**batch)
            total_loss += outputs.loss.item()

        avg_loss = total_loss / len(self.val_dataloader)
        logger.info(f'Validation loss: {avg_loss:.4f}')
        return avg_loss

def main():
    # Initialize model and processor
    # initializing model for adapter training
    config = BobVLMConfig()
    model = BobVLM(config)
    model = load_adapter_weights(model, "/kaggle/input/checkpoints/pytorch/default/21/CLS_checkpoint_step_4000 (5).pt")

    # model = PeftModel.from_pretrained(
    #         model,
    #         "/kaggle/input/lora-checkpoints/pytorch/default/1/lora_checkpoint_step_6000",
    #         device_map={'': device},
    #     is_trainable=True
    #     )
    
    print('\n\n MODEL DESIGN\n',model)
    processor = BobVLMProcessor()

    # Create datasets and dataloaders
    train_dataset = VLMDataset("/kaggle/input/llava-instruct/llava_50k_1.json", processor)
    val_dataset = VLMDataset("/kaggle/input/sample/sample_data.json", processor)
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # Initialize trainer
    trainer = AdapterTrainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        learning_rate=1e-4,
        save_steps=1500,  # Save every 1000 steps
        log_steps=10,    # Log every 10 steps
    )

    # Start training
    trainer.train(num_epochs=1)

if __name__ == "__main__":
    main()
