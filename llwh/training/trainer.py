"""Model Trainer for the Hybrid Language-World Model."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import json
from datetime import datetime


class TextDataset(Dataset):
    """Simple text dataset for training."""
    
    def __init__(self, data_path, max_length=128):
        """
        Initialize dataset.
        
        Args:
            data_path: Path to training data file
            max_length: Maximum sequence length
        """
        self.max_length = max_length
        self.data = []
        
        # Load data
        if data_path.endswith('.txt'):
            with open(data_path, 'r') as f:
                lines = f.readlines()
                self.data = [line.strip() for line in lines if line.strip()]
        elif data_path.endswith('.json'):
            with open(data_path, 'r') as f:
                self.data = json.load(f)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text = self.data[idx]
        # Simple tokenization (character-level for demo)
        tokens = [ord(c) % 256 for c in text[:self.max_length]]
        # Pad to max_length
        tokens = tokens + [0] * (self.max_length - len(tokens))
        return torch.tensor(tokens, dtype=torch.long)


class ModelTrainer:
    """
    Trainer class for the Hybrid Language-World Model.
    Handles training, validation, saving, and exporting.
    """
    
    def __init__(self, model, device=None):
        """
        Initialize the trainer.
        
        Args:
            model: The model to train
            device: Device to train on (cpu/cuda)
        """
        self.model = model
        
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        self.model.to(self.device)
        
        self.optimizer = None
        self.scheduler = None
        self.training_history = []
    
    def setup_optimizer(self, learning_rate=0.001, optimizer_type='adam'):
        """
        Setup optimizer and learning rate scheduler.
        
        Args:
            learning_rate: Initial learning rate
            optimizer_type: Type of optimizer ('adam', 'sgd', 'adamw')
        """
        if optimizer_type.lower() == 'adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        elif optimizer_type.lower() == 'sgd':
            self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate, momentum=0.9)
        elif optimizer_type.lower() == 'adamw':
            self.optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate)
        else:
            raise ValueError("Unknown optimizer type: {}".format(optimizer_type))
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            factor=0.5, 
            patience=3
        )
    
    def train_epoch(self, train_loader, criterion):
        """
        Train for one epoch.
        
        Args:
            train_loader: DataLoader for training data
            criterion: Loss function
            
        Returns:
            avg_loss: Average loss for the epoch
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(train_loader):
            batch = batch.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            # For language model training, predict next token
            inputs = batch[:, :-1]
            targets = batch[:, 1:]
            
            outputs = self.model.language_model(inputs)
            
            # Compute loss
            loss = criterion(
                outputs.reshape(-1, outputs.shape[-1]),
                targets.reshape(-1)
            )
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss
    
    def validate(self, val_loader, criterion):
        """
        Validate the model.
        
        Args:
            val_loader: DataLoader for validation data
            criterion: Loss function
            
        Returns:
            avg_loss: Average validation loss
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(self.device)
                
                inputs = batch[:, :-1]
                targets = batch[:, 1:]
                
                outputs = self.model.language_model(inputs)
                
                loss = criterion(
                    outputs.reshape(-1, outputs.shape[-1]),
                    targets.reshape(-1)
                )
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss
    
    def train(self, train_data_path, epochs=10, batch_size=32, learning_rate=0.001,
              val_data_path=None, save_dir='./checkpoints', log_callback=None):
        """
        Full training loop.
        
        Args:
            train_data_path: Path to training data
            epochs: Number of epochs to train
            batch_size: Batch size
            learning_rate: Learning rate
            val_data_path: Optional validation data path
            save_dir: Directory to save checkpoints
            log_callback: Optional callback for logging
            
        Returns:
            training_history: List of training metrics
        """
        # Setup optimizer
        self.setup_optimizer(learning_rate=learning_rate)
        
        # Create datasets
        train_dataset = TextDataset(train_data_path)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        val_loader = None
        if val_data_path:
            val_dataset = TextDataset(val_data_path)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Loss function
        criterion = nn.CrossEntropyLoss(ignore_index=0)
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Training loop
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            if log_callback:
                log_callback("Epoch {}/{}".format(epoch + 1, epochs))
            
            # Train
            train_loss = self.train_epoch(train_loader, criterion)
            
            if log_callback:
                log_callback("  Train Loss: {:.4f}".format(train_loss))
            
            # Validate
            val_loss = None
            if val_loader:
                val_loss = self.validate(val_loader, criterion)
                if log_callback:
                    log_callback("  Val Loss: {:.4f}".format(val_loss))
                
                # Update scheduler
                self.scheduler.step(val_loss)
                
                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    checkpoint_path = os.path.join(save_dir, 'best_model.pt')
                    self.save_checkpoint(checkpoint_path, epoch, val_loss)
                    if log_callback:
                        log_callback("  Saved best model!")
            
            # Save checkpoint
            if (epoch + 1) % 5 == 0:
                checkpoint_path = os.path.join(save_dir, 'checkpoint_epoch_{}.pt'.format(epoch + 1))
                self.save_checkpoint(checkpoint_path, epoch, train_loss)
            
            # Record history
            history_entry = {
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'learning_rate': self.optimizer.param_groups[0]['lr'],
                'timestamp': datetime.now().isoformat()
            }
            self.training_history.append(history_entry)
        
        if log_callback:
            log_callback("Training completed!")
        
        return self.training_history
    
    def save_checkpoint(self, path, epoch, loss):
        """
        Save a training checkpoint.
        
        Args:
            path: Path to save checkpoint
            epoch: Current epoch
            loss: Current loss
        """
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
            'training_history': self.training_history,
        }, path)
    
    def load_checkpoint(self, path):
        """
        Load a training checkpoint.
        
        Args:
            path: Path to checkpoint
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if self.optimizer and 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'training_history' in checkpoint:
            self.training_history = checkpoint['training_history']
        return checkpoint
    
    def export_model(self, export_path, export_format='pytorch'):
        """
        Export the trained model.
        
        Args:
            export_path: Path to export model
            export_format: Format to export ('pytorch', 'onnx')
        """
        if export_format == 'pytorch':
            self.model.save_model(export_path)
        elif export_format == 'onnx':
            sample_input = torch.randint(0, 100, (1, 10)).to(self.device)
            self.model.export_to_onnx(export_path, sample_input)
        else:
            raise ValueError("Unknown export format: {}".format(export_format))
    
    def save_training_history(self, path):
        """
        Save training history to JSON.
        
        Args:
            path: Path to save history
        """
        with open(path, 'w') as f:
            json.dump(self.training_history, f, indent=2)


def main():
    """Main entry point for command-line training."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Hybrid Language-World Model')
    parser.add_argument('--data', type=str, required=True, help='Path to training data')
    parser.add_argument('--val-data', type=str, help='Path to validation data')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--save-dir', type=str, default='./checkpoints', help='Save directory')
    
    args = parser.parse_args()
    
    # Import here to avoid circular dependencies
    from ..core.hybrid_model import HybridLanguageWorldModel
    
    print("Initializing model...")
    model = HybridLanguageWorldModel()
    
    print("Initializing trainer...")
    trainer = ModelTrainer(model)
    
    print("Starting training...")
    trainer.train(
        train_data_path=args.data,
        val_data_path=args.val_data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        save_dir=args.save_dir,
        log_callback=print
    )
    
    print("Training complete!")


if __name__ == '__main__':
    main()
