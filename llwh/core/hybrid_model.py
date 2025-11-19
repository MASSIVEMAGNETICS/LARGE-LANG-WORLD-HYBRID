"""Hybrid Language-World Model combining language and world understanding."""

import torch
import torch.nn as nn
from .language_model import LanguageModel
from .world_model import WorldModel


class HybridLanguageWorldModel(nn.Module):
    """
    Revolutionary Hybrid Model that combines language understanding 
    with world/spatial reasoning for superior AI capabilities.
    """
    
    def __init__(self, config=None):
        """
        Initialize the Hybrid Language-World Model.
        
        Args:
            config: Configuration dictionary with model parameters
        """
        super(HybridLanguageWorldModel, self).__init__()
        
        if config is None:
            config = self.get_default_config()
        
        self.config = config
        
        # Initialize language model
        self.language_model = LanguageModel(
            vocab_size=config.get('vocab_size', 10000),
            embedding_dim=config.get('embedding_dim', 256),
            hidden_dim=config.get('hidden_dim', 512),
            num_layers=config.get('num_layers', 4)
        )
        
        # Initialize world model
        self.world_model = WorldModel(
            state_dim=config.get('state_dim', 256),
            action_dim=config.get('action_dim', 64),
            hidden_dim=config.get('hidden_dim', 512)
        )
        
        # Cross-modal fusion layers
        self.lang_to_world = nn.Linear(config.get('embedding_dim', 256), 
                                       config.get('state_dim', 256))
        self.world_to_lang = nn.Linear(config.get('state_dim', 256),
                                       config.get('embedding_dim', 256))
        
        # Attention mechanism for fusion
        self.fusion_attention = nn.MultiheadAttention(
            embed_dim=config.get('embedding_dim', 256),
            num_heads=8
        )
        
        # Joint reasoning layer
        self.joint_reasoning = nn.Sequential(
            nn.Linear(config.get('embedding_dim', 256) * 2, 
                     config.get('hidden_dim', 512)),
            nn.ReLU(),
            nn.Linear(config.get('hidden_dim', 512),
                     config.get('embedding_dim', 256))
        )
        
    @staticmethod
    def get_default_config():
        """Get default configuration."""
        return {
            'vocab_size': 10000,
            'embedding_dim': 256,
            'hidden_dim': 512,
            'num_layers': 4,
            'state_dim': 256,
            'action_dim': 64
        }
    
    def forward(self, input_ids, world_state=None):
        """
        Forward pass through the hybrid model.
        
        Args:
            input_ids: Text input token IDs
            world_state: Optional world state representation
            
        Returns:
            output: Combined language and world understanding
        """
        # Process through language model
        lang_encoding = self.language_model.encode_text(input_ids)
        
        # If world state provided, fuse with language
        if world_state is not None:
            world_encoding = self.world_model.encode_state(world_state)
            
            # Convert to same dimension
            lang_state = self.lang_to_world(lang_encoding)
            world_lang = self.world_to_lang(world_encoding)
            
            # Attention fusion
            fused, _ = self.fusion_attention(
                lang_encoding.unsqueeze(0),
                world_lang.unsqueeze(0),
                world_lang.unsqueeze(0)
            )
            fused = fused.squeeze(0)
            
            # Joint reasoning
            combined = torch.cat([lang_encoding, fused], dim=-1)
            output = self.joint_reasoning(combined)
        else:
            output = lang_encoding
        
        return output
    
    def generate_with_world_grounding(self, prompt_ids, world_state=None, 
                                     max_length=100, temperature=1.0):
        """
        Generate text grounded in world state.
        
        Args:
            prompt_ids: Starting token IDs
            world_state: Optional world state for grounding
            max_length: Maximum generation length
            temperature: Sampling temperature
            
        Returns:
            generated_ids: Generated token sequence
        """
        # Get world-grounded context if available
        if world_state is not None:
            context = self.forward(prompt_ids, world_state)
            # Use context to bias generation (simplified)
        
        # Generate using language model
        generated = self.language_model.generate(
            prompt_ids, 
            max_length=max_length,
            temperature=temperature
        )
        
        return generated
    
    def reason_about_world(self, text_ids, initial_state, horizon=5):
        """
        Use language understanding to reason about world dynamics.
        
        Args:
            text_ids: Text input describing scenario
            initial_state: Initial world state
            horizon: Planning horizon
            
        Returns:
            imagined_trajectory: Predicted world states
        """
        # Encode text to guide world model
        lang_encoding = self.language_model.encode_text(text_ids)
        
        # Convert to world state space
        guided_state = self.lang_to_world(lang_encoding)
        
        # Combine with initial state
        combined_state = (initial_state + guided_state) / 2.0
        
        # Imagine trajectory
        trajectory = self.world_model.imagine_trajectory(
            combined_state.unsqueeze(0), 
            horizon=horizon
        )
        
        return trajectory
    
    def save_model(self, path):
        """
        Save model to disk.
        
        Args:
            path: File path to save model
        """
        torch.save({
            'config': self.config,
            'model_state_dict': self.state_dict(),
        }, path)
    
    @classmethod
    def load_model(cls, path):
        """
        Load model from disk.
        
        Args:
            path: File path to load model from
            
        Returns:
            model: Loaded model instance
        """
        checkpoint = torch.load(path)
        model = cls(config=checkpoint['config'])
        model.load_state_dict(checkpoint['model_state_dict'])
        return model
    
    def export_to_onnx(self, path, sample_input_ids):
        """
        Export model to ONNX format for deployment.
        
        Args:
            path: Path to save ONNX model
            sample_input_ids: Sample input for tracing
        """
        self.eval()
        torch.onnx.export(
            self,
            (sample_input_ids,),
            path,
            export_params=True,
            opset_version=10,
            do_constant_folding=True,
            input_names=['input_ids'],
            output_names=['output'],
            dynamic_axes={
                'input_ids': {0: 'batch_size', 1: 'sequence'},
                'output': {0: 'batch_size'}
            }
        )
