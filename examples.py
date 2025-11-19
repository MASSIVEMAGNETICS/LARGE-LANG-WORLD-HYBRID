"""Example script demonstrating the Hybrid AI system."""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llwh.core import HybridLanguageWorldModel
from llwh.training import ModelTrainer
from llwh.agents import PipelineBuilder
from llwh.models import AIChatManager
import torch


def example_basic_model():
    """Example: Create and use the basic model."""
    print("\n=== Example 1: Basic Model Usage ===")
    
    # Create model
    model = HybridLanguageWorldModel()
    print("Model created successfully!")
    
    # Create sample input
    sample_input = torch.randint(0, 100, (1, 10))
    
    # Forward pass
    output = model(sample_input)
    print("Model output shape:", output.shape)
    
    # Generate text
    generated = model.language_model.generate(sample_input, max_length=20)
    print("Generated sequence shape:", generated.shape)


def example_pipeline():
    """Example: Create and execute a pipeline."""
    print("\n=== Example 2: Pipeline Builder ===")
    
    # Create pipeline
    pipeline = PipelineBuilder()
    
    # Add blocks
    input_block = pipeline.add_block('text_input', config={'text': 'Hello AI!'})
    process_block = pipeline.add_block('language_processing')
    reason_block = pipeline.add_block('reasoning')
    output_block = pipeline.add_block('output', config={'keys': ['reasoning_result']})
    
    # Connect blocks
    pipeline.connect_blocks(input_block.id, process_block.id)
    pipeline.connect_blocks(process_block.id, reason_block.id)
    pipeline.connect_blocks(reason_block.id, output_block.id)
    
    # Visualize
    print(pipeline.visualize())
    
    # Execute
    result = pipeline.execute()
    print("\nPipeline execution result:")
    print("Final output:", result.get('final_output'))


def example_ai_chat():
    """Example: AI-to-AI conversation."""
    print("\n=== Example 3: AI-to-AI Chat ===")
    
    # Create chat manager
    manager = AIChatManager()
    
    # Add agents (using None as placeholder model for demo)
    manager.add_agent('AI-Alpha', None, 'hybrid')
    manager.add_agent('AI-Beta', None, 'language')
    
    # Start conversation
    conversation = manager.start_conversation(
        ['AI-Alpha', 'AI-Beta'],
        initial_message='What is the meaning of intelligence?',
        max_turns=5
    )
    
    print("\nConversation:")
    for turn in conversation:
        print("[Turn {}] {}: {}".format(
            turn['turn'], turn['agent'], turn['message'][:100]
        ))


def example_training_setup():
    """Example: Setup training (without actual training)."""
    print("\n=== Example 4: Training Setup ===")
    
    # Create model
    model = HybridLanguageWorldModel()
    
    # Create trainer
    trainer = ModelTrainer(model)
    
    # Setup optimizer
    trainer.setup_optimizer(learning_rate=0.001)
    
    print("Trainer initialized successfully!")
    print("Device:", trainer.device)
    print("Optimizer:", type(trainer.optimizer).__name__)


def example_model_save_load():
    """Example: Save and load model."""
    print("\n=== Example 5: Model Save/Load ===")
    
    # Create model
    model = HybridLanguageWorldModel()
    
    # Save model
    save_path = '/tmp/test_model.pt'
    model.save_model(save_path)
    print("Model saved to:", save_path)
    
    # Load model
    loaded_model = HybridLanguageWorldModel.load_model(save_path)
    print("Model loaded successfully!")
    
    # Clean up
    if os.path.exists(save_path):
        os.remove(save_path)


def main():
    """Run all examples."""
    print("=" * 60)
    print("LARGE LANGUAGE-WORLD HYBRID AI - EXAMPLES")
    print("=" * 60)
    
    try:
        example_basic_model()
        example_pipeline()
        example_ai_chat()
        example_training_setup()
        example_model_save_load()
        
        print("\n" + "=" * 60)
        print("All examples completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print("\nError running examples:", str(e))
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
