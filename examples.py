"""Example script demonstrating the Hybrid AI system."""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llwh.core import HybridLanguageWorldModel
from llwh.training import ModelTrainer
from llwh.agents import PipelineBuilder
from llwh.models import AIChatManager, ReviewAssessor
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


def example_review_assessment():
    """Example: Review Assessment."""
    print("\n=== Example 6: Review Assessment ===")
    
    # Create review assessor
    assessor = ReviewAssessor()
    
    # Sample reviews to assess
    review1 = """This is an excellent product! The quality is outstanding and 
    it exceeded all my expectations. I would highly recommend this to anyone 
    looking for a reliable solution. The design is brilliant and the 
    functionality is perfect. Five stars!"""
    
    review2 = """Bad product not good waste money."""
    
    review3 = """The product is okay. It has some good features but also some 
    issues. The price is reasonable. I think it could be improved in several 
    areas, particularly the user interface. Overall, it's decent for the price."""
    
    # Assess individual reviews
    print("\n--- Assessing Review 1 ---")
    assessment1 = assessor.assess_review(review1, include_details=True)
    print("Overall Score:", assessment1['overall_score'])
    print("Recommendation:", assessment1['recommendation'])
    print("Quality Score:", assessment1['quality_score'])
    print("Sentiment Score:", assessment1['sentiment_score'])
    print("Positive Indicators:", assessment1['details']['positive_indicators'])
    
    print("\n--- Assessing Review 2 ---")
    assessment2 = assessor.assess_review(review2)
    print("Overall Score:", assessment2['overall_score'])
    print("Recommendation:", assessment2['recommendation'])
    
    print("\n--- Assessing Review 3 ---")
    assessment3 = assessor.assess_review(review3)
    print("Overall Score:", assessment3['overall_score'])
    print("Recommendation:", assessment3['recommendation'])
    
    # Batch assessment
    print("\n--- Batch Assessment ---")
    batch_results = assessor.batch_assess([review1, review2, review3])
    for i, result in enumerate(batch_results, 1):
        print("Review {}: Score = {}".format(i, result['overall_score']))
    
    # Compare reviews
    print("\n--- Comparing Reviews ---")
    comparison = assessor.compare_reviews(review1, review2)
    print("Better Review:", comparison['better_review'])
    print("Score Difference:", comparison['score_difference'])
    
    # Get statistics
    print("\n--- Assessment Statistics ---")
    stats = assessor.get_assessment_statistics()
    print("Total Assessments:", stats['total_assessments'])
    print("Average Score:", stats['average_score'])
    print("Highest Score:", stats['highest_score'])
    print("Lowest Score:", stats['lowest_score'])


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
        example_review_assessment()
        
        print("\n" + "=" * 60)
        print("All examples completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print("\nError running examples:", str(e))
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
