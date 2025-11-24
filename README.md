# Large Language-World Hybrid AI

A REVOLUTIONARY HYBRID AI SYSTEM combining Large Language Models with World Models for superior reasoning and understanding.

## Features

🚀 **Revolutionary Hybrid Architecture**
- Combines language understanding with spatial/world reasoning
- Dual-model approach for enhanced AI capabilities
- Cross-modal fusion for integrated intelligence

💻 **ChatGPT-Style GUI**
- User-friendly chat interface
- Real-time AI conversations
- Temperature control for response creativity
- Windows 7 compatible (uses tkinter)

🎓 **Comprehensive Training Suite**
- Train your own models
- Save and export trained models
- Progress tracking and visualization
- Support for custom datasets

🔧 **Pipeline Action Agent Builder**
- Visual workflow designer
- Drag-and-drop agent blocks
- Pre-built agent types (text, reasoning, API, file I/O, etc.)
- Save and load pipelines
- Execute complex multi-step workflows

🤖 **AI-to-AI Chat Interface**
- Multiple AI models interact
- Collaborative problem-solving
- Different collaboration strategies (round-robin, voting, consensus)
- Conversation export and analysis

⭐ **Review Assessment System**
- Comprehensive text review analysis
- Quality, sentiment, and coherence scoring
- Factual consistency checking
- Batch processing capabilities
- Detailed assessment reports and statistics

## System Requirements

- **Operating System**: Windows 7 or higher, Linux, macOS
- **Python**: 2.7 or 3.x
- **RAM**: Minimum 4GB (8GB recommended)
- **Storage**: 500MB free space

## Installation

### Quick Install

```bash
# Clone the repository
git clone https://github.com/MASSIVEMAGNETICS/LARGE-LANG-WORLD-HYBRID.git
cd LARGE-LANG-WORLD-HYBRID

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

### Windows 7 Installation

```cmd
# Ensure Python is installed (2.7 or 3.6+)
python --version

# Install dependencies
pip install -r requirements.txt

# Install package
python setup.py install
```

## Usage

### Launch GUI Application

```bash
# Using command-line entry point
llwh-gui

# Or directly with Python
python -m llwh.gui.main
```

### Training a Model

#### Using the GUI
1. Launch the GUI application
2. Go to the "Training Suite" tab
3. Select your training dataset
4. Configure training parameters (epochs, batch size, learning rate)
5. Click "Start Training"
6. Monitor progress in the training log
7. Save your trained model via File > Save Model

#### Using Command Line

```bash
# Train a model
llwh-train --data path/to/training_data.txt --epochs 10 --batch-size 32 --lr 0.001

# With validation data
llwh-train --data train.txt --val-data val.txt --epochs 20 --save-dir ./my_models
```

### Building Agent Pipelines

1. Go to the "Pipeline Builder" tab
2. Select agent blocks from the left panel
3. Click "Add Block" to add them to your pipeline
4. Connect blocks to create workflows
5. Save your pipeline for reuse
6. Click "Run Pipeline" to execute

### AI-to-AI Conversations

1. Go to the "AI-to-AI Chat" tab
2. Select two or more AI models
3. Enter a topic or problem
4. Set the number of conversation turns
5. Click "Start Conversation"
6. Watch AI models interact and solve problems collaboratively

## Programming Interface

### Using the Hybrid Model

```python
from llwh.core import HybridLanguageWorldModel
import torch

# Create a new model
model = HybridLanguageWorldModel()

# Generate text
prompt = torch.tensor([[1, 2, 3, 4, 5]])
generated = model.language_model.generate(prompt, max_length=50)

# Save model
model.save_model('my_model.pt')

# Load model
loaded_model = HybridLanguageWorldModel.load_model('my_model.pt')

# Export to ONNX
sample_input = torch.randint(0, 100, (1, 10))
model.export_to_onnx('model.onnx', sample_input)
```

### Training Programmatically

```python
from llwh.core import HybridLanguageWorldModel
from llwh.training import ModelTrainer

# Create model and trainer
model = HybridLanguageWorldModel()
trainer = ModelTrainer(model)

# Train
history = trainer.train(
    train_data_path='data/train.txt',
    val_data_path='data/val.txt',
    epochs=10,
    batch_size=32,
    learning_rate=0.001,
    save_dir='./checkpoints'
)

# Export trained model
trainer.export_model('trained_model.pt', export_format='pytorch')
```

### Building Pipelines Programmatically

```python
from llwh.agents import PipelineBuilder

# Create pipeline
pipeline = PipelineBuilder()

# Add blocks
input_block = pipeline.add_block('text_input', config={'text': 'Hello AI'})
process_block = pipeline.add_block('language_processing')
reason_block = pipeline.add_block('reasoning')
output_block = pipeline.add_block('output')

# Connect blocks
pipeline.connect_blocks(input_block.id, process_block.id)
pipeline.connect_blocks(process_block.id, reason_block.id)
pipeline.connect_blocks(reason_block.id, output_block.id)

# Execute pipeline
result = pipeline.execute()
print(result)

# Save pipeline
pipeline.save_pipeline('my_pipeline.json')
```

### AI-to-AI Chat

```python
from llwh.models import AIChatManager
from llwh.core import HybridLanguageWorldModel

# Create chat manager
manager = AIChatManager()

# Add AI agents
model1 = HybridLanguageWorldModel()
model2 = HybridLanguageWorldModel()

manager.add_agent('AI-1', model1, 'hybrid')
manager.add_agent('AI-2', model2, 'hybrid')

# Start conversation
conversation = manager.start_conversation(
    ['AI-1', 'AI-2'],
    initial_message='Discuss the future of AI',
    max_turns=10
)

# Collaborative problem solving
solution = manager.collaborative_solve(
    ['AI-1', 'AI-2'],
    'How to optimize energy consumption?',
    strategy='consensus'
)

# Export conversation
manager.export_conversation(0, 'conversation.txt')
```

### Review Assessment

```python
from llwh.models import ReviewAssessor

# Create review assessor
assessor = ReviewAssessor()

# Assess a single review
review_text = """This is an excellent product! The quality is outstanding 
and it exceeded all my expectations. I would highly recommend this."""

assessment = assessor.assess_review(review_text, include_details=True)

# View results
print("Overall Score:", assessment['overall_score'])
print("Recommendation:", assessment['recommendation'])
print("Quality Score:", assessment['quality_score'])
print("Sentiment Score:", assessment['sentiment_score'])
print("Coherence Score:", assessment['coherence_score'])
print("Clarity Score:", assessment['clarity_score'])

# Batch assessment
reviews = [review1, review2, review3]
batch_results = assessor.batch_assess(reviews)

# Compare two reviews
comparison = assessor.compare_reviews(review1, review2)
print("Better Review:", comparison['better_review'])
print("Score Difference:", comparison['score_difference'])

# Check factual consistency
consistency_score = assessor.assess_factual_consistency(
    review_text, 
    reference_texts=[reference1, reference2]
)

# Get statistics
stats = assessor.get_assessment_statistics()
print("Total Assessments:", stats['total_assessments'])
print("Average Score:", stats['average_score'])

# Export report
assessor.export_assessment_report('assessment_report.txt')
```

## Architecture

### Hybrid Model Components

1. **Language Model**: Transformer-based architecture for text understanding and generation
2. **World Model**: Neural network for spatial reasoning and environment modeling
3. **Fusion Layer**: Cross-modal attention mechanism for integrating language and world understanding

### Agent Block Types

- **Text Input**: Capture and process text input
- **Language Processing**: Apply language model transformations
- **World State**: Manage and update world state representations
- **Reasoning**: Perform logical reasoning operations
- **Action**: Execute actions based on reasoning
- **Output**: Format and return results
- **Conditional**: Branch execution based on conditions
- **Loop**: Repeat operations
- **API Call**: Integrate with external services
- **File I/O**: Read/write files

## Configuration

Model configuration can be customized:

```python
config = {
    'vocab_size': 10000,
    'embedding_dim': 256,
    'hidden_dim': 512,
    'num_layers': 4,
    'state_dim': 256,
    'action_dim': 64
}

model = HybridLanguageWorldModel(config=config)
```

## Windows 7 Compatibility

This system is specifically designed to run on Windows 7:

- Uses Python 2.7 compatible code
- tkinter GUI (built-in, no external dependencies)
- PyTorch 1.4.0 (last version supporting Python 2.7)
- No modern dependencies that require newer Windows versions
- Lightweight architecture suitable for older hardware

## Performance Optimization

For optimal performance on Windows 7:

1. Close unnecessary applications
2. Use smaller batch sizes if memory is limited
3. Reduce model dimensions in config for faster inference
4. Export to ONNX for optimized deployment

## Troubleshooting

**Issue**: GUI doesn't launch
- **Solution**: Ensure tkinter is installed (`python -m tkinter`)

**Issue**: Training is slow
- **Solution**: Reduce batch size or use GPU if available

**Issue**: Out of memory errors
- **Solution**: Reduce model dimensions or batch size

**Issue**: Module import errors
- **Solution**: Run `pip install -e .` in the project directory

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues.

## License

MIT License - See LICENSE file for details

## Author

MASSIVE MAGNETICS

## Acknowledgments

This revolutionary AI system combines cutting-edge research in:
- Large Language Models
- World Models and Environment Understanding
- Multi-Agent Systems
- Human-Computer Interaction

---

**REVOLUTIONARY AI FOR EVERYONE - EVEN ON WINDOWS 7!** 🚀
