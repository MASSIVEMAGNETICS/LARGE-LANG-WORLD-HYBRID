# Quick Start Guide

## Installation

### Step 1: Install Python
Ensure Python 3.6+ is installed on your system.

For Windows 7:
- Download Python from python.org
- During installation, check "Add Python to PATH"

### Step 2: Install Dependencies

```bash
cd LARGE-LANG-WORLD-HYBRID
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

Note: For Windows 7, you may need PyTorch 1.7 or earlier:
```bash
pip install torch==1.7.1+cpu torchvision==0.8.2+cpu -f https://download.pytorch.org/whl/torch_stable.html
```

### Step 3: Install the Package

```bash
pip install -e .
```

## Running the Application

### GUI Application

```bash
# Option 1: Use the launcher script
python run_gui.py

# Option 2: Use the installed command
llwh-gui

# Option 3: Run directly
python -m llwh.gui.main
```

### Running Examples

```bash
python examples.py
```

This will demonstrate:
- Basic model creation and usage
- Pipeline building and execution
- AI-to-AI conversations
- Training setup
- Model save/load

## First Steps

### 1. Create Your First Model

Launch the GUI and:
1. Click File → New Model
2. The model is now ready to use

### 2. Chat with the AI

1. Go to the "Chat Interface" tab
2. Type your message in the input box
3. Press Ctrl+Enter or click "Send"
4. Adjust temperature for more creative responses

### 3. Build a Pipeline

1. Go to the "Pipeline Builder" tab
2. Select agent blocks from the left panel
3. Click "Add Block" to add them
4. Click "Run Pipeline" to execute

### 4. AI-to-AI Chat

1. Go to the "AI-to-AI Chat" tab
2. Select two AI models
3. Enter a topic
4. Click "Start Conversation"

## Training Your Own Model

### Prepare Your Data

Create a text file with your training data:
```
data/train.txt
```

Each line should be a training example.

### Train via GUI

1. Go to "Training Suite" tab
2. Click "Browse" to select your dataset
3. Set epochs, batch size, and learning rate
4. Click "Start Training"
5. Monitor progress in the log
6. Save your model when done

### Train via CLI

```bash
llwh-train --data data/train.txt --epochs 10 --batch-size 32 --lr 0.001 --save-dir ./my_models
```

## Troubleshooting

**Problem**: GUI doesn't launch
**Solution**: Install tkinter (`sudo apt-get install python3-tk` on Linux)

**Problem**: Import errors
**Solution**: Run `pip install -e .` in the project directory

**Problem**: PyTorch not found
**Solution**: Install PyTorch: `pip install torch`

**Problem**: Out of memory
**Solution**: Reduce batch size or model dimensions

## Next Steps

- Read the full README.md for detailed documentation
- Explore examples.py for code samples
- Check the API documentation in each module
- Build custom agent pipelines
- Train on your own datasets

## Support

For issues, questions, or contributions:
- Open an issue on GitHub
- Check the documentation in README.md
- Review the code examples

---

**Welcome to the Revolutionary AI!** 🚀
