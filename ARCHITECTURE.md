# System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                   LARGE LANGUAGE-WORLD HYBRID AI                        │
│                         Main Application                                │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    │               │               │
                    ▼               ▼               ▼
        ┌───────────────┐  ┌───────────────┐  ┌───────────────┐
        │  GUI Layer    │  │  Core Models  │  │  Services     │
        │  (tkinter)    │  │               │  │               │
        └───────────────┘  └───────────────┘  └───────────────┘
                │                  │                  │
        ┌───────┴───────┐  ┌──────┴──────┐  ┌───────┴───────┐
        │               │  │             │  │               │
        ▼               ▼  ▼             ▼  ▼               ▼
   ┌────────┐    ┌─────────┐  ┌──────────┐  ┌──────────┐
   │ Chat   │    │Training │  │ Language │  │ Pipeline │
   │Interface│   │  Suite  │  │  Model   │  │ Builder  │
   └────────┘    └─────────┘  └──────────┘  └──────────┘
        │             │             │             │
        │             │             │             │
        ▼             ▼             ▼             ▼
   ┌────────┐    ┌─────────┐  ┌──────────┐  ┌──────────┐
   │Pipeline│    │AI-to-AI │  │  World   │  │ AI Chat  │
   │Builder │    │  Chat   │  │  Model   │  │ Manager  │
   └────────┘    └─────────┘  └──────────┘  └──────────┘
                                    │
                                    ▼
                            ┌───────────────┐
                            │ Hybrid Fusion │
                            │   Layer       │
                            └───────────────┘
```

## Component Details

### 1. GUI Layer (llwh/gui/)
```
MainApplication (tkinter.Tk)
    ├── Menu Bar
    │   ├── File Menu (New, Load, Save, Export)
    │   └── Help Menu (About)
    │
    ├── Tab 1: Chat Interface
    │   ├── Chat Display (ScrolledText)
    │   ├── Input Area (Text)
    │   └── Controls (Temperature, Send, Clear)
    │
    ├── Tab 2: Training Suite
    │   ├── Training Controls
    │   │   ├── Dataset Selection
    │   │   ├── Hyperparameters
    │   │   └── Start/Stop Buttons
    │   ├── Training Log (ScrolledText)
    │   └── Progress Bar
    │
    ├── Tab 3: Pipeline Builder
    │   ├── Agent Blocks List (Listbox)
    │   ├── Pipeline Canvas (Canvas)
    │   ├── Controls (Add, Clear, Save, Load, Run)
    │   └── Output Display (ScrolledText)
    │
    └── Tab 4: AI-to-AI Chat
        ├── Model Selection (Comboboxes)
        ├── Conversation Display (ScrolledText)
        └── Controls (Topic, Turns, Start)
```

### 2. Core Models (llwh/core/)
```
HybridLanguageWorldModel (nn.Module)
    │
    ├── LanguageModel
    │   ├── Token Embeddings
    │   ├── Positional Embeddings
    │   ├── Transformer Encoder (4 layers)
    │   │   └── Multi-head Attention (8 heads)
    │   ├── Output Projection
    │   └── Generation Methods
    │
    ├── WorldModel
    │   ├── State Encoder
    │   ├── Dynamics Predictor
    │   ├── Reward Predictor
    │   └── Value Estimator
    │
    └── Fusion Mechanism
        ├── Language → World Projection
        ├── World → Language Projection
        ├── Cross-modal Attention
        └── Joint Reasoning Layer
```

### 3. Training System (llwh/training/)
```
ModelTrainer
    ├── Optimizer Setup
    │   ├── Adam
    │   ├── SGD
    │   └── AdamW
    │
    ├── Training Loop
    │   ├── Epoch Iteration
    │   ├── Batch Processing
    │   ├── Loss Calculation
    │   └── Backpropagation
    │
    ├── Validation
    │   └── Loss Evaluation
    │
    ├── Checkpointing
    │   ├── Regular Checkpoints (every 5 epochs)
    │   └── Best Model Checkpoint
    │
    └── Export
        ├── PyTorch Format (.pt)
        └── ONNX Format (.onnx)
```

### 4. Pipeline System (llwh/agents/)
```
PipelineBuilder
    │
    ├── Agent Blocks
    │   ├── TextInputBlock
    │   ├── LanguageProcessingBlock
    │   ├── WorldStateBlock
    │   ├── ReasoningBlock
    │   ├── ActionBlock
    │   ├── OutputBlock
    │   ├── ConditionalBlock
    │   ├── LoopBlock
    │   ├── APICallBlock
    │   └── FileIOBlock
    │
    ├── Pipeline Execution
    │   ├── Topological Sort
    │   ├── Cycle Detection
    │   ├── Context Management
    │   └── Error Handling
    │
    └── Persistence
        ├── Save to JSON
        └── Load from JSON
```

### 5. AI Chat System (llwh/models/)
```
AIChatManager
    │
    ├── AIAgent (multiple instances)
    │   ├── Model Reference
    │   ├── Conversation History
    │   └── Response Generation
    │
    ├── Conversation Management
    │   ├── Turn-based Dialogue
    │   ├── Topic Tracking
    │   └── History Recording
    │
    └── Collaboration Strategies
        ├── Round-Robin
        │   └── Sequential Contributions
        ├── Voting
        │   └── Democratic Selection
        └── Consensus
            └── Iterative Refinement
```

## Data Flow

### Chat Interface Flow
```
User Input → GUI → HybridModel → Language Model
                      ↓
                  World Model
                      ↓
                 Fusion Layer
                      ↓
                 Generate Response → Display
```

### Training Flow
```
Dataset → DataLoader → Batch
                         ↓
                   Forward Pass
                         ↓
                   Loss Calculation
                         ↓
                   Backward Pass
                         ↓
                   Optimizer Step
                         ↓
                   Checkpoint Save
```

### Pipeline Flow
```
Input Block → Process Block → Reason Block → Output Block
      ↓              ↓              ↓              ↓
   Context    →   Context    →   Context    →  Final Result
```

### AI-to-AI Flow
```
Agent 1 → Message → Agent 2
   ↓                    ↓
Response ← Agent 2  Response
   ↓
Agent 1 ← Message
```

## Integration Points

1. **GUI ↔ Models**: Direct instantiation and method calls
2. **GUI ↔ Training**: Threading for non-blocking training
3. **GUI ↔ Pipeline**: Canvas visualization and execution
4. **Models ↔ Training**: Trainer wraps model for training
5. **Pipeline ↔ Models**: Blocks can use models internally
6. **AI Chat ↔ Models**: Manager coordinates multiple models

## External Dependencies

- **PyTorch**: Core tensor operations and neural networks
- **tkinter**: GUI framework (built-in)
- **Python Standard Library**: os, json, threading, etc.

## Deployment Architecture

```
┌─────────────────────────────────────┐
│         User's Computer             │
│  (Windows 7 or later)               │
│                                      │
│  ┌────────────────────────────┐    │
│  │   Python Runtime           │    │
│  │   (2.7 or 3.x)             │    │
│  │                             │    │
│  │  ┌──────────────────────┐  │    │
│  │  │  GUI Application     │  │    │
│  │  │  (tkinter window)    │  │    │
│  │  └──────────────────────┘  │    │
│  │                             │    │
│  │  ┌──────────────────────┐  │    │
│  │  │  AI Models           │  │    │
│  │  │  (PyTorch)           │  │    │
│  │  └──────────────────────┘  │    │
│  │                             │    │
│  │  ┌──────────────────────┐  │    │
│  │  │  Local Storage       │  │    │
│  │  │  (models, data)      │  │    │
│  │  └──────────────────────┘  │    │
│  └────────────────────────────┘    │
└─────────────────────────────────────┘
```

## Performance Characteristics

- **Startup Time**: < 5 seconds
- **Model Loading**: 1-5 seconds
- **Inference**: Real-time (< 1 second per response)
- **Training**: Varies by dataset size and hardware
- **Memory Usage**: 500MB - 2GB depending on model size
- **Storage**: ~100MB for code + models

---

**Complete System Architecture for Revolutionary AI!** 🏗️
