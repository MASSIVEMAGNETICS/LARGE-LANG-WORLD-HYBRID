"""Pipeline Action Agent Builder for creating automated workflows."""

import json
import uuid
from typing import List, Dict, Any, Optional


class AgentBlock:
    """
    Base class for agent blocks in the pipeline.
    Each block performs a specific action.
    """
    
    def __init__(self, block_type, name=None, config=None):
        """
        Initialize an agent block.
        
        Args:
            block_type: Type of the block
            name: Optional custom name
            config: Configuration dictionary
        """
        self.id = str(uuid.uuid4())
        self.block_type = block_type
        self.name = name or block_type
        self.config = config or {}
        self.inputs = []
        self.outputs = []
    
    def execute(self, context):
        """
        Execute the agent block.
        
        Args:
            context: Execution context with shared data
            
        Returns:
            result: Output of the block
        """
        raise NotImplementedError("Subclasses must implement execute()")
    
    def connect_to(self, other_block):
        """
        Connect this block's output to another block's input.
        
        Args:
            other_block: The block to connect to
        """
        self.outputs.append(other_block.id)
        other_block.inputs.append(self.id)
    
    def to_dict(self):
        """Convert block to dictionary for serialization."""
        return {
            'id': self.id,
            'type': self.block_type,
            'name': self.name,
            'config': self.config,
            'inputs': self.inputs,
            'outputs': self.outputs
        }
    
    @classmethod
    def from_dict(cls, data):
        """Create block from dictionary."""
        block = cls(data['type'], data['name'], data['config'])
        block.id = data['id']
        block.inputs = data['inputs']
        block.outputs = data['outputs']
        return block


class TextInputBlock(AgentBlock):
    """Block for text input."""
    
    def __init__(self, name=None, config=None):
        super(TextInputBlock, self).__init__('text_input', name, config)
    
    def execute(self, context):
        """Get text input."""
        text = self.config.get('text', '')
        context['text_input'] = text
        return text


class LanguageProcessingBlock(AgentBlock):
    """Block for language processing."""
    
    def __init__(self, name=None, config=None):
        super(LanguageProcessingBlock, self).__init__('language_processing', name, config)
    
    def execute(self, context):
        """Process text with language model."""
        text = context.get('text_input', '')
        # Simulate processing
        result = "Processed: {}".format(text)
        context['processed_text'] = result
        return result


class WorldStateBlock(AgentBlock):
    """Block for world state management."""
    
    def __init__(self, name=None, config=None):
        super(WorldStateBlock, self).__init__('world_state', name, config)
    
    def execute(self, context):
        """Manage world state."""
        state = self.config.get('state', {})
        context['world_state'] = state
        return state


class ReasoningBlock(AgentBlock):
    """Block for reasoning operations."""
    
    def __init__(self, name=None, config=None):
        super(ReasoningBlock, self).__init__('reasoning', name, config)
    
    def execute(self, context):
        """Perform reasoning."""
        text = context.get('processed_text', '')
        state = context.get('world_state', {})
        result = "Reasoning about: {} with state: {}".format(text, state)
        context['reasoning_result'] = result
        return result


class ActionBlock(AgentBlock):
    """Block for action execution."""
    
    def __init__(self, name=None, config=None):
        super(ActionBlock, self).__init__('action', name, config)
    
    def execute(self, context):
        """Execute an action."""
        action = self.config.get('action', 'default_action')
        result = "Executed action: {}".format(action)
        context['action_result'] = result
        return result


class OutputBlock(AgentBlock):
    """Block for output."""
    
    def __init__(self, name=None, config=None):
        super(OutputBlock, self).__init__('output', name, config)
    
    def execute(self, context):
        """Generate output."""
        output_keys = self.config.get('keys', ['reasoning_result'])
        output = {}
        for key in output_keys:
            output[key] = context.get(key, None)
        context['final_output'] = output
        return output


class ConditionalBlock(AgentBlock):
    """Block for conditional branching."""
    
    def __init__(self, name=None, config=None):
        super(ConditionalBlock, self).__init__('conditional', name, config)
    
    def execute(self, context):
        """Execute conditional logic."""
        condition = self.config.get('condition', 'True')
        # Evaluate condition (simplified)
        result = eval(condition, {}, context)
        context['condition_result'] = result
        return result


class LoopBlock(AgentBlock):
    """Block for loop operations."""
    
    def __init__(self, name=None, config=None):
        super(LoopBlock, self).__init__('loop', name, config)
    
    def execute(self, context):
        """Execute loop."""
        iterations = self.config.get('iterations', 1)
        context['loop_iterations'] = iterations
        return iterations


class APICallBlock(AgentBlock):
    """Block for API calls."""
    
    def __init__(self, name=None, config=None):
        super(APICallBlock, self).__init__('api_call', name, config)
    
    def execute(self, context):
        """Make API call."""
        url = self.config.get('url', '')
        method = self.config.get('method', 'GET')
        result = "API call to {} using {}".format(url, method)
        context['api_result'] = result
        return result


class FileIOBlock(AgentBlock):
    """Block for file I/O operations."""
    
    def __init__(self, name=None, config=None):
        super(FileIOBlock, self).__init__('file_io', name, config)
    
    def execute(self, context):
        """Perform file I/O."""
        operation = self.config.get('operation', 'read')
        filepath = self.config.get('filepath', '')
        result = "File {} operation on {}".format(operation, filepath)
        context['file_result'] = result
        return result


class PipelineBuilder:
    """
    Pipeline Builder for creating and executing agent workflows.
    Allows visual/programmatic construction of multi-agent pipelines.
    """
    
    # Map of block types to classes
    BLOCK_TYPES = {
        'text_input': TextInputBlock,
        'language_processing': LanguageProcessingBlock,
        'world_state': WorldStateBlock,
        'reasoning': ReasoningBlock,
        'action': ActionBlock,
        'output': OutputBlock,
        'conditional': ConditionalBlock,
        'loop': LoopBlock,
        'api_call': APICallBlock,
        'file_io': FileIOBlock,
    }
    
    def __init__(self):
        """Initialize the pipeline builder."""
        self.blocks = {}
        self.connections = []
        self.execution_order = []
    
    def add_block(self, block_type, name=None, config=None):
        """
        Add a block to the pipeline.
        
        Args:
            block_type: Type of block to add
            name: Optional custom name
            config: Block configuration
            
        Returns:
            block: The created block
        """
        if block_type not in self.BLOCK_TYPES:
            raise ValueError("Unknown block type: {}".format(block_type))
        
        block_class = self.BLOCK_TYPES[block_type]
        block = block_class(name=name, config=config)
        self.blocks[block.id] = block
        
        return block
    
    def connect_blocks(self, source_id, target_id):
        """
        Connect two blocks.
        
        Args:
            source_id: ID of source block
            target_id: ID of target block
        """
        if source_id not in self.blocks or target_id not in self.blocks:
            raise ValueError("Invalid block IDs")
        
        source = self.blocks[source_id]
        target = self.blocks[target_id]
        
        source.connect_to(target)
        self.connections.append((source_id, target_id))
    
    def build_execution_order(self):
        """
        Build the execution order using topological sort.
        
        Returns:
            execution_order: List of block IDs in execution order
        """
        # Simple topological sort
        in_degree = {block_id: len(block.inputs) for block_id, block in self.blocks.items()}
        queue = [block_id for block_id, degree in in_degree.items() if degree == 0]
        execution_order = []
        
        while queue:
            current = queue.pop(0)
            execution_order.append(current)
            
            block = self.blocks[current]
            for output_id in block.outputs:
                in_degree[output_id] -= 1
                if in_degree[output_id] == 0:
                    queue.append(output_id)
        
        if len(execution_order) != len(self.blocks):
            raise ValueError("Pipeline contains cycles!")
        
        self.execution_order = execution_order
        return execution_order
    
    def execute(self, initial_context=None):
        """
        Execute the pipeline.
        
        Args:
            initial_context: Initial context dictionary
            
        Returns:
            context: Final execution context
        """
        if not self.execution_order:
            self.build_execution_order()
        
        context = initial_context or {}
        context['execution_log'] = []
        
        for block_id in self.execution_order:
            block = self.blocks[block_id]
            
            try:
                result = block.execute(context)
                context['execution_log'].append({
                    'block_id': block_id,
                    'block_name': block.name,
                    'status': 'success',
                    'result': str(result)
                })
            except Exception as e:
                context['execution_log'].append({
                    'block_id': block_id,
                    'block_name': block.name,
                    'status': 'error',
                    'error': str(e)
                })
                raise
        
        return context
    
    def save_pipeline(self, filepath):
        """
        Save pipeline to JSON file.
        
        Args:
            filepath: Path to save pipeline
        """
        pipeline_data = {
            'blocks': [block.to_dict() for block in self.blocks.values()],
            'connections': self.connections
        }
        
        with open(filepath, 'w') as f:
            json.dump(pipeline_data, f, indent=2)
    
    def load_pipeline(self, filepath):
        """
        Load pipeline from JSON file.
        
        Args:
            filepath: Path to load pipeline from
        """
        with open(filepath, 'r') as f:
            pipeline_data = json.load(f)
        
        # Clear current pipeline
        self.blocks = {}
        self.connections = []
        self.execution_order = []
        
        # Load blocks
        for block_data in pipeline_data['blocks']:
            block_type = block_data['type']
            if block_type in self.BLOCK_TYPES:
                block_class = self.BLOCK_TYPES[block_type]
                block = block_class.from_dict(block_data)
                self.blocks[block.id] = block
        
        # Load connections
        self.connections = pipeline_data['connections']
    
    def get_available_block_types(self):
        """
        Get list of available block types.
        
        Returns:
            block_types: List of available block type names
        """
        return list(self.BLOCK_TYPES.keys())
    
    def clear(self):
        """Clear the pipeline."""
        self.blocks = {}
        self.connections = []
        self.execution_order = []
    
    def to_dict(self):
        """Convert pipeline to dictionary."""
        return {
            'blocks': [block.to_dict() for block in self.blocks.values()],
            'connections': self.connections,
            'execution_order': self.execution_order
        }
    
    def visualize(self):
        """
        Generate a text-based visualization of the pipeline.
        
        Returns:
            visualization: String representation of pipeline
        """
        lines = ["Pipeline Visualization", "=" * 50]
        
        for block_id in self.execution_order:
            block = self.blocks[block_id]
            lines.append("\n[{}] {}".format(block.block_type, block.name))
            if block.inputs:
                lines.append("  Inputs from: {}".format(
                    [self.blocks[inp].name for inp in block.inputs]))
            if block.outputs:
                lines.append("  Outputs to: {}".format(
                    [self.blocks[out].name for out in block.outputs]))
        
        return "\n".join(lines)
