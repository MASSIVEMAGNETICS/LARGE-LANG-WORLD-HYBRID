"""AI-to-AI Chat Manager for multi-model interactions."""

import torch
from typing import List, Dict, Any, Optional


class AIAgent:
    """
    Represents an AI agent that can participate in conversations.
    """
    
    def __init__(self, name, model, model_type='hybrid'):
        """
        Initialize an AI agent.
        
        Args:
            name: Name of the agent
            model: The AI model
            model_type: Type of model (hybrid, language, world)
        """
        self.name = name
        self.model = model
        self.model_type = model_type
        self.conversation_history = []
    
    def respond(self, message, context=None):
        """
        Generate a response to a message.
        
        Args:
            message: Input message
            context: Optional conversation context
            
        Returns:
            response: Generated response
        """
        # Simple response generation (would use actual model in production)
        response = "[{} - {}]: Response to '{}'".format(
            self.name, self.model_type, message
        )
        
        self.conversation_history.append({
            'role': 'received',
            'message': message
        })
        self.conversation_history.append({
            'role': 'sent',
            'message': response
        })
        
        return response
    
    def get_conversation_history(self):
        """Get conversation history."""
        return self.conversation_history
    
    def clear_history(self):
        """Clear conversation history."""
        self.conversation_history = []


class AIChatManager:
    """
    Manages conversations between multiple AI models.
    Enables AI-to-AI interactions for collaborative problem-solving.
    """
    
    def __init__(self):
        """Initialize the chat manager."""
        self.agents = {}
        self.conversations = []
    
    def add_agent(self, name, model, model_type='hybrid'):
        """
        Add an AI agent to the manager.
        
        Args:
            name: Name of the agent
            model: The AI model
            model_type: Type of model
            
        Returns:
            agent: The created agent
        """
        agent = AIAgent(name, model, model_type)
        self.agents[name] = agent
        return agent
    
    def remove_agent(self, name):
        """
        Remove an AI agent.
        
        Args:
            name: Name of the agent to remove
        """
        if name in self.agents:
            del self.agents[name]
    
    def start_conversation(self, agent_names, initial_message, max_turns=10):
        """
        Start a conversation between specified agents.
        
        Args:
            agent_names: List of agent names to include
            initial_message: Initial message to start conversation
            max_turns: Maximum number of conversation turns
            
        Returns:
            conversation: List of conversation turns
        """
        if not agent_names or len(agent_names) < 2:
            raise ValueError("Need at least 2 agents for a conversation")
        
        # Validate agents exist
        for name in agent_names:
            if name not in self.agents:
                raise ValueError("Agent '{}' not found".format(name))
        
        conversation = [{
            'turn': 0,
            'agent': 'system',
            'message': initial_message
        }]
        
        current_message = initial_message
        
        for turn in range(max_turns):
            # Alternate between agents
            agent_name = agent_names[turn % len(agent_names)]
            agent = self.agents[agent_name]
            
            # Generate response
            response = agent.respond(current_message)
            
            conversation.append({
                'turn': turn + 1,
                'agent': agent_name,
                'message': response
            })
            
            current_message = response
        
        self.conversations.append(conversation)
        return conversation
    
    def collaborative_solve(self, agent_names, problem, strategy='round_robin'):
        """
        Have multiple agents collaboratively solve a problem.
        
        Args:
            agent_names: List of agent names
            problem: Problem description
            strategy: Collaboration strategy (round_robin, voting, consensus)
            
        Returns:
            solution: Collaborative solution
        """
        if strategy == 'round_robin':
            return self._round_robin_solve(agent_names, problem)
        elif strategy == 'voting':
            return self._voting_solve(agent_names, problem)
        elif strategy == 'consensus':
            return self._consensus_solve(agent_names, problem)
        else:
            raise ValueError("Unknown strategy: {}".format(strategy))
    
    def _round_robin_solve(self, agent_names, problem):
        """Round-robin problem solving."""
        solutions = []
        
        for agent_name in agent_names:
            agent = self.agents[agent_name]
            solution = agent.respond("Solve: {}".format(problem))
            solutions.append({
                'agent': agent_name,
                'solution': solution
            })
        
        # Combine solutions
        combined = "Combined solutions from {}:\n".format(agent_names)
        for sol in solutions:
            combined += "- {}: {}\n".format(sol['agent'], sol['solution'])
        
        return combined
    
    def _voting_solve(self, agent_names, problem):
        """Voting-based problem solving."""
        # Each agent proposes a solution
        proposals = []
        
        for agent_name in agent_names:
            agent = self.agents[agent_name]
            proposal = agent.respond("Propose solution for: {}".format(problem))
            proposals.append({
                'agent': agent_name,
                'proposal': proposal
            })
        
        # Each agent votes on proposals
        votes = {i: 0 for i in range(len(proposals))}
        
        for agent_name in agent_names:
            agent = self.agents[agent_name]
            # Simplified voting (would use actual model evaluation)
            vote_idx = hash(agent_name + problem) % len(proposals)
            votes[vote_idx] += 1
        
        # Get winner
        winner_idx = max(votes, key=votes.get)
        winner = proposals[winner_idx]
        
        return "Winning solution from {} (votes: {}):\n{}".format(
            winner['agent'], votes[winner_idx], winner['proposal']
        )
    
    def _consensus_solve(self, agent_names, problem):
        """Consensus-based problem solving."""
        # Iterative refinement until consensus
        max_iterations = 5
        current_solution = problem
        
        for iteration in range(max_iterations):
            refinements = []
            
            for agent_name in agent_names:
                agent = self.agents[agent_name]
                refinement = agent.respond(
                    "Refine solution (iteration {}): {}".format(
                        iteration + 1, current_solution
                    )
                )
                refinements.append(refinement)
            
            # Merge refinements (simplified)
            current_solution = "Consensus solution (iter {}): {}".format(
                iteration + 1, " | ".join(refinements)
            )
        
        return current_solution
    
    def get_agent_stats(self, agent_name):
        """
        Get statistics for an agent.
        
        Args:
            agent_name: Name of the agent
            
        Returns:
            stats: Dictionary of statistics
        """
        if agent_name not in self.agents:
            raise ValueError("Agent '{}' not found".format(agent_name))
        
        agent = self.agents[agent_name]
        history = agent.get_conversation_history()
        
        return {
            'name': agent_name,
            'model_type': agent.model_type,
            'total_messages': len(history),
            'sent_messages': len([h for h in history if h['role'] == 'sent']),
            'received_messages': len([h for h in history if h['role'] == 'received'])
        }
    
    def export_conversation(self, conversation_idx, filepath):
        """
        Export a conversation to file.
        
        Args:
            conversation_idx: Index of conversation to export
            filepath: Path to save conversation
        """
        if conversation_idx >= len(self.conversations):
            raise ValueError("Invalid conversation index")
        
        conversation = self.conversations[conversation_idx]
        
        with open(filepath, 'w') as f:
            for turn in conversation:
                f.write("[Turn {}] {}: {}\n".format(
                    turn['turn'], turn['agent'], turn['message']
                ))
    
    def clear_all_histories(self):
        """Clear conversation histories for all agents."""
        for agent in self.agents.values():
            agent.clear_history()
        self.conversations = []
