"""World Model component of the hybrid system."""

import torch
import torch.nn as nn


class WorldModel(nn.Module):
    """
    World Model component that handles spatial/environmental reasoning.
    Maintains state representation and predicts world dynamics.
    """
    
    def __init__(self, state_dim=256, action_dim=64, hidden_dim=512):
        """
        Initialize the World Model.
        
        Args:
            state_dim: Dimension of world state representation
            action_dim: Dimension of action space
            hidden_dim: Hidden layer dimension
        """
        super(WorldModel, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        
        # State encoder
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim)
        )
        
        # Dynamics model (predicts next state)
        self.dynamics = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim)
        )
        
        # Reward predictor
        self.reward_predictor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Value estimator
        self.value_estimator = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def encode_state(self, state):
        """
        Encode a state into latent representation.
        
        Args:
            state: Raw state tensor
            
        Returns:
            encoded_state: Latent state representation
        """
        return self.state_encoder(state)
    
    def predict_next_state(self, state, action):
        """
        Predict the next state given current state and action.
        
        Args:
            state: Current state [batch_size, state_dim]
            action: Action to take [batch_size, action_dim]
            
        Returns:
            next_state: Predicted next state
        """
        state_action = torch.cat([state, action], dim=-1)
        next_state = self.dynamics(state_action)
        return next_state
    
    def predict_reward(self, state):
        """
        Predict reward for a given state.
        
        Args:
            state: State tensor
            
        Returns:
            reward: Predicted reward
        """
        return self.reward_predictor(state)
    
    def estimate_value(self, state):
        """
        Estimate value of a state.
        
        Args:
            state: State tensor
            
        Returns:
            value: Estimated value
        """
        return self.value_estimator(state)
    
    def rollout(self, initial_state, actions):
        """
        Perform a rollout in the world model.
        
        Args:
            initial_state: Starting state
            actions: Sequence of actions
            
        Returns:
            states: Sequence of predicted states
            rewards: Sequence of predicted rewards
        """
        states = [initial_state]
        rewards = []
        
        current_state = initial_state
        for action in actions:
            next_state = self.predict_next_state(current_state, action)
            reward = self.predict_reward(next_state)
            
            states.append(next_state)
            rewards.append(reward)
            current_state = next_state
        
        return torch.stack(states), torch.stack(rewards)
    
    def imagine_trajectory(self, initial_state, horizon=10):
        """
        Imagine a trajectory through the world model.
        
        Args:
            initial_state: Starting state
            horizon: Number of steps to imagine
            
        Returns:
            imagined_states: Sequence of imagined states
        """
        imagined_states = [initial_state]
        current_state = initial_state
        
        for _ in range(horizon):
            # Sample random action for exploration
            batch_size = current_state.shape[0]
            random_action = torch.randn(batch_size, self.action_dim)
            next_state = self.predict_next_state(current_state, random_action)
            imagined_states.append(next_state)
            current_state = next_state
        
        return torch.stack(imagined_states)
