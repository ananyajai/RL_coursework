from abc import ABC, abstractmethod
from copy import deepcopy
import gymnasium as gym
import numpy as np
import os.path
from torch import Tensor
from torch.distributions.categorical import Categorical
import torch.nn
from torch.optim import Adam
from typing import Dict, Iterable, List

from rl2024.exercise3.networks import FCNetwork
from rl2024.exercise3.replay import Transition


class Agent(ABC):
    """Base class for Deep RL Exercise 3 Agents

    **DO NOT CHANGE THIS CLASS**

    :attr action_space (gym.Space): action space of used environment
    :attr observation_space (gym.Space): observation space of used environment
    :attr saveables (Dict[str, torch.nn.Module]):
        mapping from network names to PyTorch network modules

    Note:
        see https://gymnasium.farama.org/api/spaces/ for more information on Gymnasium spaces
    """

    def __init__(self, action_space: gym.Space, observation_space: gym.Space):
        """The constructor of the Agent Class

        :param action_space (gym.Space): environment's action space
        :param observation_space (gym.Space): environment's observation space
        """
        self.action_space = action_space
        self.observation_space = observation_space

        self.saveables = {}

    def save(self, path: str, suffix: str = "") -> str:
        """Saves saveable PyTorch models under given path

        The models will be saved in directory found under given path in file "models_{suffix}.pt"
        where suffix is given by the optional parameter (by default empty string "")

        :param path (str): path to directory where to save models
        :param suffix (str, optional): suffix given to models file
        :return (str): path to file of saved models file
        """
        torch.save(self.saveables, path)
        return path

    def restore(self, save_path: str):
        """Restores PyTorch models from models file given by path

        :param save_path (str): path to file containing saved models
        """
        dirname, _ = os.path.split(os.path.abspath(__file__))
        save_path = os.path.join(dirname, save_path)
        checkpoint = torch.load(save_path)
        for k, v in self.saveables.items():
            v.load_state_dict(checkpoint[k].state_dict())

    @abstractmethod
    def act(self, obs: np.ndarray):
        ...

    @abstractmethod
    def schedule_hyperparameters(self, timestep: int, max_timestep: int):
        """Updates the hyperparameters

        This function is called before every episode and allows you to schedule your
        hyperparameters.

        :param timestep (int): current timestep at the beginning of the episode
        :param max_timestep (int): maximum timesteps that the training loop will run for
        """
        ...

    @abstractmethod
    def update(self):
        ...


class DQN(Agent):
    """The DQN agent for exercise 3

    **YOU NEED TO IMPLEMENT FUNCTIONS IN THIS CLASS**

    :attr critics_net (FCNetwork): fully connected DQN to compute Q-value estimates
    :attr critics_target (FCNetwork): fully connected DQN target network
    :attr critics_optim (torch.optim): PyTorch optimiser for DQN critics_net
    :attr learning_rate (float): learning rate for DQN optimisation
    :attr update_counter (int): counter of updates for target network updates
    :attr target_update_freq (int): update frequency (number of iterations after which the target
        networks should be updated)
    :attr batch_size (int): size of sampled batches of experience
    :attr gamma (float): discount rate gamma
    """

    def __init__(
        self,
        action_space: gym.Space,
        observation_space: gym.Space,
        learning_rate: float,
        hidden_size: Iterable[int],
        target_update_freq: int,
        batch_size: int,
        gamma: float,
        epsilon_start: float,
        epsilon_min: float,
        epsilon_decay_strategy: str = "constant",
        epsilon_decay: float = None,
        exploration_fraction: float = None,
        **kwargs,
        ):
        """The constructor of the DQN agent class

        :param action_space (gym.Space): environment's action space
        :param observation_space (gym.Space): environment's observation space
        :param learning_rate (float): learning rate for DQN optimisation
        :param hidden_size (Iterable[int]): list of hidden dimensionalities for fully connected DQNs
        :param target_update_freq (int): update frequency (number of iterations after which the target
            networks should be updated)
        :param batch_size (int): size of sampled batches of experience
        :param gamma (float): discount rate gamma
        :param epsilon_start (float): initial value of epsilon for epsilon-greedy action selection
        :param epsilon_min (float): minimum value of epsilon for epsilon-greedy action selection
        :param epsilon_decay (float, optional): decay rate of epsilon for epsilon-greedy action. If not specified,
                                                epsilon will be decayed linearly from epsilon_start to epsilon_min.
        """
        super().__init__(action_space, observation_space)

        STATE_SIZE = observation_space.shape[0]
        ACTION_SIZE = action_space.n

        # ######################################### #
        #  BUILD YOUR NETWORKS AND OPTIMIZERS HERE  #
        # ######################################### #
        self.critics_net = FCNetwork(
            (STATE_SIZE, *hidden_size, ACTION_SIZE), output_activation=None
            )

        self.critics_target = deepcopy(self.critics_net)

        self.critics_optim = Adam(
            self.critics_net.parameters(), lr=learning_rate, eps=1e-3
            )

        # ############################################# #
        # WRITE ANY HYPERPARAMETERS YOU MIGHT NEED HERE #
        # ############################################# #
        self.learning_rate = learning_rate
        self.update_counter = 0
        self.target_update_freq = target_update_freq
        self.batch_size = batch_size
        self.gamma = gamma

        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_min = epsilon_min

        self.epsilon_decay_strategy = epsilon_decay_strategy
        if epsilon_decay_strategy == "constant":
            assert epsilon_decay is None, "epsilon_decay should be None for epsilon_decay_strategy == 'constant'"
            assert exploration_fraction is None, "exploration_fraction should be None for epsilon_decay_strategy == 'constant'"
            self.epsilon_exponential_decay_factor = None
            self.exploration_fraction = None
        elif self.epsilon_decay_strategy == "linear":
            assert epsilon_decay is None, "epsilon_decay is only set for epsilon_decay_strategy='exponential'"
            assert exploration_fraction is not None, "exploration_fraction must be set for epsilon_decay_strategy='linear'"
            assert exploration_fraction > 0, "exploration_fraction must be positive"
            self.epsilon_exponential_decay_factor = None
            self.exploration_fraction = exploration_fraction
        elif self.epsilon_decay_strategy == "exponential":
            assert epsilon_decay is not None, "epsilon_decay must be set for epsilon_decay_strategy='exponential'"
            assert exploration_fraction is None, "exploration_fraction is only set for epsilon_decay_strategy='linear'"
            self.epsilon_exponential_decay_factor = epsilon_decay
            self.exploration_fraction = None
        else:
            raise ValueError("epsilon_decay_strategy must be either 'linear' or 'exponential'")
        # ######################################### #
        self.saveables.update(
            {
                "critics_net"   : self.critics_net,
                "critics_target": self.critics_target,
                "critic_optim"  : self.critics_optim,
                }
            )

    def schedule_hyperparameters(self, timestep: int, max_timestep: int):
        """Updates the hyperparameters

        **YOU MUST IMPLEMENT THIS FUNCTION FOR Q3**
        ** Implement both epsilon_linear_decay() and epsilon_exponential_decay() functions **
        ** You may modify the signature of these functions if you wish to pass additional arguments **

        This function is called before every episode and allows you to schedule your
        hyperparameters.

        :param timestep (int): current timestep at the beginning of the episode
        :param max_timestep (int): maximum timesteps that the training loop will run for
        """

        def epsilon_linear_decay(*args, **kwargs):
            timestep_ratio = timestep/max_timestep

            if timestep_ratio < self.exploration_fraction:
                self.epsilon = self.epsilon_start - (timestep_ratio/self.exploration_fraction) * (self.epsilon_start - self.epsilon_min)
            
            else:
                self.epsilon = self.epsilon_min

            return self.epsilon

        def epsilon_exponential_decay(*args, **kwargs):
            decay_rate = self.epsilon_exponential_decay_factor**(timestep/max_timestep)

            if self.epsilon > self.epsilon_min:
                self.epsilon *= decay_rate

            else:
                self.epsilon = self.epsilon_min

            return self.epsilon

        if self.epsilon_decay_strategy == "constant":
            pass
        elif self.epsilon_decay_strategy == "linear":
            # linear decay
            ### PUT YOUR CODE HERE ###
            self.epsilon = epsilon_linear_decay(self.epsilon_start, self.epsilon_min, self.exploration_fraction)
        elif self.epsilon_decay_strategy == "exponential":
            # exponential decay
            ### PUT YOUR CODE HERE ###
            self.epsilon = epsilon_exponential_decay(self.epsilon_start, self.epsilon_min, self.epsilon_exponential_decay_factor)
        else:
            raise ValueError("epsilon_decay_strategy must be either 'constant', 'linear' or 'exponential'")

    def act(self, obs: np.ndarray, explore: bool):
        """Returns an action (should be called at every timestep)

        **YOU MUST IMPLEMENT THIS FUNCTION FOR Q3**

        When explore is False you should select the best action possible (greedy). However, during
        exploration, you should be implementing an exploration strategy (like e-greedy). Use
        schedule_hyperparameters() for any hyperparameters that you want to change over time.

        :param obs (np.ndarray): observation vector from the environment
        :param explore (bool): flag indicating whether we should explore
        :return (sample from self.action_space): action the agent should perform
        """
        state = torch.tensor(obs, dtype=torch.float32)
        action_values = self.critics_net(state)

        if explore:
            if np.random.uniform(0, 1) < self.epsilon:
                # Explore - sample a random action
                sampled_action = np.random.choice(self.action_space.n)
            else:
                sampled_action = torch.argmax(action_values).item()
        else:
            # Exploit - choose the action with the highest probability
            sampled_action = torch.argmax(action_values).item()

        return sampled_action

    def update(self, batch: Transition) -> Dict[str, float]:
        """Update function for DQN

        **YOU MUST IMPLEMENT THIS FUNCTION FOR Q3**

        This function is called after storing a transition in the replay buffer. This happens
        every timestep. It should update your network, update the target network at the given
        target update frequency, and return the Q-loss in the form of a dictionary.

        :param batch (Transition): batch vector from replay buffer
        :return (Dict[str, float]): dictionary mapping from loss names to loss values
        """
        state_j, action_j, next_state, reward_j, done = batch

        action_j = action_j.to(torch.long)

        # Compute the next Q-values using the target network
        next_q = self.critics_target(next_state)
        # Find the best action for the next states
        best_action = next_q.gather(1, next_q.max(1)[1].unsqueeze(-1))

        # Calculate current and taget Q-values
        y = reward_j + self.gamma * (1 - done) * best_action
        q = self.critics_net(state_j).gather(1, action_j)
        q_loss = torch.nn.functional.mse_loss(q, y)   # Minimise mean squared error

        # Backpropogate and perform optimisation step
        self.critics_optim.zero_grad()
        q_loss.backward()
        self.critics_optim.step()

        # Hard update for parameter theta
        self.update_counter += 1
        if self.update_counter % self.target_update_freq == 0:
            self.critics_target.load_state_dict(self.critics_net.state_dict())

        return {"q_loss": float(q_loss)}


class Reinforce(Agent):
    """ The Reinforce Agent for Ex 3

    ** YOU NEED TO IMPLEMENT THE FUNCTIONS IN THIS CLASS **

    :attr policy (FCNetwork): fully connected network for policy
    :attr policy_optim (torch.optim): PyTorch optimiser for policy network
    :attr learning_rate (float): learning rate for DQN optimisation
    :attr gamma (float): discount rate gamma
    """

    def __init__(
        self,
        action_space: gym.Space,
        observation_space: gym.Space,
        learning_rate: float,
        hidden_size: Iterable[int],
        gamma: float,
        **kwargs,
        ):
        """
        **YOU MUST IMPLEMENT THIS FUNCTION FOR Q3**

        :param action_space (gym.Space): environment's action space
        :param observation_space (gym.Space): environment's observation space
        :param learning_rate (float): learning rate for DQN optimisation
        :param hidden_size (Iterable[int]): list of hidden dimensionalities for fully connected DQNs
        :param gamma (float): discount rate gamma
        """
        super().__init__(action_space, observation_space)
        STATE_SIZE = observation_space.shape[0]
        ACTION_SIZE = action_space.n

        # ######################################### #
        #  BUILD YOUR NETWORKS AND OPTIMIZERS HERE  #
        # ######################################### #
        self.policy = FCNetwork(
            (STATE_SIZE, *hidden_size, ACTION_SIZE), output_activation=torch.nn.modules.activation.Softmax
            )

        self.policy_optim = Adam(self.policy.parameters(), lr=learning_rate, eps=1e-3)

        # ############################################# #
        # WRITE ANY HYPERPARAMETERS YOU MIGHT NEED HERE #
        # ############################################# #
        self.learning_rate = learning_rate
        self.gamma = gamma

        # ############################### #
        # WRITE ANY AGENT PARAMETERS HERE #
        # ############################### #

        # ###############################################
        self.saveables.update(
            {
                "policy": self.policy,
                }
            )

    def schedule_hyperparameters(self, timestep: int, max_timesteps: int):
        """Updates the hyperparameters

        This function is called before every episode and allows you to schedule your
        hyperparameters.

        :param timestep (int): current timestep at the beginning of the episode
        :param max_timestep (int): maximum timesteps that the training loop will run for
        """
        pass

    def act(self, obs: np.ndarray, explore: bool):
        """Returns an action (should be called at every timestep)

        **YOU MUST IMPLEMENT THIS FUNCTION FOR Q3**

        Select an action from the model's stochastic policy by sampling a discrete action
        from the distribution specified by the model output

        :param obs (np.ndarray): observation vector from the environment
        :param explore (bool): flag indicating whether we should explore
        :return (sample from self.action_space): action the agent should perform
        """
        state = torch.tensor(obs, dtype=torch.float32)
        action_prob = self.policy(state)

        if explore:
            # Explore - sample a random action
            sampled_action = np.random.choice(self.action_space.n)
        else:
            # Exploit - choose the action with the highest probability
            sampled_action = torch.argmax(action_prob).item()

        return sampled_action
    

    def update(
        self, rewards: List[float], observations: List[np.ndarray], actions: List[int],
        ) -> Dict[str, float]:
        """Update function for policy gradients

        **YOU MUST IMPLEMENT THIS FUNCTION FOR Q3**

        :param rewards (List[float]): rewards of episode (from first to last)
        :param observations (List[np.ndarray]): observations of episode (from first to last)
        :param actions (List[int]): applied actions of episode (from first to last)
        :return (Dict[str, float]): dictionary mapping from loss names to loss values
            losses
        """
        # Initialise loss and returns
        p_loss = 0
        G = 0
        traj_length = len(observations)

        # Compute action probabilities using the current policy
        action_probabilities = self.policy(torch.tensor(observations, dtype=torch.float32))

        # Loop backwards in the episode
        for t in range(traj_length - 2, -1, -1):
            G = self.gamma * G + rewards[t+1]
            action_prob = action_probabilities[t, actions[t]]   # Probability of the action at time step t
            p_loss = p_loss - G * torch.log(action_prob)   # Minimise loss function

        p_loss = p_loss/traj_length   # Normalise policy loss

        # Backpropogate and perform optimisation step
        self.policy_optim.zero_grad()
        p_loss.backward()
        self.policy_optim.step()

        return {"p_loss": float(p_loss)}

