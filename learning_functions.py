import copy
import itertools
import pickle
from collections import defaultdict, namedtuple
import json

import numpy as np

import plotting


def make_epsilon_greedy_policy(Q, epsilon, nA):
    """
    Creates an epsilon-greedy policy based on a given Q-function and epsilon.

    Args:
        Q: A dictionary that maps from state -> action-values.
            Each value is a numpy array of length nA (see below)
        epsilon: The probability to select a random action . float between 0 and 1.
        nA: Number of actions in the environment.

    Returns:
        A function that takes the observation as an argument and returns
        the probabilities for each action in the form of a numpy array of length nA.

    """

    def policy_fn(observation):
        A = np.ones(nA, dtype=float) * epsilon / nA
        best_action = np.argmax(Q[observation])
        A[best_action] += (1.0 - epsilon)
        return A

    return policy_fn


def make_epsilon_greedy_policy_double(Q_a, Q_b, epsilon, nA):
    """
    Creates an epsilon-greedy policy based on a given Q-function and epsilon.

    Args:
        Q: A dictionary that maps from state -> action-values.
            Each value is a numpy array of length nA (see below)
        epsilon: The probability to select a random action . float between 0 and 1.
        nA: Number of actions in the environment.

    Returns:
        A function that takes the observation as an argument and returns
        the probabilities for each action in the form of a numpy array of length nA.

    """

    def policy_fn(observation):
        A = np.ones(nA, dtype=float) * epsilon / nA
        best_action = np.argmax(Q_a[observation] / 2 + Q_b[observation] / 2)
        A[best_action] += (1.0 - epsilon)
        return A

    return policy_fn


def serialize_obj(observation_arg):
    try:
        hash(observation_arg)
        return observation_arg
    except:
        try:
            observation = json.dumps(observation_arg)
        except:
            if isinstance(observation_arg, dict):
                obs_copy = copy.deepcopy(observation_arg)
                for k, v in obs_copy.items():
                    obs_copy[k] = str(v)
                observation = json.dumps(obs_copy)
            elif isinstance(observation_arg, np.ndarray):
                observation = str(observation_arg)
            else:
                observation = hash(pickle.dumps(observation_arg))
    return observation



def sarsa(env, num_episodes, discount_factor=1.0, alpha=0.6, epsilon=0.1):
    """
    SARSA algorithm: On-policy TD control. Finds the optimal epsilon-greedy policy.

    Args:
        env: OpenAI environment.
        num_episodes: Number of episodes to run for.
        discount_factor: Gamma discount factor.
        alpha: TD learning rate.
        epsilon: Chance the sample a random action. Float betwen 0 and 1.

    Returns:
        A tuple (Q, stats).
        Q is the optimal action-value function, a dictionary mapping state -> action values.
        stats is an EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
    """
    # The final action-value function.
    # A nested dictionary that maps state -> (action -> action-value).
    Q = defaultdict(lambda: np.random.randn(env.action_space.n) / 100000000)

    # Keeps track of useful statistics
    stats = plotting.EpisodeStats(episode_lengths=np.zeros(num_episodes), episode_rewards=np.zeros(num_episodes))

    # The policy we're following
    policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)

    for i_episode in range(num_episodes):

        # Reset the environment and pick the first action
        state = env.reset()
        state_ser = serialize_obj(state)
        action_probs = policy(state_ser)
        action = np.random.choice(np.arange(len(action_probs)), p=action_probs)

        # One step in the environment
        for t in itertools.count():
            state_ser = serialize_obj(state)
            # Take a step
            next_state, reward, done, _ = env.step(action)
            next_state_ser = serialize_obj(next_state)

            # Pick the next action
            next_action_probs = policy(next_state_ser)
            next_action = np.random.choice(np.arange(len(next_action_probs)), p=next_action_probs)

            # Update statistics
            stats.episode_rewards[i_episode] += reward
            stats.episode_lengths[i_episode] = t

            # TD Update
            td_target = reward + discount_factor * Q[next_state_ser][next_action]
            td_delta = td_target - Q[state_ser][action]
            Q[state_ser][action] += alpha * td_delta

            if done:
                break

            action = next_action
            state = next_state

    return Q, stats


def qLearning(env, num_episodes: int, discount_factor: float = 1.0, alpha: float = 0.6, epsilon: float = 0.1):
    """
    Q-Learning algorithm: Off-policy TD control.
    Finds the optimal greedy policy while improving
    following an epsilon-greedy policy"""

    # Action value function
    # A nested dictionary that maps
    # state -> (action -> action-value).
    Q = defaultdict(lambda: np.random.randn(env.action_space.n) / 100000000)

    # Keeps track of useful statistics
    stats = plotting.EpisodeStats(episode_lengths=np.zeros(num_episodes), episode_rewards=np.zeros(num_episodes))

    # Create an epsilon greedy policy function
    # appropriately for environment action space
    policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)

    # For every episode
    for ith_episode in range(num_episodes):

        # Reset the environment and pick the first action
        state: int = env.reset()

        for t in itertools.count():

            # get probabilities of all actions from current state
            state_ser = serialize_obj(state)
            action_probabilities = policy(state_ser)

            # choose action according to
            # the probability distribution
            action: int = np.random.choice(np.arange(len(action_probabilities)), p=action_probabilities)

            # take action and get reward, transit to next state
            next_state: int
            next_state, reward, done, _ = env.step(action)

            # Update statistics
            stats.episode_rewards[ith_episode] += reward
            stats.episode_lengths[ith_episode] = t

            # TD Update
            next_state_ser = serialize_obj(next_state)
            best_next_action = np.argmax(Q[next_state_ser])
            td_target = reward + discount_factor * Q[next_state_ser][best_next_action]
            td_delta = td_target - Q[state_ser][action]
            Q[state_ser][action] += alpha * td_delta

            # done is True if episode terminated
            if done:
                break

            state = next_state

    return Q, stats


def double_Q_Learning(env, num_episodes: int, discount_factor: float = 1.0, alpha: float = 0.6, epsilon: float = 0.1):
    """
    Q-Learning algorithm: Off-policy TD control.
    Finds the optimal greedy policy while improving
    following an epsilon-greedy policy"""

    # Action value function
    # A nested dictionary that maps
    # state -> (action -> action-value).
    Q_a = defaultdict(lambda: np.random.randn(env.action_space.n) / 100000000)
    Q_b = defaultdict(lambda: np.random.randn(env.action_space.n) / 100000000)

    # Keeps track of useful statistics
    stats = plotting.EpisodeStats(episode_lengths=np.zeros(num_episodes), episode_rewards=np.zeros(num_episodes))

    # Create an epsilon greedy policy function
    # appropriately for environment action space
    policy = make_epsilon_greedy_policy_double(Q_a, Q_b, epsilon, env.action_space.n)

    # For every episode
    for ith_episode in range(num_episodes):

        # Reset the environment and pick the first action
        state: int = env.reset()

        for t in itertools.count():
            state_ser = serialize_obj(state)

            # get probabilities of all actions from current state
            action_probabilities = policy(state_ser)

            # choose action according to
            # the probability distribution
            action: int = np.random.choice(np.arange(len(action_probabilities)), p=action_probabilities)

            # take action and get reward, transit to next state
            next_state: int
            next_state, reward, done, _ = env.step(action)

            # Update statistics
            stats.episode_rewards[ith_episode] += reward
            stats.episode_lengths[ith_episode] = t

            Q_A, Q_B = np.random.permutation([Q_a, Q_b])
            # TD Update
            next_state_ser = serialize_obj(next_state)
            best_next_action = np.argmax(Q_A[next_state_ser])
            td_target = reward + discount_factor * Q_B[next_state_ser][best_next_action]
            td_delta = td_target - Q_A[state_ser][action]
            Q_A[state_ser][action] += alpha * td_delta

            # done is True if episode terminated
            if done:
                break

            state = next_state

    return Q_b, stats


def avg_res(res_dict):
    if not res_dict:
        return
    Q = defaultdict(lambda: np.zeros_like(next(iter(res_dict[0][0].values()))))

    episode_lengths = np.zeros_like(res_dict[0][1].episode_lengths)
    episode_rewards = np.zeros_like(res_dict[0][1].episode_rewards)
    n = len(res_dict)
    for q, st in res_dict:
        for k, v in q.items():
            Q[k] += v / n
        episode_lengths += st.episode_lengths / n
        episode_rewards += st.episode_rewards / n
    return Q, plotting.EpisodeStats(episode_lengths=episode_lengths, episode_rewards=episode_rewards)
