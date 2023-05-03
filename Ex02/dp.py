import numpy as np
from lib import GridworldMDP, print_value, print_deterministic_policy, init_value, random_policy


def policy_evaluation_one_step(mdp, V, policy, discount=0.99):
    """ Computes one step of policy evaluation.
    Arguments: MDP, value function, policy, discount factor
    Returns: Value function of policy
    """

    # TODO: Write your implementation here
    V_new = V.copy()        # Init value function array
    for state in range(mdp.num_states):
        v = 0
        ordered_names = [0, 1, 2, 3]
        mat = np.squeeze(np.array([mdp.P[state][i] for i in ordered_names]))
        for action in range(4):
            prob = mat[action, 0]
            next_state = int(mat[action, 1])
            reward = mat[action, 2]
            v += policy[state, action] * prob * (reward + discount * V[int(next_state)])
        V_new[state] = v
    return V_new


def policy_evaluation(mdp, policy, discount=0.99, theta=0.01):
    """ Computes full policy evaluation until convergence.
    Arguments: MDP, policy, discount factor, theta
    Returns: Value function of policy
    """

    # TODO: Write your implementation here
    V = init_value(mdp)     # Init value function array
    while True:
        delta = 0
        V_old = V.copy()
        V = policy_evaluation_one_step(mdp=mdp, V=V, policy=policy, discount=discount)
        delta = max(delta, np.max(np.abs(V_old - V)))
        if delta < theta:
            break
    return V


def policy_improvement(mdp, V, discount=0.99):
    """ Computes greedy policy w.r.t a given MDP and value function.
    Arguments: MDP, value function, discount factor
    Returns: policy
    """
    # TODO: Write your implementation here
    # Initialize a policy array in which to save the greed policy 
    policy = np.zeros_like(random_policy(mdp))
    for state in range(mdp.num_states):
        chosen_actions = np.argmax(policy[state])
        action_values = np.zeros(mdp.num_actions)
        for a in range(mdp.num_actions):
            for p, s, r, is_terminal in mdp.P[state][a]:
                action_values[a] += p * (r + discount * V[s])
        best_action = np.argmax(action_values)

        if chosen_actions != best_action:
            policy_stable = False
        policy[state] = np.eye(mdp.num_actions)[best_action]
    return policy


def policy_iteration(mdp, discount=0.99, theta=0.01):
    """ Computes the policy iteration (PI) algorithm.
    Arguments: MDP, discount factor, theta
    Returns: value function, policy
    """
    # initialization
    # Start from random policy
    policy = random_policy(mdp)
    # This is only here for the skeleton to run.
    # V = init_value(mdp)
    # TODO: Write your implementation here
    while True:
        # evaluate the current policy
        V = policy_evaluation(mdp=mdp, policy=policy, discount=discount, theta=theta)

        # Will be set to false if we make any changes to the policy
        policy_stable = True

        # improve the current policy by applying greedy search
        policy = policy_improvement(mdp=mdp, V=V, discount=discount)

        if policy_stable is True:
            return V, policy


def value_iteration(mdp, discount=0.99, theta=0.01):
    """ Computes the value iteration (VI) algorithm.
    Arguments: MDP, discount factor, theta
    Returns: value function, policy
    """
    # Init value function array
    V = init_value(mdp)

    # TODO: Write your implementation here

    # Get the greedy policy w.r.t the calculated value function
    while True:
        delta = 0
        for state in range(mdp.num_states):
            A = np.zeros(mdp.num_actions)
            for a in range(mdp.num_actions):
                for p, s, r, is_terminal in mdp.P[state][a]:
                    A[a] += p * (r + discount * V[s])
            best_action_value = max(A)
            delta = max(delta, np.abs(best_action_value - V[state]))
            V[state] = best_action_value
        if delta < theta:
            break
    policy = np.zeros([mdp.num_states, mdp.num_actions])
    for state in range(mdp.num_states):
        A = np.zeros(mdp.num_actions)
        for a in range(mdp.num_actions):
            for p, s, r, is_terminal in mdp.P[state][a]:
                A[a] += p * (r + discount * V[s])
        best_action = np.argmax(A)
        policy[state, best_action] = 1.0
    return V, policy


if __name__ == "__main__":
    # Create the MDP
    mdp = GridworldMDP([6, 6])
    discount = 0.99
    theta = 0.01

    # Print the gridworld to the terminal
    print('---------')
    print('GridWorld')
    print('---------')
    mdp.render()

    # Create a random policy
    V = init_value(mdp)
    policy = random_policy(mdp)
    # Do one step of policy evaluation and print
    print('----------------------------------------------')
    print('One step of policy evaluation (random policy):')
    print('----------------------------------------------')
    V = policy_evaluation_one_step(mdp, V, policy, discount=discount)
    print_value(V, mdp)

    # Do a full (random) policy evaluation and print
    print('---------------------------------------')
    print('Full policy evaluation (random policy):')
    print('---------------------------------------')
    V = policy_evaluation(mdp, policy, discount=discount, theta=theta)
    print_value(V, mdp)

    # Do one step of policy improvement and print
    # "Policy improvement" basically means "Take greedy action w.r.t given a given value function"
    print('-------------------')
    print('Policy improvement:')
    print('-------------------')
    policy = policy_improvement(mdp, V, discount=discount)
    print_deterministic_policy(policy, mdp)

    # Do a full PI and print
    print('-----------------')
    print('Policy iteration:')
    print('-----------------')
    V, policy = policy_iteration(mdp, discount=discount, theta=theta)
    print_value(V, mdp)
    print_deterministic_policy(policy, mdp)

    # Do a full VI and print
    print('---------------')
    print('Value iteration')
    print('---------------')
    V, policy = value_iteration(mdp, discount=discount, theta=theta)
    print_value(V, mdp)
    print_deterministic_policy(policy, mdp)