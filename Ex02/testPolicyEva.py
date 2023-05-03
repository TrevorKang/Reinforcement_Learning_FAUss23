import numpy as np
from lib import GridworldMDP, print_value, print_deterministic_policy, init_value, random_policy


def policy_evaluation_one_step(mdp, V, policy, discount=0.99):
    """ Computes one step of policy evaluation.
    Arguments: MDP, value function, policy, discount factor
    Returns: Value function of policy
    """
    # Init value function array
    # TODO: Write your implementation here
    V_new = V.copy()
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
    # Init value function array
    # TODO: Write your implementation here
    V = init_value(mdp)
    while True:
        delta = 0
        V_old = V.copy()
        V = policy_evaluation_one_step(mdp=mdp, V=V, policy=policy, discount=discount)
        delta = max(delta, np.max(np.abs(V_old - V)))
        if delta < theta:
            break
    return V


if __name__ == '__main__':
    mdp = GridworldMDP([6, 6])
    policy = random_policy(mdp)
    V = init_value(mdp)
    print(V.reshape([6, 6]))
    discount = 0.99
    theta = 0.01
    i = 0
    V = policy_evaluation_one_step(mdp, V, policy, discount=discount)
    V = policy_evaluation(mdp, policy, discount=discount, theta=theta)
    # for cnt in range(3):
    #     V_old = V.copy()
    #     delta = 0
    #     V = policy_evaluation_one_step(mdp, V, policy, discount)
    #     print(V.reshape([4, 4]))
    #     delta = max(delta, np.max(np.abs(V_old - V)))
    #     print(delta)
    print(V)