import numpy as np
from lib import GridworldMDP, print_value, print_deterministic_policy, init_value, random_policy
from dp import policy_evaluation


def policy_improvement(mdp, V, discount=0.99):
    """ Computes greedy policy w.r.t a given MDP and value function.
    Arguments: MDP, value function, discount factor
    Returns: policy
    """
    # TODO: Write your implementation here
    # Initialize a policy array in which to save the greed policy
    policy = np.zeros_like(random_policy(mdp))
    while True:
        # Will be set to false if we make any changes to the policy
        policy_stable = True
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

        if policy_stable is True:
            break
    return policy


if __name__ == '__main__':
    mdp = GridworldMDP([6, 6])
    discount = 0.99
    theta = 0.01
    mdp.render()
    # V = init_value(mdp)
    policy = random_policy(mdp)
    V = policy_evaluation(mdp, policy, discount=discount, theta=theta)
    print(V.reshape([6, 6]))
    policy = policy_improvement(mdp, V, discount=discount)

    print_deterministic_policy(policy, mdp)
