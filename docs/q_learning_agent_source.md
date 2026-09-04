# Q-learning Teaching Source

Q-learning is a model-free reinforcement learning algorithm for learning an
action-value function. It estimates how good it is to take an action in a state
and then continue with the best known future action. The method can learn from
experience without knowing the environment transition probabilities.

The objective is to learn an approximately optimal action-value function Q(s, a)
and use it to derive a greedy or epsilon-greedy policy. In a teaching GridWorld
example, the agent starts in a cell, chooses actions such as up, down, left, and
right, receives rewards, and updates the Q-table after every transition.

Q-learning assumes that the environment can be described as states, actions,
rewards, and next states. It also assumes that learning can be improved by
balancing exploration and exploitation. A common exploration strategy is
epsilon-greedy action selection: with probability epsilon the agent tries a
random action, and otherwise it chooses the action with the largest current
Q-value.

Inputs:

- a finite set of states S
- a finite set of actions A
- observed transitions (s, a, r, s')
- learning rate alpha
- discount factor gamma
- exploration rate epsilon
- number of training episodes

Outputs:

- a learned Q-table or action-value function
- an improved policy derived from Q
- episode rewards or learning-curve metrics

The core update rule is:

Q(s,a) <- Q(s,a) + alpha * [r + gamma * max_a' Q(s',a') - Q(s,a)]

The term in brackets is the temporal-difference error. It compares the current
Q-value with a target made from the immediate reward plus the best estimated
future value.

Pseudocode:

1. Initialize Q(s, a) to zero for every state-action pair.
2. For each episode, reset the environment and observe the starting state.
3. While the episode is not finished, choose an action using epsilon-greedy
   exploration.
4. Execute the action and observe reward r and next state s'.
5. Update Q(s, a) using the Q-learning update rule.
6. Move to the next state.
7. After training, return the Q-table and the greedy policy.

Important hyperparameters:

- alpha: learning rate, usually between 0 and 1
- gamma: discount factor for future rewards, usually between 0 and 1
- epsilon: exploration probability
- episodes: number of training episodes
- max_steps_per_episode: safety limit for long episodes

Supported teaching environments include small GridWorld tasks, FrozenLake-style
tabular environments, and CliffWalking. This source describes tabular
Q-learning, not deep Q-learning. If the state space is very large or continuous,
the table representation may not be practical.
