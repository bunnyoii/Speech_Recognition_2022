import numpy as np

# Define the states in the Hidden Markov Model (HMM)
states = ['Good', 'Neutral', 'Bad']

# Define the possible observations
observations = ['A', 'B', 'C']

# Sequence of observations that we want to evaluate
obs_sequence = ['A', 'C', 'B', 'A', 'C']

# Transition probabilities between states
# transition_prob[state][next_state] gives the probability of moving to next_state from state
transition_prob = {
    'Good': {'Good': 0.2, 'Neutral': 0.3, 'Bad': 0.5},
    'Neutral': {'Good': 0.2, 'Neutral': 0.2, 'Bad': 0.6},
    'Bad': {'Good': 0.0, 'Neutral': 0.2, 'Bad': 0.8}
}

# Emission probabilities (probability of observing a specific observation from a state)
# emission_prob[state][observation] gives the probability of seeing observation in state
emission_prob = {
    'Good': {'A': 0.7, 'B': 0.2, 'C': 0.1},
    'Neutral': {'A': 0.3, 'B': 0.4, 'C': 0.3},
    'Bad': {'A': 0.0, 'B': 0.1, 'C': 0.9}
}

# Number of states and observations
n_states = len(states)
n_obs = len(obs_sequence)

# Initialize the Viterbi table with zeros
# viterbi[state, observation_index] will store the max probability of any path that ends in state at time observation_index
viterbi = np.zeros((n_states, n_obs))

# Path table to store the most likely previous state for each state at each time step
# path[state, observation_index] stores the previous state that led to state at time observation_index with the highest probability
path = np.zeros((n_states, n_obs), dtype=int)

# Define the initial probability for each state (assuming uniform distribution here)
start_prob = {state: 1 / n_states for state in states}

print("Starting Viterbi Calculation:\n")

# Initialization step: Fill in the first column of the Viterbi table with initial probabilities
print("Initialization:")
for s in range(n_states):
    state = states[s]
    # Initial probability for each state times the probability of observing the first observation in that state
    viterbi[s, 0] = start_prob[state] * emission_prob[state][obs_sequence[0]]
    # Initialize the path for each state
    path[s, 0] = s
    print(f"t=0 | State={state} | Observation='{obs_sequence[0]}' | "
          f"Initial Probability = Start Prob({start_prob[state]:.2f}) * Emission({emission_prob[state][obs_sequence[0]]:.2f}) = {viterbi[s, 0]:.6f}")

# Iterative step: Fill in the rest of the Viterbi table column by column
for t in range(1, n_obs):
    print(f"\nTime Step t={t} | Current Observation = '{obs_sequence[t]}'")
    for s in range(n_states):
        state = states[s]
        # For each state, calculate the maximum probability of reaching that state at time t from any previous state
        max_prob, max_state = max(
            (viterbi[prev_state, t-1] * transition_prob[states[prev_state]][state] * emission_prob[state][obs_sequence[t]], prev_state)
            for prev_state in range(n_states)
        )
        # Store the maximum probability and the corresponding previous state
        viterbi[s, t] = max_prob
        path[s, t] = max_state
        print(f"State={state} | Max Probability={viterbi[s, t]:.6f} from previous state '{states[max_state]}' "
              f"(Transition Prob={transition_prob[states[max_state]][state]:.2f}, "
              f"Emission Prob={emission_prob[state][obs_sequence[t]]:.2f})")

# Final step: Output the complete Viterbi table for reference
print("\nCompleted Viterbi Table:\n")
for t in range(n_obs):
    for s in range(n_states):
        print(f"t={t} | State='{states[s]}' | Probability={viterbi[s, t]:.6f}")
