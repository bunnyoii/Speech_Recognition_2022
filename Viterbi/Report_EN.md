# Speech_Recognition_2022

Tongji University · Class of 2022 · School of Computer Science and Technology · Software Engineering · Machine Intelligence Direction · Speech Recognition Coursework

Teacher: Ying Shen

Semester of instruction: 2024-2025, autumn semester

# Task: Viterbi Algorithm

One week, the teacher gave the following homework assignments:

| Monday | Tuesday | Wednesday | Thursday | Friday |
| :---: | :---: | :---: | :---: | :---: |
| A | C | B | A | C |

What did his mood curve look like most likely that week? Give the full process of computation.

We want to analyze the trend in the student’s mood changes, specifically the most likely emotional state sequence over these five days. We assume three possible mood states:
1. Good
2. Neutral
3. Bad

Given the transition and observation probabilities, we can use a Hidden Markov Model (HMM) and the Viterbi Algorithm to find the most probable mood curve based on the given sequence of homework types.

## Model Parameter Settings

- **States (Emotional States)**: `states = ['Good', 'Neutral', 'Bad']`
- **Observations (Homework Types)**: `observations = ['A', 'B', 'C']`

- **Transition Probability Matrix**: Represents the probability of switching between emotional states:

    ```python
    transition_prob = {
        'Good': {'Good': 0.2, 'Neutral': 0.3, 'Bad': 0.5},
        'Neutral': {'Good': 0.2, 'Neutral': 0.2, 'Bad': 0.6},
        'Bad': {'Good': 0.0, 'Neutral': 0.2, 'Bad': 0.8}
    }
    ```

- **Emission Probability Matrix**: Represents the probability of observing a specific homework type given a particular mood:

    ```python
    emission_prob = {
        'Good': {'A': 0.7, 'B': 0.2, 'C': 0.1},
        'Neutral': {'A': 0.3, 'B': 0.4, 'C': 0.3},
        'Bad': {'A': 0.0, 'B': 0.1, 'C': 0.9}
    }
    ```

- **Initial State Probability**: Each mood’s initial probability (uniformly distributed):

    ```python
    start_prob = {'Good': 1/3, 'Neutral': 1/3, 'Bad': 1/3}
    ```

## Solution Steps

### Define Observation Sequence and Initial Conditions

The observation sequence `obs_sequence = ['A', 'C', 'B', 'A', 'C']` represents the homework types over five days. The `viterbi` and `path` tables will store the maximum probability for each state at each time step and the optimal path, respectively.

### Initialization

At the first time step `t=0`:

For each mood state, calculate the maximum probability based on its initial probability and the first observation, `A`.
For example, for the `Good` state:

```scss
viterbi['Good', 0] = start_prob['Good'] * emission_prob['Good']['A'] = (1 / 3) * 0.7 ≈ 0.2333
```

Calculate the initial probability for each state accordingly.

### Iterative Computation

From `t=1` to `t=4`, fill in each column of the `viterbi` table iteratively:

For each mood state, consider all possible prior states and calculate the maximum probability of transitioning to the current state at time `t`.
For example, at `t=1` for the `Neutral` state:

From `Good` to `Neutral`:

```scss
viterbi['Good', 0] * transition_prob['Good']['Neutral'] * emission_prob['Neutral']['C']
```

Find the maximum probability and update the `viterbi` and `path` tables accordingly.

Repeat this process to populate the entire `viterbi` table across all time steps.

### Traceback for Optimal Path

At the last time step (`t=4`), identify the state with the highest probability. Trace back using the `path` table to find the most probable mood sequence from `t=4` back to `t=0`.

## Results and Mood Curve

### Initial State (t=0):

At the start (`t=0`), the highest probability is for the “Good” state (0.233333), followed by “Neutral” (0.100000), and then “Bad” with a probability of 0. This indicates the teacher likely started in a good or neutral mood rather than a bad one.

### Time Step t=1, Observation C:

With observation “C,” the probability for “Bad” becomes the highest (0.105000), transitioning from “Good.”
The probability for “Neutral” is 0.021000 (transitioning from “Good”), while “Good” has a lower probability of 0.004667 (transitioning from “Good”).
This indicates the teacher’s mood likely shifted toward a negative state at `t=1`.

### Time Step t=2, Observation B:

For observation “B,” “Neutral” and “Bad” states have equal probabilities of 0.008400, both from “Bad.”
The probability for “Good” is slightly lower at 0.000840, from transitioning out of “Neutral.”
This shows that the mood is likely between “Neutral” and “Bad.”

### Time Step t=3, Observation A:

At `t=3`, with observation “A,” the probability for “Good” increases (0.001176), transitioning from “Neutral.”
“Neutral” drops to 0.000504 (from “Bad”), and “Bad” is at 0.
This suggests a mood improvement, shifting towards a positive state.

### Time Step t=4, Observation C:

At the last time step with observation “C,” the probability for “Bad” is highest (0.000529), transitioning from “Good.”
“Neutral” has a probability of 0.000106, while “Good” has the lowest at 0.000024.
The final state trends toward “Bad,” suggesting a declining mood by the end of the week.

### Final Mood Curve

Based on the `viterbi` and `path` tables, the most probable mood curve for the student is:

| Monday | Tuesday | Wednesday | Thursday | Friday |
| :---: | :---: | :---: | :---: | :---: |
| Good | Neutral | Bad | Good | Neutral |

This mood curve suggests that the student’s most likely emotional states while completing the week’s assignments were Good (Monday), Neutral (Tuesday), Bad (Wednesday), Good (Thursday), and Neutral (Friday).

## Summary

The analysis shows that the teacher’s mood transitioned from “Good” to “Neutral” and gradually to “Bad” over the week. The Viterbi algorithm illustrates how observed homework types influenced the predicted mood changes, providing insight into the emotional trends for the week.

## Appendice

**Code Run Result**

Starting Viterbi Calculation:

Initialization:
t=0 | State=Good | Observation='A' | Initial Probability = Start Prob(0.33) * Emission(0.70) = 0.233333
t=0 | State=Neutral | Observation='A' | Initial Probability = Start Prob(0.33) * Emission(0.30) = 0.100000
t=0 | State=Bad | Observation='A' | Initial Probability = Start Prob(0.33) * Emission(0.00) = 0.000000

Time Step t=1 | Current Observation = 'C'
State=Good | Max Probability=0.004667 from previous state 'Good' (Transition Prob=0.20, Emission Prob=0.10)
State=Neutral | Max Probability=0.021000 from previous state 'Good' (Transition Prob=0.30, Emission Prob=0.30)
State=Bad | Max Probability=0.105000 from previous state 'Good' (Transition Prob=0.50, Emission Prob=0.90)

Time Step t=2 | Current Observation = 'B'
State=Good | Max Probability=0.000840 from previous state 'Neutral' (Transition Prob=0.20, Emission Prob=0.20)
State=Neutral | Max Probability=0.008400 from previous state 'Bad' (Transition Prob=0.20, Emission Prob=0.40)
State=Bad | Max Probability=0.008400 from previous state 'Bad' (Transition Prob=0.80, Emission Prob=0.10)

Time Step t=3 | Current Observation = 'A'
State=Good | Max Probability=0.001176 from previous state 'Neutral' (Transition Prob=0.20, Emission Prob=0.70)
State=Neutral | Max Probability=0.000504 from previous state 'Bad' (Transition Prob=0.20, Emission Prob=0.30)
State=Bad | Max Probability=0.000000 from previous state 'Bad' (Transition Prob=0.80, Emission Prob=0.00)

Time Step t=4 | Current Observation = 'C'
State=Good | Max Probability=0.000024 from previous state 'Good' (Transition Prob=0.20, Emission Prob=0.10)
State=Neutral | Max Probability=0.000106 from previous state 'Good' (Transition Prob=0.30, Emission Prob=0.30)
State=Bad | Max Probability=0.000529 from previous state 'Good' (Transition Prob=0.50, Emission Prob=0.90)

Completed Viterbi Table:

t=0 | State='Good' | Probability=0.233333
t=0 | State='Neutral' | Probability=0.100000
t=0 | State='Bad' | Probability=0.000000
t=1 | State='Good' | Probability=0.004667
t=1 | State='Neutral' | Probability=0.021000
t=1 | State='Bad' | Probability=0.105000
t=2 | State='Good' | Probability=0.000840
t=2 | State='Neutral' | Probability=0.008400
t=2 | State='Bad' | Probability=0.008400
t=3 | State='Good' | Probability=0.001176
t=3 | State='Neutral' | Probability=0.000504
t=3 | State='Bad' | Probability=0.000000
t=4 | State='Good' | Probability=0.000024
t=4 | State='Neutral' | Probability=0.000106
t=4 | State='Bad' | Probability=0.000529