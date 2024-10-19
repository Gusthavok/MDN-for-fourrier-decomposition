# Fourier-based Neural Network for Probability Distribution Estimation

The code is not functionnal. I stopped the devellopment after I found another solution for MadpodRacing.

## 1. Problem Description
We aim to model a function, denoted by **Π (Pi)**, which takes a set of parameters `(α1, α2, ..., αN)` and returns a corresponding probability distribution. To estimate this function, we have access to the random draws of **N random variables** `(X_n)` that follow the distributions defined by Π for each specific set of parameters `(α1_n, α2_n, ..., αN_n)`.

The goal is to model Π using a neural network, which outputs a representation of the probability distribution based on the input parameters.

## 2. Motivation
In many **reinforcement learning** problems, an intelligent agent must predict how its environment will evolve. This evolution is often stochastic or dependent on parameters that are unknown to the agent. To make informed decisions, the agent needs to estimate the range of possible future states of its environment. 

By modeling the probability distribution Π(α1, α2, ..., αN) based on observable parameters, we provide the agent with a way to understand and anticipate the stochastic behavior of the environment.

## 3. Model Description
To achieve this, we employ a **neural network** that takes in the parameters `(α1, α2, ..., αN)` and outputs `2M` parameters representing the **amplitudes and phases** of the first `M` Fourier basis functions.

### Step-by-step Breakdown:

1. **Input**: Each time a new variable `X` is drawn from the distribution Π(α1, ..., αN):
   - We compute the parameters `(G_1, G_2, ..., G_2M)` of the Fourier functions that model a **Dirac delta** centered at `X`. (In reality, we slightly adjust the phases to model a Dirac delta at 0, and then shift it to `X`.)

2. **Model Output**: The neural network outputs parameters `(F_1, F_2, ..., F_2M)` which represent a Fourier-based approximation of the target probability distribution. This function is noted as `F`.

3. **Combining Functions**: We compute the parameters `(H_1, H_2, ..., H_2M)` that represent a mixture of the predicted function `F` and the true distribution Dirac delta `G`. This mixture is controlled by a factor `γ_n`, and the combined function is defined as:
   \[
   H = (1 - γ_n) * F + γ_n * G
   \]

4. **Training**: We backpropagate through the neural network to minimize the distance between the predicted parameters `(F_1, F_2, ..., F_2M)` and the target parameters `(H_1, H_2, ..., H_2M)`. The goal is for the model's output `F` to gradually converge towards `H`, improving its approximation of the underlying probability distribution.

### Why Fourier?
The Fourier basis provides a powerful way to approximate complex functions, including probability distributions. By learning the amplitudes and phases of the Fourier series, the model can efficiently represent distributions with different degrees of complexity, making it well-suited for stochastic environments.

---

### 4. Future Work
- **Scalability**: Explore the extension of this model to higher-dimensional parameter spaces.
- **Additional Regularization**: Investigate methods to ensure better generalization, such as regularizing the Fourier coefficients.
- **Real-world Applications**: Apply the model in real-world reinforcement learning tasks where agents face environments with stochastic or uncertain dynamics.
