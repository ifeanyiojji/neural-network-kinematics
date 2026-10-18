# Project Guide: Learning Robot Kinematics

This guide walks through the project in everyday language, then connects each
step to the underlying mathematics. The project is a small, controlled study:
can a neural network learn the position of a simulated robot arm, and does its
derivative behave like the arm's known Jacobian?

## The problem

The robot has two rigid links, with lengths 0.30 m and 0.25 m. Its two joint
angles are the input. The position of the end of the second link is the output:

```text
x = L1 cos(theta1) + L2 cos(theta1 + theta2)
y = L1 sin(theta1) + L2 sin(theta1 + theta2)
```

These equations are the simulator's ground truth. The network is not given the
equations as a model; it sees example angle-position pairs and learns an
approximation from them.

## How the network learns

`nn.py` implements a multilayer perceptron (MLP) using NumPy arrays. It has two
hidden layers with 64 `tanh` units each and a linear output layer. Given one
angle pair, it successively multiplies by learned weight matrices, adds biases,
and applies `tanh` in the hidden layers to predict x and y.

The training script generates 4,000 training examples, 500 validation examples,
and 1,000 held-out test examples. Each angle is sampled uniformly from
`[-pi, pi]`. Inputs and outputs are scaled to values near `[-1, 1]`, which keeps
the optimization numerically well behaved. The validation set tracks learning
progress; the test set is reserved for the reported final position metrics.

For each training batch, the script:

1. Runs the inputs through the network to get predicted positions.
2. Measures mean squared error against the simulated positions.
3. Uses backpropagation, an application of the chain rule, to find how each
   weight and bias contributed to the error.
4. Uses Adam to update the parameters, while gradually reducing the learning
   rate with a cosine schedule.

## Checks beyond training loss

Training loss alone does not show that the implementation is correct. Before
training, `train.py` compares every parameter's backpropagated gradient in a
small network with a centered finite-difference estimate. A close match checks
the derivatives used by learning independently of the training result.

The project also checks the network's input Jacobian. A Jacobian is a matrix of
partial derivatives: here it describes how a small change in each joint angle
changes each end-effector coordinate. The robot's equations give an analytic
Jacobian, and the network computes its own by chaining derivatives through its
layers. Comparing the two tests whether the learned mapping has a useful local
slope, not just plausible output positions.

Input and output scaling changes the units of the network's derivative. The
training script converts it back to metres per radian before comparing it with
the analytic robot Jacobian.

## Reading the results

The checked-in `results.json` records one run. On its 1,000 test examples, the
mean position error is about 1.91 mm and the 95th-percentile error is about
4.02 mm. For 300 test configurations, the median relative Jacobian error is
about 1.8%, and the mean is about 2.8%. The plots in `figures/` show the loss
curves, predicted workspace, and Jacobian comparison.

These results support a limited conclusion: this network approximates the
simulated forward-kinematics function and its local derivatives for samples
from the same angle distribution. They do not show that it generalizes to a
physical robot, noisy measurements, a different arm, or angles outside that
distribution. The simulation is noiseless, and the errors have not been
characterized over multiple training seeds.

## Where to look in the code

- `nn.py`: model initialization, forward pass, mean squared error,
  backpropagation, Adam updates, gradient checks, and network Jacobian.
- `train.py`: arm equations, analytic Jacobian, random data generation,
  normalization, training loop, evaluation, and plots.
- `test_nn.py`: fast tests that compare gradients and Jacobians against
  numerical finite differences.
- `results.json`: the latest saved scalar metrics.

To reproduce the run, follow the setup and run instructions in `README.md`.