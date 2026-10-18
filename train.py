"""
train.py - Learn the forward kinematics of a 2-link planar robot arm with a
NumPy-only neural network, then check that the network also recovers the
arm's Jacobian.

Run:  python train.py
Outputs: results.json and three figures in ./figures/
"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from nn import MLP

L1, L2 = 0.30, 0.25  # link lengths in metres
rng = np.random.default_rng(42)


# ---------- ground truth: the robot's own maths ----------
def forward_kinematics(theta):
    """theta: (n, 2) joint angles [rad] -> (n, 2) end-effector position [m]."""
    t1, t2 = theta[:, 0], theta[:, 1]
    x = L1 * np.cos(t1) + L2 * np.cos(t1 + t2)
    y = L1 * np.sin(t1) + L2 * np.sin(t1 + t2)
    return np.stack([x, y], axis=1)


def analytic_jacobian(theta):
    """d(x, y)/d(theta1, theta2) for one configuration, shape (2, 2)."""
    t1, t2 = theta
    return np.array([
        [-L1 * np.sin(t1) - L2 * np.sin(t1 + t2), -L2 * np.sin(t1 + t2)],
        [ L1 * np.cos(t1) + L2 * np.cos(t1 + t2),  L2 * np.cos(t1 + t2)],
    ])


# ---------- data ----------
def make_data(n):
    theta = rng.uniform(-np.pi, np.pi, size=(n, 2))
    return theta, forward_kinematics(theta)


SCALE_IN = np.pi          # inputs scaled to [-1, 1]
SCALE_OUT = L1 + L2       # outputs scaled to [-1, 1]

th_train, y_train = make_data(4000)
th_val, y_val = make_data(500)
th_test, y_test = make_data(1000)

X_train, Y_train = th_train / SCALE_IN, y_train / SCALE_OUT
X_val, Y_val = th_val / SCALE_IN, y_val / SCALE_OUT
X_test = th_test / SCALE_IN

# ---------- model ----------
net = MLP([2, 64, 64, 2], seed=1)

# 1) verify backprop against finite differences before trusting it
chk = MLP([2, 5, 4, 2], seed=3)
gc_err = chk.gradient_check(X_train[:20], Y_train[:20])
print(f"Gradient check max relative error: {gc_err:.2e}")
assert gc_err < 1e-6, "backprop does not match numerical gradients"

# 2) train with mini-batch Adam and a cosine learning-rate decay
epochs, batch, lr0 = 600, 128, 3e-3
train_curve, val_curve = [], []
for ep in range(epochs):
    lr = 0.5 * lr0 * (1 + np.cos(np.pi * ep / epochs)) + 1e-5
    perm = rng.permutation(len(X_train))
    for i in range(0, len(perm), batch):
        idx = perm[i:i + batch]
        _, grads = net.gradients(X_train[idx], Y_train[idx])
        net.adam_step(grads, lr=lr)
    train_curve.append(net.mse(net.forward(X_train), Y_train))
    val_curve.append(net.mse(net.forward(X_val), Y_val))
    if ep % 100 == 0 or ep == epochs - 1:
        print(f"epoch {ep:4d}  train {train_curve[-1]:.3e}  val {val_curve[-1]:.3e}")

# ---------- evaluation ----------
pred = net.forward(X_test) * SCALE_OUT
err = np.linalg.norm(pred - y_test, axis=1)  # metres
print(f"Test position error: mean {err.mean()*1000:.2f} mm, "
      f"95th pct {np.percentile(err, 95)*1000:.2f} mm, max {err.max()*1000:.2f} mm")

# Jacobian: network (via chain rule) vs analytic
# net input is theta/pi and output is y/SCALE_OUT, so dy/dtheta = J_net * SCALE_OUT / SCALE_IN
J_err, J_norm = [], []
for th in th_test[:300]:
    J_net = net.jacobian(th / SCALE_IN) * SCALE_OUT / SCALE_IN
    J_true = analytic_jacobian(th)
    J_err.append(np.linalg.norm(J_net - J_true))
    J_norm.append(np.linalg.norm(J_true))
J_err, J_norm = np.array(J_err), np.array(J_norm)
rel_J = J_err / J_norm
print(f"Jacobian relative error (Frobenius): median {np.median(rel_J)*100:.2f}%, "
      f"mean {rel_J.mean()*100:.2f}%")

# ---------- figures ----------
os.makedirs("figures", exist_ok=True)

plt.figure(figsize=(5, 3.5))
plt.semilogy(train_curve, label="train")
plt.semilogy(val_curve, label="validation")
plt.xlabel("epoch"); plt.ylabel("MSE (normalised units)")
plt.title("Training curve"); plt.legend(); plt.grid(alpha=0.3)
plt.tight_layout(); plt.savefig("figures/loss_curve.png", dpi=150); plt.close()

fig, ax = plt.subplots(1, 2, figsize=(9, 4), sharex=True, sharey=True)
ax[0].scatter(y_test[:, 0], y_test[:, 1], s=4, c="tab:blue")
ax[0].set_title("True end-effector positions")
sc = ax[1].scatter(pred[:, 0], pred[:, 1], s=4, c=err * 1000, cmap="viridis")
ax[1].set_title("Network predictions (colour = error, mm)")
for a in ax:
    a.set_aspect("equal"); a.set_xlabel("x (m)"); a.grid(alpha=0.3)
ax[0].set_ylabel("y (m)")
plt.colorbar(sc, ax=ax[1], fraction=0.046)
plt.tight_layout(); plt.savefig("figures/workspace.png", dpi=150); plt.close()

# Jacobian entries: learned vs analytic
J_net_all, J_true_all = [], []
for th in th_test[:300]:
    J_net_all.append((net.jacobian(th / SCALE_IN) * SCALE_OUT / SCALE_IN).ravel())
    J_true_all.append(analytic_jacobian(th).ravel())
J_net_all, J_true_all = np.array(J_net_all), np.array(J_true_all)
plt.figure(figsize=(4.5, 4.5))
plt.scatter(J_true_all.ravel(), J_net_all.ravel(), s=4, alpha=0.5)
lim = [J_true_all.min(), J_true_all.max()]
plt.plot(lim, lim, "r--", lw=1, label="y = x")
plt.xlabel("analytic Jacobian entry"); plt.ylabel("network Jacobian entry")
plt.title("Learned vs analytic Jacobian"); plt.legend(); plt.grid(alpha=0.3)
plt.gca().set_aspect("equal")
plt.tight_layout(); plt.savefig("figures/jacobian.png", dpi=150); plt.close()

with open("results.json", "w") as f:
    json.dump({
        "gradient_check_max_rel_error": float(gc_err),
        "final_train_mse": float(train_curve[-1]),
        "final_val_mse": float(val_curve[-1]),
        "test_error_mm_mean": float(err.mean() * 1000),
        "test_error_mm_p95": float(np.percentile(err, 95) * 1000),
        "test_error_mm_max": float(err.max() * 1000),
        "jacobian_rel_error_median_pct": float(np.median(rel_J) * 100),
        "jacobian_rel_error_mean_pct": float(rel_J.mean() * 100),
    }, f, indent=2)
