import argparse
import time
import numpy as np
import matplotlib.pyplot as plt

try:
    import poe as poe
except Exception as e:
    poe = None


def run_native(args):
    # Fallback pure NumPy implementation if pybind not built
    rng = np.random.default_rng(args.seed)
    a = rng.uniform(0.1, 5.0, size=args.agents)
    b = rng.normal(0.0, 5.0, size=args.agents)

    x_star = np.sum(a * b) / np.sum(a)

    traj = []
    if args.mode == 'naive':
        x = np.mean(b)
        traj.append(x)
    else:
        x = 0.0
        traj.append(x)
        for _ in range(args.iters):
            g = 2.0 * a * (x - b)
            x = x - args.lr * np.sum(g)
            traj.append(x)
    return x, x_star, traj


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--agents', type=int, default=100)
    parser.add_argument('--iters', type=int, default=1000)
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--mode', type=str, default='collaborative', choices=['naive', 'collaborative'])
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'gpu'])
    args = parser.parse_args()

    xs = []
    trajs = []
    t0 = time.time()

    for r in range(args.repeats):
        if poe is not None:
            cfg = poe.RunConfig()
            cfg.numAgents = args.agents
            cfg.iterations = args.iters
            cfg.seed = args.seed + r
            cfg.learningRate = args.lr
            cfg.mode = args.mode
            cfg.device = args.device
            engine = poe.OptimizationEngine(cfg)
            res = engine.run()
            xs.append(res.xFinal)
            trajs.append(res.xTrajectory)
            x_star = res.xStar
        else:
            x, x_star, traj = run_native(args)
            xs.append(x)
            trajs.append(traj)

    t1 = time.time()

    print(f"x*: {x_star:.6f}, mean x_final: {np.mean(xs):.6f}, std: {np.std(xs):.6f}")
    print(f"total wall time: {(t1-t0)*1000:.2f} ms over {args.repeats} runs")

    # Visualization
    max_len = max(len(t) for t in trajs)
    plt.figure(figsize=(6,4))
    for t in trajs:
        plt.plot(t, alpha=0.6)
    plt.axhline(x_star, color='k', linestyle='--', label='x*')
    plt.xlabel('iteration')
    plt.ylabel('x')
    plt.title('Convergence Trajectories')
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
