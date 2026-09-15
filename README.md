# DungBall FC

An experimental playground for training fly-inspired agents to play soccer with a dung ball.

This first version is deliberately small and testable: one fly learns to push a ball into a goal using a dopamine-modulated plastic policy. It includes a live browser dashboard showing the arena, action scores, reward, dopamine, and training progress.

> [!IMPORTANT]
> The MVP uses a small synthetic plastic brain, not the full MaleCNS connectome. The interface is designed so a MaleCNS/DOOMFLY-compatible adapter can replace it later. Neural activity or changing weights alone are not evidence of biological learning.

## Quick start

Requires Python 3.11+ and no third-party packages.

```bash
python -m dungball_fc.server
```

Open <http://127.0.0.1:8000>. Training begins automatically. Use **Pause**, **Reset brain**, or switch between training and evaluation. Evaluation freezes plasticity.

Run the tests:

```bash
python -m unittest discover -s tests -v
```

Run headless training:

```bash
python -m dungball_fc.train --episodes 1000 --seed 42
```

## What is implemented

- 2D field physics with a fly, dung ball, and goal
- Egocentric sensory encoding (distance and bearing; no absolute coordinates)
- Plastic action policy with an eligibility trace and dopamine-like reward signal
- Curriculum-friendly shaped rewards
- Frozen evaluation mode
- JSON API and dependency-free live dashboard
- Deterministic unit tests

## Roadmap

1. Validate learned policy against naive and shuffled-reward controls.
2. Add 1v1, then 2v2, and only then 5v5.
3. Add replay files, tournaments, heatmaps, and checkpoints.
4. Replace `PlasticBrain` behind the `Brain` protocol with a MaleCNS adapter.
5. Map rendered fly vision to biological visual inputs and descending-neuron activity to actions.

Large MaleCNS data and generated checkpoints belong in `data/` and `checkpoints/`; both are ignored by Git.

## License

MIT. MaleCNS data is distributed separately under its own license and is not included here.
