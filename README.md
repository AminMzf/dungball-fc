# DungBall FC

An experimental playground for training fly-inspired agents to play soccer with a dung ball.

Four fly agents learn through 2v2 self play using dopamine-modulated plastic policies. The live WebGL match lab renders the game in 3D and shows team possession, action scores, reward, dopamine, training progress, and checkpoint evaluation. Match size can scale from 1v1 to the 5v5 end goal.

> [!IMPORTANT]
> The MVP uses a small synthetic plastic brain, not the full MaleCNS connectome. The interface is designed so a MaleCNS/DOOMFLY-compatible adapter can replace it later. Neural activity or changing weights alone are not evidence of biological learning.

## Quick start

Requires Python 3.11+ and no third-party packages.

```bash
python -m dungball_fc.server
```

Open <http://127.0.0.1:8000>. Training begins automatically as a 2v2 match. Use **Save** in the Model Vault to create a checkpoint. Click **Evaluate** beside any checkpoint to load its weights and freeze learning, so its true performance is visible without exploration or weight updates.

Run the tests:

```bash
python -m unittest discover -s tests -v
```

Run headless training:

```bash
python -m dungball_fc.train --episodes 1000 --team-size 2 --checkpoint trained-team
```

## What is implemented

- 3D WebGL stadium with fly players, broadcast camera, shadows, and lighting
- Symmetric multi-agent physics for 1v1 through 5v5
- Egocentric sensory encoding (distance and bearing; no absolute coordinates)
- Plastic action policy with an eligibility trace and dopamine-like reward signal
- Curriculum-friendly shaped rewards
- Frozen evaluation mode
- Save, import, load, and evaluate versioned JSON checkpoints
- JSON API and dependency-free live dashboard
- Deterministic unit tests

## Roadmap

1. Validate learned policy against naive and shuffled-reward controls.
2. Validate the current 2v2 curriculum before promoting policies to 5v5.
3. Add replay files, tournaments, and heatmaps.
4. Replace `PlasticBrain` behind the `Brain` protocol with a MaleCNS adapter.
5. Map rendered fly vision to biological visual inputs and descending-neuron activity to actions.

Large MaleCNS data and generated checkpoints belong in `data/` and `checkpoints/`; both are ignored by Git.

## License

MIT. MaleCNS data is distributed separately under its own license and is not included here.
