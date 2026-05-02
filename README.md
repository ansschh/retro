# retro

Retrosynthesis benchmark harness and stack — Caltech project.

## Goal

Beat every current SOTA retrosynthesis model on every public benchmark by a large margin, then add an execution-aware scoring layer trained on real lab data as the company moat.

Targeted baselines:
- **Multistep:** DMS (Flex Duo / Flash / Explorer XL), Retro\*, AiZynthFinder (MCTS / Retro\* / DFPN), RetroSynFormer, TempRe, SynPlanner.
- **Single-step:** ReactionT5, RetroCaptioner, EditRetro, Graph2Edits, Retroformer, Chemformer, LocalRetro, GLN, RetroSim, Molecular Transformer.

## Phases

1. **Benchmark harness + baseline reproduction** (this phase) — every baseline runs in our harness under one frozen protocol.
2. Strong single-step proposer.
3. Route search with learned value function.
4. Direct route generation, DMS-style+.
5. Candidate union + reranker.
6. Execution-aware scorer.

## Layout

```
benchmarks/        USPTO-50K, USPTO-FULL, PaRoutes n1/n5 split refs and loaders
baselines/         one subdir per baseline with a uniform CLI runner
metrics/           single-step metrics, route metrics, canonicalization
harness/           runner, leaderboard writer, protocol enforcement
configs/           protocol.yaml frozen evaluation invariants
scripts/           one-off ops download checkpoints, prep data, submit batches
results/           leaderboard.csv and small result files
data/              gitignored datasets on scratch
checkpoints/       gitignored baseline weights on scratch
eval_artifacts/    gitignored per-target candidate dumps
logs/              gitignored SLURM and runtime logs
```

## Caltech HPC bootstrap

This project lives at `/resnick/scratch/atiwari2/retro/` on Caltech Resnick HPC. Home is off-limits; everything goes to scratch via `env.sh` cache redirects.

```bash
cd /resnick/scratch/atiwari2
git clone https://github.com/ansschh/retro.git
cd retro
source env.sh
uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install -e .
uv pip install torch --index-url https://download.pytorch.org/whl/cu128
```

If you already have a working tree at `/resnick/scratch/atiwari2/retro/` from earlier scaffold work, sync it to GitHub instead of re-cloning:

```bash
cd /resnick/scratch/atiwari2/retro
git remote add origin https://github.com/ansschh/retro.git 2>/dev/null || true
git fetch origin
git reset --hard origin/master
```

That keeps the existing `.venv/`, `.local/`, `.cache/` and any other untracked dirs in place.

## Evaluation protocol

Every leaderboard row is reproduced under one frozen protocol — same split SHA, stock list SHA, search budget, beam size, time limit, augmentation policy, canonicalization rules, eval script git SHA.

See `configs/protocol.yaml` for the pinned set. Any baseline run that doesn't match the protocol SHA is excluded from the leaderboard.
