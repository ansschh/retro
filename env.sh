# Source this every shell to keep all caches off /resnick/home.
# Caltech HPC home is reserved; never write project data, caches, venvs, or weights there.

export RETRO_ROOT=/resnick/scratch/atiwari2/retro

export HF_HOME=$RETRO_ROOT/.cache/huggingface
export HUGGINGFACE_HUB_CACHE=$RETRO_ROOT/.cache/hub
export TRANSFORMERS_CACHE=$RETRO_ROOT/.cache/transformers
export HF_DATASETS_CACHE=$RETRO_ROOT/.cache/huggingface/datasets

export TORCH_HOME=$RETRO_ROOT/.cache/torch
export PIP_CACHE_DIR=$RETRO_ROOT/.cache/pip
export CONDA_PKGS_DIRS=$RETRO_ROOT/.cache/conda
export XDG_CACHE_HOME=$RETRO_ROOT/.cache/xdg
export MPLCONFIGDIR=$RETRO_ROOT/.cache/mpl

export WANDB_DIR=$RETRO_ROOT/.cache/wandb
export WANDB_CACHE_DIR=$RETRO_ROOT/.cache/wandb
export WANDB_DATA_DIR=$RETRO_ROOT/.cache/wandb

export PATH=$RETRO_ROOT/.local/bin:$PATH
