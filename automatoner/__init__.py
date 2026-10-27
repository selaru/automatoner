import os

__version__ = "0.1.0"

# Windows without developer mode cannot create symlinks; the Hugging Face cache then copies files. Silence the warning.
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
