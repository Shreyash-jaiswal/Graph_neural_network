# Enforce pytorch version 1.6.0
import torch
if torch.__version__ != '1.6.0':
    pip uninstall torch -y
    pip uninstall torchvision -y
    pip install torch==1.6.0
    pip install torchvision==0.7.0

# Check pytorch version and make sure you use a GPU Kernel
python -c "import torch; print(torch.__version__)"
python -c "import torch; print(torch.version.cuda)"
python --version
nvidia-smi