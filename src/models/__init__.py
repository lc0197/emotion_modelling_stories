import os
from pathlib import Path

import torch

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
EMBEDDINGS_DIR = os.path.join(Path(os.path.dirname(os.path.abspath(__file__))).parent.parent, 'embeddings')