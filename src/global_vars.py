import torch
import os
from pathlib import Path


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

ROOT_DIR = Path(os.path.dirname(os.path.abspath(__file__))).parent

CHECKPOINT_DIR = os.path.join(ROOT_DIR, 'checkpoints')
DATA_DIR = os.path.join(ROOT_DIR, 'data', 'splits')

LOGS_DIR = os.path.join(ROOT_DIR, 'logs')
RESULTS_DIR = os.path.join(ROOT_DIR, 'results')
RESULT_CSV_DIR = os.path.join(ROOT_DIR, 'result_csvs')
PREDICTIONS_DIR = os.path.join(ROOT_DIR, 'predictions')

EMBEDDINGS_DIR = os.path.join(ROOT_DIR, 'embeddings')

MAX_LENGTH = 512