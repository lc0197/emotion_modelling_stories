 # Emotion Modelling in Written Stories
 
Python code for the paper *Emotion Modelling in Written Stories*. (ArXiv link will follow here).

## Data
The experiments use the data provided by Cecilia Alm [here](http://people.rc.rit.edu/~coagla/affectdata/index.html). In order to obtain the exact variant of the 
data that was utilised in the paper (including the newly proposed valence/arousal gold standard), please run the script ``get_data.py``. Note that you are downloading the data at 
your own risk. Please also make sure to review the [copyright notice](http://people.rc.rit.edu/~coagla/affectdata/notice.pdf). The data will be placed in the directory ``data/splits`` by the script.  

## Installation
A ``requirements.txt`` for the creation of a virtual environment is provided, as well as a conda requirements yaml named ``tales_env.yaml``.


## Overview
The repository contains scripts for finetuning ELECTRA on the dataset via different context windows and corresponding LSTM/Transformer experiments 
based on such finetuned ELECTRA models (cf. Paper).
In both cases, the model is initialised based on a ``.yaml`` configuration file.

## ELECTRA Finetuning
Minimal example:
``python src/electra_ft.py --config_file ft_params.yaml --config_name context_lr2`` 

This would load the model as specified in the ``context_lr2`` entry of ``ft_params.yaml``. 
The sampling strategy can only be configured in the ``.yaml`` file. For examples, please see the provided ``ft_params.yaml``. Other hyperparameters such as 
``max_epochs`` or ``batch_size`` can be overriden via command line arguments. For all parameters, see the ``parse_args`` 
method in ``electra_ft.py``. Parameters that are not model hyperparameters of the model or training process, 
e.g. ``--dataset``, can only be set via command line arguments.

By default, the model employs the "main" data split. This can be changed with the argument ``--dataset``.

A ``.json`` file with logs will be created in the ``results`` directory, the best checkpoint among all seeds will 
be kept under ``checkpoints``. 

## Context Modelling with LSTMs/Transformers
First, sentence embeddings must be computed with a fine-tuned ELECTRA model. Subsequently, they 
can be fed into a LSTM/LSTM+Transformer model.

### Extract Embeddings from finetuned ELECTRA models
A sentence embedding is the sentence's ``[CLS]`` representation. These are extracted with the ``extract_electra_backbone.py`` script which 
also computes and saves the corresponding predictions. The model parameters must be specified with ``.yaml`` configuration files as above. 
Moreover, a checkpoint to load and a name for the extracted embeddings must be given. ``--checkpoint`` must be a path leading to a ``.pt`` file, relative to the ``checkpoints`` directory.
A few checkpoints from the experiments reported in the paper can be found here: [https://drive.google.com/drive/folders/1cTFvwj9MVD6sHGIVqgkZvtSthHusZzlb](https://drive.google.com/drive/folders/1cTFvwj9MVD6sHGIVqgkZvtSthHusZzlb). 
The ``.zip`` file must be unzipped in the ``checkpoints`` directory, such that, ``checkpoints`` is, e.g., the parent directory of ``lr4``. 

Example: ``python src/extract_electra_backbone.py --config_file ft_params.yaml --config_name context_lr4 --checkpoint lr4/main/V_EWE_A_EWE/checkpoint.pt --name lr4_embeddings``  

The computed embeddings will be placed in the ``embeddings`` directory, the predictions in the folder ``predictions``.

A few precomputed embeddings can be found here: [https://drive.google.com/drive/folders/1Fdp_UjqJAcbhKX3asBtNsGI0HTpEpr-H?usp=sharing](https://drive.google.com/drive/folders/1Fdp_UjqJAcbhKX3asBtNsGI0HTpEpr-H?usp=sharing)
The unzipped directories such as ´´lr4´´ must be placed in ``embeddings/main``.

### Training LSTM/LSTM+Transformer
Example: ``python src/training_cont.py --config_file context_params.yaml --config_name baseline_lstm --embeddings lr4_embeddings``

where ``--embeddings`` denotes the name of previously extracted embeddings.

All the model and training hyperparameters are set in the configuration ``.yaml`` file but can be overriden with corresponding command line arguments. 
For details, please check the provided ``context_params.yaml`` and the ``parse_args`` method of the ``training_cont.py``
