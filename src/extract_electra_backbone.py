import os
from argparse import Namespace, ArgumentParser
from glob import glob

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import pandas as pd
from transformers import AutoTokenizer

from data.context_sampling import get_sampler_class
from data.dataset import SampledBERTRegressionDataset
from data.utils import collate_heterogeneous_batch
from global_vars import device, CHECKPOINT_DIR, PREDICTIONS_DIR, EMBEDDINGS_DIR
from models.bert import SimpleBERTLikeClassifier

import torch.nn as nn


def get_dataset(params, tokenizer, partition):
    samplers = []
    sampler_params = []
    for sampler_config in params.samplers:
        samplers.append(get_sampler_class(sampler_config['type']))
        sampler_params.append(Namespace(**sampler_config))
    return SampledBERTRegressionDataset(params.dataset, partition=partition, label_columns=label_cols,
                                       tokenizer=tokenizer, meta_cols=['ID', 'author', 'story'], samplers=samplers,
                                        sampler_params = sampler_params)


def load_data(params, tokenizer, shuffle_train=True, devtest_batch_size=64):
    loaders = []
    for partition in ['train','val', 'test']:
        ds = get_dataset(params, tokenizer, partition=partition)
        shuffle = shuffle_train if partition=='train' else False
        batch_size = params.batch_size if partition=='train' else devtest_batch_size
        loaders.append(DataLoader(ds, shuffle=shuffle, batch_size=batch_size, collate_fn=collate_heterogeneous_batch))
    return tuple(loaders)


def init_model(seed, num_classes, backbone_id, load_cp=None, train=True):
    torch.manual_seed(seed)
    model = SimpleBERTLikeClassifier(num_classes=num_classes, backbone_id=backbone_id, return_embedding=True)
    if not load_cp is None:
        state_dict = torch.load(os.path.join(CHECKPOINT_DIR, load_cp), map_location=device)
        model.load_state_dict(state_dict, strict=False)
    model.to(device)
    if train:
        model.train()
    else:
        model.eval()
    return model


def extract(model, loader:DataLoader, label_names, pred_file_name, emb_dir):
    '''

    Extracts embeddings and predictions for a given model and DataLoader
    @param model: finetuned ELECTRA
    @param loader: data loader
    @param label_names: list of label names
    @param pred_file_name: file to save predictions in
    @param emb_dir: directory to save embeddings in
    @return: (None)
    '''
    model.eval()
    predictions = []
    embeddings = []

    gold_standard = []
    all_author_ids = []
    all_tale_ids = []
    all_sentence_ids = []

    sigmoid = nn.Sigmoid()

    with torch.no_grad():
        for i,batch in tqdm(enumerate(loader)):
            (ids, masks, lengths), ys, metas = batch
            logits, encodings = model((ids.to(device), masks.to(device), lengths.to(device)))

            preds = sigmoid(logits).detach().cpu().numpy()

            predictions.append(preds)
            embeddings.append(encodings.detach().cpu().numpy())
            gold_standard.extend(list(ys.detach().cpu().numpy()))
            all_sentence_ids.extend(metas[:,0])
            all_author_ids.extend(metas[:,1])
            all_tale_ids.extend(metas[:,2])

            if i % 5 == 0 and i > 0:
                embeddings = np.concatenate(embeddings)
                np.save(open(os.path.join(emb_dir, f'{i}.npy'), 'wb+'), embeddings)
                embeddings = []

    predictions = np.concatenate(predictions)
    gold_standard = np.array(gold_standard)

    dct = {'ID': all_sentence_ids, 'author':all_author_ids, 'tale':all_tale_ids}
    for i, label in enumerate(label_names):
        dct.update({f'gs_{label}':gold_standard[:,i], f'pred_{label}':predictions[:,i]})

    pred_df = pd.DataFrame(dct)

    if pred_file_name is None:
        pred_file_name = 'predictions'
    csv_dir = os.path.join(PREDICTIONS_DIR, params.dataset, params.name)
    os.makedirs(csv_dir, exist_ok=True)
    pred_df.to_csv(os.path.join(csv_dir, f"{pred_file_name}.csv"), index=False)

    if len(embeddings) > 0:
        embeddings = np.concatenate(embeddings)
        np.save(open(os.path.join(emb_dir, '50000.npy'), 'wb+'), embeddings)

    embedding_files = sorted(glob(f'{emb_dir}/*.npy'), key=lambda f: int(os.path.basename(f).split(".")[0]))
    arrays = [np.load(f) for f in embedding_files]
    emb_arr = np.concatenate(arrays)
    np.save(open(os.path.join(emb_dir, 'embeddings.npy'), 'wb+'), emb_arr)
    for f in embedding_files:
        os.remove(f)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--config_file', default='../sampling_params.yaml', help='Configuration file as in '
                                                                                 'electra_ft.py')
    parser.add_argument('--config_name', default='context_lr4', help='Name of a configuration in the configuration file')
    parser.add_argument('--dataset', default='main', help='Dataset name. Must be a directory under data/splits and '
                                                               'contain train.csv, val.csv, test.csv.')
    #parser.add_argument('--columns', nargs='+', required=False)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--name', required=True)

    args = parser.parse_args()
    config_file = args.config_file
    with open(config_file, "r") as f:
        yaml_params = yaml.safe_load(f)[args.config_name]

    # CLI and param file arguments
    args_dict = vars(args)
    for yaml_key, yaml_value in yaml_params.items():
        if yaml_key in args_dict.keys():
            args_dict[yaml_key] = yaml_value if args_dict[yaml_key] is None else args_dict[yaml_key]
        else:
            args_dict[yaml_key] = yaml_value

    return Namespace(**args_dict)


if __name__ == '__main__':
    params = parse_args()

    dataset = params.dataset
    label_cols = ['V_EWE', 'A_EWE']

    model = init_model(seed=42, num_classes=2, backbone_id=params.model_backend, load_cp=params.checkpoint, train=False)
    tokenizer = AutoTokenizer.from_pretrained(params.model_backend)

    data = get_dataset(params, tokenizer, partition='full')
    loader = DataLoader(data, shuffle=False, batch_size=32, collate_fn=collate_heterogeneous_batch)


    embedding_target_dir = os.path.join(EMBEDDINGS_DIR, dataset, params.name)
    os.makedirs(embedding_target_dir, exist_ok=True)

    predictions_target_dir = os.path.join(PREDICTIONS_DIR, dataset, params.name)

    extract(model, loader, label_cols, 'predictions', embedding_target_dir)