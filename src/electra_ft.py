import copy
import json
import os
import random
from argparse import ArgumentParser, Namespace
from time import ctime, time

import torch
import yaml
from torch import nn
from torch.nn import MSELoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from data.context_sampling import get_sampler_class
from data.dataset import SampledBERTRegressionDataset
from data.utils import collate_heterogeneous_batch

import numpy as np
import pandas as pd

from eval.metrics import tales_metrics_continuous
from global_vars import CHECKPOINT_DIR, device, RESULTS_DIR, PREDICTIONS_DIR, RESULT_CSV_DIR
from models.bert import SimpleBERTLikeClassifier


def get_dataset(params, tokenizer, partition):
    samplers = []
    sampler_params = []
    for sampler_config in params.samplers:
        samplers.append(get_sampler_class(sampler_config['type']))
        sampler_params.append(Namespace(**sampler_config))
    return SampledBERTRegressionDataset(params.dataset, partition=partition, label_columns=params.label_cols,
                                        tokenizer=tokenizer, meta_cols=['ID', 'author', 'story'], samplers=samplers,
                                        sampler_params=sampler_params)


def load_data(params, tokenizer, shuffle_train=True, devtest_batch_size=64):
    loaders = []
    for partition in ['train', 'val', 'test']:
        ds = get_dataset(params, tokenizer, partition=partition)
        shuffle = shuffle_train if partition == 'train' else False
        batch_size = params.batch_size if partition == 'train' else devtest_batch_size
        loaders.append(DataLoader(ds, shuffle=shuffle, batch_size=batch_size, collate_fn=collate_heterogeneous_batch))
    return tuple(loaders)


def init_model(seed, num_classes, backbone_id, load_cp=None, train=True):
    torch.manual_seed(seed)
    model = SimpleBERTLikeClassifier(num_classes=num_classes, backbone_id=backbone_id)
    if not load_cp is None:
        state_dict = torch.load(os.path.join(CHECKPOINT_DIR, load_cp), map_location=device)
        model.load_state_dict(state_dict, strict=False)
    model.to(device)
    if train:
        model.train()
    else:
        model.eval()
    return model


def init_optimizer(model, params):
    return torch.optim.Adam(model.parameters(), lr=params.lr)


def training_epoch(model, optimizer, loss_fn, train_loader):
    start_seconds = time()
    model.train()
    model.to(device)
    losses = []

    sigmoid = nn.Sigmoid()

    for batch in tqdm(train_loader):
        optimizer.zero_grad()
        (ids, masks, lengths), ys, metas = batch

        logits = model((ids.to(device), masks.to(device), lengths.to(device))).float()
        preds = sigmoid(logits)

        loss = loss_fn(preds, ys.to(device).float())
        loss.backward()
        losses.append(loss.detach().cpu().item())
        optimizer.step()

    time_took = time() - start_seconds
    return {'model': model, 'loss': np.sum(np.array(losses)), 'time': int(time_took)}


def eval(model, loader, label_names, save_preds=False, pred_file_name=None):
    """
    Evaluation routine
    @param model: model to evaluate
    @param loader: loader for the evaluation data
    @param label_names: class names for human-readable results
    @param save_preds: save predictions?
    @param pred_file_name: file to save predictions to
    @return: dictionaries of metrics on the global, story and author-level
    """
    model.eval()
    predictions = []
    gold_standard = []
    all_author_ids = []
    all_tale_ids = []
    all_sentence_ids = []

    sigmoid = nn.Sigmoid()

    with torch.no_grad():
        for batch in tqdm(loader):
            (ids, masks, lengths), ys, metas = batch
            logits = model((ids.to(device), masks.to(device), lengths.to(device)))

            preds = sigmoid(logits).detach().cpu().numpy()

            predictions.append(preds)
            gold_standard.extend(list(ys.detach().cpu().numpy()))
            all_sentence_ids.extend(metas[:, 0])
            all_author_ids.extend(metas[:, 1])
            all_tale_ids.extend(metas[:, 2])

    predictions = np.concatenate(predictions)
    gold_standard = np.array(gold_standard)

    dct = {'ID': all_sentence_ids, 'author': all_author_ids, 'tale': all_tale_ids}
    for i, label in enumerate(label_names):
        dct.update({f'gs_{label}': gold_standard[:, i], f'pred_{label}': predictions[:, i]})

    pred_df = pd.DataFrame(dct)
    gs_cols = [c for c in pred_df.columns if str(c).startswith('gs_')]
    pred_cols = [c for c in pred_df.columns if str(c).startswith('pred_')]

    if save_preds:
        if pred_file_name is None:
            pred_file_name = 'predictions'
        csv_dir = os.path.join(PREDICTIONS_DIR, 'electra_ft', params.dataset, run_name)
        os.makedirs(csv_dir, exist_ok=True)
        pred_df.to_csv(os.path.join(csv_dir, f"{pred_file_name}.csv"), index=False)

    global_metrics = tales_metrics_continuous(predictions, gold_standard, label_names)
    tales = sorted(list(set(all_tale_ids)))
    tale_metrics = {}
    for t in tales:
        tale_df = pred_df[pred_df['tale'] == t]
        tale_metrics[t] = tales_metrics_continuous(predictions=tale_df[pred_cols].values,
                                                   gold_standards=tale_df[gs_cols].values,
                                                   labels=label_names)
    authors = sorted(list(set(all_author_ids)))
    author_metrics = {}
    for a in authors:
        auth_df = pred_df[pred_df['author'] == a]
        author_metrics[a] = tales_metrics_continuous(predictions=auth_df[pred_cols].values,
                                                     gold_standards=auth_df[gs_cols].values,
                                                     labels=label_names)

    return global_metrics, tale_metrics, author_metrics


def train_val_epoch(model, optimizer, train_loader, val_loader, loss_fn, label_names):
    """
    Unite one iteration of training and validation
    """
    epoch_dict = {'train': {}, 'val': {}}
    tr_results = training_epoch(model=model, optimizer=optimizer,
                                train_loader=train_loader, loss_fn=loss_fn)

    model = tr_results['model']
    loss = tr_results['loss']
    epoch_dict['train']['loss'] = loss
    time_took = tr_results['time']
    epoch_dict['train']['time'] = time_took

    print(f".........Validation.......")
    val_metrics, val_metrics_stories, val_metrics_authors = eval(model, val_loader, label_names=label_names,
                                                                 save_preds=False)

    epoch_dict['val']['global'] = val_metrics
    epoch_dict['val']['by_story'] = val_metrics_stories
    epoch_dict['val']['by_author'] = val_metrics_authors

    epoch_val = np.mean(np.array([val_metrics[l]['CCC'] for l in label_names]))
    epoch_dict['val']['monitored_metric'] = epoch_val

    print(epoch_dict)

    return epoch_dict, model


def training(model, optimizer, loss_fn, train_loader, val_loader, epochs, patience, cp_file):

    best_quality = -1234.
    patience_counter = 0
    best_epoch = -1

    log_dict = {'epochs': {}, 'summary': {}}

    for epoch in range(1, epochs + 1):

        print(f".........Training for Epoch {epoch}........")
        epoch_results, model = train_val_epoch(model=model, optimizer=optimizer,
                                               train_loader=train_loader, val_loader=val_loader, loss_fn=loss_fn,
                                               label_names=params.label_cols)

        print(f"Loss: {epoch_results['train']['loss']} (Took {epoch_results['train']['time']} seconds)")

        epoch_val = epoch_results['val']['monitored_metric']

        if epoch_val <= best_quality:
            patience_counter += 1
            if patience_counter > patience:
                print("Early stopping, aborting")
                break
        else:
            best_quality = epoch_val
            best_epoch = epoch
            torch.save(model.state_dict(), cp_file)
            patience_counter = 0

        log_dict['epochs'][epoch] = epoch_results

    log_dict['summary'] = {'best_quality': best_quality, 'best_epoch': log_dict['epochs'][best_epoch]}

    return log_dict


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--config_file', default='../ft_params.yaml', help='.yaml configuration file for model and '
                                                                           'training hyperparameters. '
                                                                                 'Cf. the provided ft_params.yaml.')
    parser.add_argument('--config_name', default='dummy_experiment', help='Configuration name in the given .yaml file. '
                                                                          'Cf. the provided ft_params.yaml')
    parser.add_argument('--dataset', default='main', help='Dataset name. Must be a directory under data/splits and '
                                                               'contain train.csv, val.csv, test.csv.')
    parser.add_argument('--label_cols', nargs='+', default=['V_EWE', 'A_EWE'])
    parser.add_argument('--result_csv', required=False, help='Optionally, store the results also as a row in a .csv file.'
                                                             'This file will be placed in a result_csvs directory')
    # may be used to override config files
    parser.add_argument('--model_backend', type=str, required=False,
                        help='Huggingface model ID, e.g. google/electra-base-discriminator')
    parser.add_argument('--max_epochs', type=int, required=False)
    parser.add_argument('--patience', type=int, required=False, help='Patience in early stopping')
    parser.add_argument('--seed', type=int, required=False, help='Initial seed. Seeds will be [seed, seed+num_seeds]')
    parser.add_argument('--num_seeds', type=int, required=False, help='Number of seeds to train with')
    parser.add_argument('--batch_size', type=int, required=False)
    parser.add_argument('--lr', type=float, required=False, help='learning_rate')
    parser.add_argument('--dropout', type=float, required=False, help='Dropout after sentence embedding.')
    parser.add_argument('--save_preds', action='store_true', help='Save test predictions')

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


def test(model, loader, label_names, save_preds=False, pred_file_name=None):
    log_dict = {}
    model.eval()
    global_metrics, story_metrics, author_metrics = eval(model, loader, label_names, save_preds=save_preds,
                                                         pred_file_name=pred_file_name)
    log_dict['global'] = global_metrics
    log_dict['by_story'] = story_metrics
    log_dict['by_author'] = author_metrics

    return log_dict


def create_dict_summary(d1, d2, name1, name2):
    '''
    Takes two (at least partly) identical structured dicts and collapses them into one dict, e.g.
    name1 = {a:1, b:2}, name2 = {a:3, b:4}   =>   {a:{name1:1, name2:3}, b:{name1:2, name2:4}}
    @param d1: first dict
    @param d2: second dict
    @param name1: name for first dict
    @param name2: name for second dict
    @return: collapsed dict
    '''
    overlap_keys = set(d1.keys()) & set(d2.keys())
    new_dict = {}
    for k in overlap_keys:
        v1 = d1[k]
        v2 = d2[k]
        if isinstance(v1, dict):
            if not isinstance(v2, dict):
                continue
            new_dict[k] = create_dict_summary(v1, v2, name1, name2)
        else:
            if isinstance(v2, dict):
                continue
            new_dict[k] = {}
            new_dict[k][name1] = v1
            new_dict[k][name2] = v2
    return new_dict


if __name__ == '__main__':
    np.random.seed(42)
    random.seed(42)

    params = parse_args()

    log_dict = {'config': vars(params), 'seeds': {}, 'summary': {}}

    num_classes = len(params.label_cols)

    tokenizer = AutoTokenizer.from_pretrained(params.model_backend)

    print('Preparing data.')
    train_loader, val_loader, test_loader = load_data(params, tokenizer, shuffle_train=True)

    loss_fn = MSELoss()

    cp_files = []

    best_seed = -1.
    best_metric = -1234.
    best_cp = None

    run_name = ctime(time()).replace(" ", "-").replace(":", "_") + "_" + params.config_name
    log_dir = os.path.join(RESULTS_DIR, run_name, params.dataset, "_".join(params.label_cols))
    log_file = os.path.join(log_dir, 'log.json')
    cp_dir = os.path.join(CHECKPOINT_DIR, run_name, params.dataset, "_".join(params.label_cols))
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(cp_dir, exist_ok=True)

    for seed in range(params.seed, params.seed + params.num_seeds):
        print(f"Seed {seed}")

        model = init_model(seed=seed, backbone_id=params.model_backend, num_classes=len(params.label_cols))

        cp_file = os.path.join(cp_dir, f'{seed}.pt')
        cp_files.append(cp_file)
        optimizer = init_optimizer(model, params)

        seed_log_dict = {'train_val': {}, 'test': {}}
        seed_log_dict['train_val'] = training(model, optimizer, loss_fn=loss_fn,
                                              epochs=params.max_epochs, patience=params.patience,
                                              train_loader=train_loader, val_loader=val_loader,
                                              cp_file=cp_file)

        best_quality = seed_log_dict['train_val']['summary']['best_quality']

        if best_quality > best_metric:
            best_metric = best_quality
            best_seed = seed
            best_cp = cp_file

        # test

        model = init_model(seed=seed, backbone_id=params.model_backend, num_classes=len(params.label_cols),
                           load_cp=cp_file)

        test_dict = test(model, test_loader, label_names=params.label_cols, save_preds=params.save_preds,
                         pred_file_name=seed)

        print(f'Test: {test_dict["global"]["V_EWE"]["CCC"]} CCC (Valence), '
              f'{test_dict["global"]["A_EWE"]["CCC"]} CCC (Arousal)')
        seed_log_dict['test'] = test_dict
        log_dict['seeds'][seed] = seed_log_dict
        with open(log_file, 'w+') as f:
            json.dump(log_dict, f)
        print()
    print()

    # delete all but best checkpoint
    for cp in cp_files:
        if not cp == best_cp and os.path.exists(cp):
            os.remove(cp)

    os.rename(best_cp, os.path.join(cp_dir, 'checkpoint.pt'))

    best_seed_dict = log_dict['seeds'][best_seed]
    best_seed_val = best_seed_dict['train_val']['summary']['best_epoch']['val']
    best_seed_test = best_seed_dict['test']
    log_dict['summary'] = create_dict_summary(best_seed_val, best_seed_test, 'val', 'test')
    # does not make sense, drop
    del (log_dict['summary']['by_story'])
    log_dict['summary']['by_story_val'] = best_seed_val['by_story']
    log_dict['summary']['by_story_test'] = best_seed_test['by_story']
    with open(log_file, 'w+') as f:
        json.dump(log_dict, f)

    if not params.result_csv is None:
        dct = vars(copy.deepcopy(params))
        for emo_dim in ['V', 'A']:
            for partition in ['val', 'test']:
                dct[f'{emo_dim}_EWE_CCC'] = log_dict['summary']['global'][f'{emo_dim}_EWE']['CCC'][partition]
        for k, v in dct.items():
            if isinstance(v, list):
                dct[k] = "_".join([str(s) for s in v])
            dct[k] = [v]

        # mean and standard deviations of results
        summarized_metrics = list(
            log_dict['seeds'][params.seed]['train_val']['summary']['best_epoch']['val']['global'][params.label_cols[0]]
            .keys())
        for lbl in params.label_cols:
            for mtr in summarized_metrics:
                res_val = np.array([log_dict['seeds'][seed]['train_val']['summary']['best_epoch']['val']['global']
                                    [lbl][mtr] for seed in range(params.seed, params.seed + params.num_seeds)])
                dct[f'mean_val_{lbl}_{mtr}'] = np.mean(res_val)
                dct[f'std_val_{lbl}_{mtr}'] = np.std(res_val)
                res_test = np.array([log_dict['seeds'][seed]['test']['global'][lbl][mtr] for seed in
                                     range(params.seed, params.seed + params.num_seeds)])
                dct[f'mean_test_{lbl}_{mtr}'] = np.mean(res_test)
                dct[f'std_test_{lbl}_{mtr}'] = np.std(res_test)

        new_result_df = pd.DataFrame(dct)

        target_csv = os.path.join(RESULT_CSV_DIR, params.result_csv)
        os.makedirs(os.path.dirname(target_csv), exist_ok=True)
        if os.path.exists(target_csv):
            old_result_df = pd.read_csv(target_csv)
            result_df = pd.concat([old_result_df, new_result_df])
        else:
            result_df = new_result_df
        result_df.to_csv(target_csv, index=False)
