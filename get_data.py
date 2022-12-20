import os
import shutil
from itertools import permutations
from pathlib import Path
from shutil import rmtree

import requests
import tarfile
import pandas as pd
from glob import glob

data_dir = os.path.join(Path(os.path.abspath(__file__)).parent, 'data')
downloaded_dir = os.path.join(data_dir, 'downloaded')
os.makedirs(downloaded_dir, exist_ok=True)

authors = ['Grimms', 'HCAndersen', 'Potter']
potter_gz = "http://people.rc.rit.edu/~coagla/affectdata/Potter.tar.gz"
andersen_gz = "http://people.rc.rit.edu/~coagla/affectdata/HCAndersen.tar.gz"
grimm_gz = "http://people.rc.rit.edu/~coagla/affectdata/Grimms.tar.gz"
gzs = [potter_gz, andersen_gz, grimm_gz]

def download(url, target_file):
    r = requests.get(url, allow_redirects=True)
    os.makedirs(os.path.dirname(target_file), exist_ok=True)
    open(target_file, 'wb').write(r.content)

def emmood_to_text(emmood_file):
    with open(emmood_file, 'r') as f:
        emmood_rows = [l.replace('\n', '') for l in f.readlines()]
    return [row[row.rfind('\t')+1:] for row in emmood_rows]

for gz in gzs:
    target_file = os.path.join(downloaded_dir, gz.split("/")[-1])
    if not os.path.exists(target_file):
        download(gz, target_file)
    tar = tarfile.open(target_file)
    tar.extractall(downloaded_dir)
    tar.close()


tmp_dir = os.path.join(data_dir, '.temp')
os.makedirs(tmp_dir, exist_ok=True)
to_remove_df = pd.read_csv(os.path.join(data_dir, 'provided', 'remove_sentences.csv'))

for author in authors:
    emmood_files = glob(os.path.join(downloaded_dir, author, 'emmood', '*.emmood'))
    for emmood_file in emmood_files:
        try:
            text = emmood_to_text(emmood_file)
            story_name = os.path.basename(emmood_file).replace('.emmood','')
            # remove sentences?
            to_remove = to_remove_df[to_remove_df.story==story_name]
            if len(to_remove) > 0:
                text = [t for t in text if not t in to_remove.sentence.values]

            df = pd.DataFrame({'author':[author]*len(text), 'story':[story_name]*len(text), 'text':text})
            df.to_csv(os.path.join(tmp_dir, f'{story_name}.csv'), index=False)
        # fails for two tales
        except:
            pass

# partition the stories
main_split_dir = os.path.join(data_dir, 'splits', 'main')
os.makedirs(main_split_dir, exist_ok=True)

main_partition_df = pd.read_csv(os.path.join(data_dir, 'provided', 'partitioning.csv'))
partition_lists = {'train':[], 'val':[], 'test':[]}
author_lists = {'Grimms':[], 'HCAndersen':[], 'Potter':[]}

gold_standard_df = pd.read_csv(os.path.join(data_dir, 'provided', 'gold_standard.csv'))

for _,row in main_partition_df.iterrows():
    story = row['story']
    author = row['author']
    partition = row['partition']
    story_text = pd.read_csv(os.path.join(tmp_dir, f'{story}.csv')).text.values
    start_positions = list(range(len(story_text)))
    story_gs = gold_standard_df[gold_standard_df.story==story].iloc[:,-2:].values
    story_ids = gold_standard_df[gold_standard_df.story==story]['ID'].values
    assert story_gs.shape[0] == len(story_text)
    story_len = len(story_text)
    story_df = pd.DataFrame({'ID':story_ids,
                             'story':[story]*story_len,
                             'start_position': start_positions,
                             'author':[author]*story_len,
                             'text':story_text,
                             'V_EWE':list(story_gs[:,0]),
                             'A_EWE':list(story_gs[:,1])})
    partition_lists[partition].append(story_df)
    author_lists[author].append(story_df)

for partition, dfs in partition_lists.items():
    df = pd.concat(dfs)
    df.to_csv(os.path.join(main_split_dir, f'{partition}.csv'), index=False)

for author in author_lists.keys():
    author_df = pd.concat(author_lists[author])
    author_lists[author] = author_df

# all author permutations
perms = permutations(sorted(list(author_lists.keys())))
for perm in perms:
    split_name = '_'.join(perm)
    split_dir = os.path.join(data_dir, 'splits', split_name)
    os.makedirs(split_dir, exist_ok=True)
    author_lists[perm[0]].to_csv(os.path.join(split_dir, 'train.csv'), index=False)
    author_lists[perm[1]].to_csv(os.path.join(split_dir, 'val.csv'), index=False)
    author_lists[perm[2]].to_csv(os.path.join(split_dir, 'test.csv'), index=False)

# create the full.csvs
for split in [d for d in sorted(glob(os.path.join(data_dir, 'splits', '*'))) if os.path.isdir(d)]:
    train_df = pd.read_csv(os.path.join(split, 'train.csv'))
    val_df = pd.read_csv(os.path.join(split, 'val.csv'))
    test_df = pd.read_csv(os.path.join(split, 'test.csv'))
    full_df = pd.concat([train_df,  val_df, test_df])
    full_df.sort_values(by='ID', inplace=True)
    full_df.to_csv(os.path.join(split, 'full.csv'), index=False)

# retain the copyright notice
shutil.copyfile(os.path.join(data_dir, 'downloaded', 'Potter', 'readme.txt'), os.path.join(data_dir, 'splits', 'readme.txt'))
rmtree(tmp_dir)