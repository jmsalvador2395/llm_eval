from datasets import (
    load_dataset, DatasetDict, Dataset, concatenate_datasets
)
from nltk import sent_tokenize

from llm_eval.utils import files

def load_data(name, cfg, ds_info):

    text_col = ds_info['text_column']
    split = ds_info['split']
    cache = cfg.infill.get('ds_cache')
    seed = cfg.infill.get('seed', 100)

    special_sets = ['sind', 'roc', 'wikipedia']

    if name in special_sets:
        temp_ds = load_special_set(
            name, ds_info, cfg, cache, seed, split
        )
    else:
        name_args = ds_info.get('args', [])
        temp_ds = load_dataset(
            name,
            *name_args,
            cache_dir=cache,
        )

        # combine selected splits
        if isinstance(split, list):
            temp_ds = concatenate_datasets([
                temp_ds[splt] for splt in split]
            )
        else:
            temp_ds = temp_ds[split]

    return temp_ds[text_col]

def load_special_set(name, ds_info, cfg, cache, seed, split):
    match name:
        case 'sind':
            ds = load_sind(ds_info, cfg, cache, split, seed)
        case 'roc':
            ds = load_roc(ds_info, cfg, cache, split, seed)
        case 'wikipedia':
            ds = load_wiki(ds_info, cfg, cache, split, seed)
        case _:
            raise ValueError(
                'variable `name` did not match any special set names'
            )
    
    return ds

def load_sind(ds_info, cfg, cache, split, seed=None):

    sind_base = ds_info['path']
    sind_files = ds_info['file_map']
    for key in sind_files:
        sind_files[key] = f'{sind_base}/{sind_files[key]}'

    sind = load_dataset(
        'text', data_files=sind_files, cache_dir=cache,
    )
    sind = return_split(sind, split)
    return sind

def load_roc(ds_info, cfg, cache, split, seed=None):

    # load in ROCstories and randomly create splits
    #base_path = f'{files.project_root()}/data/ROCStories/'

    base_path = ds_info['path']
    def roc_xform(sample):
        sents = [sample[f'sentence{i}'] for i in range(1, 6)]
        return {'text': ' <eos> '.join(sents)}
    roc = load_dataset(base_path, cache_dir=cache)

    train_test = roc['train'].train_test_split(test_size=0.2, seed=seed)
    test_val = train_test['test'].train_test_split(
        test_size=0.5, seed=seed
    )
    roc = DatasetDict({
        'train': train_test['train'], 'validation': test_val['train'],
        'test': test_val['test'],
    })
    roc = return_split(roc, split)
    roc = roc.map(roc_xform).select_columns('text')
    return roc

def load_wiki(ds_info, cfg, cache, split, seed=None):

    min_sents = ds_info.get('min_sents', 0)

    # load in wikipedia
    wiki = load_dataset(
        "wikimedia/wikipedia", 
        "20231101.en", 
        cache_dir=cache,
    )
    train_test = wiki['train'].train_test_split(
        test_size=20000, seed=seed
    )
    test_val = train_test['test'].train_test_split(
        test_size=0.5, seed=seed
    )
    wiki = DatasetDict({
        'train': train_test['train'], 'validation': test_val['train'],
        'test': test_val['test'],
    })
    wiki = return_split(wiki, split)
    if min_sents > 0:
        wiki = wiki.filter(
            lambda x: len(sent_tokenize(x['text'])) > min_sents,
            num_proc=8,
        )

    return wiki

def return_split(ds, split):
    if isinstance(split, list):
        ds = concatenate_datasets([
            ds[splt] for splt in split]
        )
    else:
        ds = ds[split]
    return ds
