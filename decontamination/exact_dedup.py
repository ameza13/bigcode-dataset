from copy import deepcopy
from glob import glob
import json
import os
import shutil
import sys
import argparse

from utils.dataset_sharding import shard_dataset
# from utils.utils import add_dict

import datasets
import pandas as pd

"""
Examples:
python exact_dedup.py --dataset-path ../data/mbpp/mbpp.jsonl --output-dir ../output/ > test.log
python exact_dedup.py --dataset-path ~/workspace/evol-instruct-playground/outputs/merged_datasets/evol-instruct-92a0_merged.jsonl --output-dir ../output/ --ds-output-name open-platypus
"""

SHARD_SIZE = 1000 << 20  # 1GB

def dump_benchmarks(file_path: str, benchmark: dict):
    """
    Dump the dictionary of benchmark samples that are filtered out
    """
    with open(file_path, "w") as f:
        json.dump(benchmark, f, indent=4)

"""
sample: the instance to be checked
sample_idx: index of sample in the master dataset, not in batch
column: column name to take for comparison
filter_out: Dict[str, List[str]] mapping from benchmark name to list of strings that need to be filtered-out.

Returns:
    True, matching substrings 
    Fals, None
"""
def find_substrings(sample, sample_idx, column, filter_out):
    content = sample[column] # TO DO: update column name

    # For each substring, try to find it in the file (case insensitive)
    temp_substrings = [] 
    for benchmark, substrings in filter_out.items():
        for substring in substrings:
            temp_substrings.append(substring)

    # print(f"substring deleted from benchmark: {temp_substrings[sample_idx]}")
    del temp_substrings[sample_idx]

    matching_list = []
    for substring in temp_substrings:
        if substring.lower() in content.lower(): # It cannot be compared to itself because we already removed it from benchmark.
            matching_list.append(substring)

    is_duplicate = len(matching_list)>0

    # TEST
    if is_duplicate:
        print(f"SAMPLE {sample_idx}: {content}") 
        print(f"MATCHING INSTANCES:")  
        for match in matching_list:
            print(f"{match}")

    return is_duplicate, matching_list

# def aggregate_meta(tmp_meta_dir: str):
#     res = {}
#     for file in glob(f"{tmp_meta_dir}/*-meta.json"):
#         with open(file, "r") as f:
#             meta = json.load(f)
#         add_dict(res, meta)
#     return res

# def concatenate_meta(tmp_meta_dir: str):
#     res = []
#     for file in glob(f"{tmp_meta_dir}/*-excluded-data.json"):
#         with open(file, "r") as f:
#             meta = json.load(f)
#         res += meta
#     return res

class Meta:
    def __init__(self) -> None:
        self.meta_dict = dict()
    
    def update(self, lang: str, filter_reason: str):
        if lang not in self.meta_dict:
            self.meta_dict[lang] = {}
        if filter_reason not in self.meta_dict[lang]:
            self.meta_dict[lang][filter_reason] = 0
        self.meta_dict[lang][filter_reason] += 1

class SubstringFilterer(object):
    def __init__(
            self,
            output_dir: str,
            filter_out: dict,
            ds_name:str,
            column:str = "input",
            tmp_meta_dir = None,
            data_dir = None
    ) -> None:
        self.output_dir = output_dir
        self.tmp_meta_dir = tmp_meta_dir if tmp_meta_dir is not None else f"{output_dir}/tmp/meta"
        self.data_dir = data_dir if data_dir is not None else f"{output_dir}/data"
        os.makedirs(self.tmp_meta_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)

        # Save benchmark data
        self.excluded_data_cache = os.path.join(self.output_dir, "excluded-data.json")
        self.benchmarks_cache = os.path.join(output_dir, f"benchmarks-{ds_name}.json") # saves merged dataset, but in different format
        self.filter_out = filter_out
        dump_benchmarks(self.benchmarks_cache, self.filter_out)
        self.column = column
    
    def _find_duplicate(self, sample, sample_idx, column):
        is_duplicate, matched_substrings = find_substrings(sample=sample, sample_idx=sample_idx, column=column, filter_out=self.filter_out) 
        return is_duplicate, matched_substrings

    """
    -batch: a dictionary with a column name as key, and a list of batch values as values associted to the list
    {'input':['input', 'input', 'inputN']
    'code':['code1', 'code2', 'codeN']
    'task_id':[...],
    ...}
    -idx: list of idexes of elements from original data that are part of the batch
    -features: column names
    -res: output dictionary, it has the same structure than batch dictionary
    """
    def _find_duplicates(self, batch: dict, idx):
        meta = Meta()
        excluded_data = []

        duplicates_idx = []
        
        features = batch.keys()
        res = {k: [] for k in features}
            
        print(f"batch size: {len(idx)}") # TEST
        i = 0 
        for sample in zip(*[batch[k] for k in features]):
            sample = {k: v for k, v in zip(features, sample)}
            is_duplicate, matched_substrings = self._find_duplicate(sample=sample, sample_idx=idx[i],column=self.column)

            # TO DO: Build hashtable of duplicates {idx: [matched_substrings]}
            if is_duplicate: 
                duplicates_idx.append(idx[i])
            else:
                # Add to output
                for k in features:
                    res[k].append(sample[k])
            i+=1
        
        print(f"# of duplicates in batch: {len(duplicates_idx)}")
        # TO DO: How do we record the duplicates
        # Record Meta
        # with open(os.path.join(self.tmp_meta_dir, f"{idx[0]}-{idx[-1]}-meta.json"), "w") as f:
        #     json.dump(meta.meta_dict, f)
        # with open(os.path.join(self.tmp_meta_dir, f"{idx[0]}-{idx[-1]}-excluded-data.json"), "w") as f:
        #     json.dump(excluded_data, f, indent=4)
        return res

    def find_duplicates(self, ds, num_proc, batch_size):
        filtered = ds.map(
            self._find_duplicates,
            batched=True,
            batch_size=batch_size,
            with_indices=True,
            num_proc=num_proc,
            load_from_cache_file=False,
        )
        return filtered # Returns filtered DatasetDict

    # def finalize(self):
    #     # Dump meta
    #     meta = aggregate_meta(self.tmp_meta_dir)
    #     print(meta)
    #     with open(os.path.join(self.output_dir, "meta.json"), "w") as f:
    #         json.dump(meta, f)
    #     # Dump excluded-data.json
    #     meta = concatenate_meta(self.tmp_meta_dir)
    #     print("Number of excluded examples: ", len(meta))
    #     with open(self.excluded_data_cache, "w") as f:
    #         json.dump(meta, f)
    #     # delete temporary meta data
    #     shutil.rmtree(self.tmp_meta_dir)

    # Save shards
    def save(self, filtered, num_proc, name):
        shard_dataset(filtered, SHARD_SIZE, self.data_dir, num_proc=16, name=name)
    
    def run(self, dataset, num_proc, batch_size, name):
        filtered = self.find_duplicates(dataset, num_proc, batch_size)
        print("TOTAL ROWS IN DEDUP DATASET: ", len(filtered['train']))
        # print(filtered['train']['input']) #TEST
        # Finalize meta-data
        # self.finalize()
        # Save filtered dataset.
        self.save(filtered['train'], num_proc, name=name)

# Load merged dataset (json array)
def load_ds(file_path: str):
    instances = []
    with open(file_path) as f:
        for line in f:
            instances.append(json.loads(line))   
    return instances

# Convert to HF Dataset
def convert_jsonarray_to_hf_dataset(ds_list:list):
    df = pd.DataFrame.from_dict(ds_list) 
    ds_train_split = datasets.Dataset.from_pandas(df)
    ds = datasets.DatasetDict()
    ds['train'] = ds_train_split
    return ds

# Put data in benchmakr format
def merged_ds_strings(ds_list:list, column:str):
    return [sample[column] for sample in ds_list]

def arguments():
    parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     "--dataset-name",
    #     default="../data/mbpp/mbpp.jsonl",
    #     type=str,
    #     help="Name or path of the HF dataset to decontaminate"
    # )
    parser.add_argument(
        "--dataset-path",
        required=True,
        type=str,
        help="Name or path of the dataset to decontaminate"
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=str,
        help="Path to save output data and metadata"
    )
    parser.add_argument(
        "--num-proc",
        type=int,
        default=200,
        help="Number of processes"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10000,
        help="Size of batches passed to Dataset.map"
    )
    parser.add_argument(
        "--ds-output-name",
        required=True,
        type=str,
        help="Name to identify output dataset"
    )
    return parser.parse_args()


def main():
    args = arguments()
    
    # Load merged dataset
    merged_ds_as_list = load_ds(file_path=args.dataset_path)
    # Convert merged dataset to HF datasets format
    ds = convert_jsonarray_to_hf_dataset(ds_list=merged_ds_as_list)

    # Create benchmark ds
    FILTER_OUT = {
        "merged_inputs": merged_ds_strings(ds_list=merged_ds_as_list, column="input"),
        # "merged_outputs": merged_ds_strings(ds_list=merged_ds_as_list, column="output")
    }   

    # TEST
    total_rows = len(ds['train']) 
    print(f"TOTAL ROWS IN MERGED DATASET: {total_rows}") 

    # TO DO: Remove the use of benchmark data from here, we will use the merged dataset.
    filterer = SubstringFilterer(
        output_dir=args.output_dir,
        filter_out=FILTER_OUT,
        ds_name=args.ds_output_name,
        column = 'input'
    )

    # Run filtering
    filterer.run(ds, args.num_proc, args.batch_size, args.ds_output_name)

if __name__ == "__main__":
    main()
