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
import time

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

    if len(column)>0:
        content = sample[column]
    else:
        content = f"{sample['input']} {sample['output']}"
        # content = f"{sample['text']} {sample['code']}"

    # # For each substring, try to find it in the file (case insensitive)
    # temp_substrings = [] 
    # # print(type(filter_out.items())) # TEMP
    # for benchmark, substrings in filter_out.items():
    #     for substring in substrings:
    #         temp_substrings.append(substring)

    # # print(f"substring deleted from benchmark: {temp_substrings[sample_idx]}")
    # del temp_substrings[sample_idx]

    # matching_list = []
    # for substring in temp_substrings:
    #     if substring.lower() in content.lower(): # It cannot be compared to itself because we already removed it from benchmark.
    #         matching_list.append(substring)
    ###
    matching_list = []
    for benchmark, substrings in filter_out.items():
        for idx, substring in zip(range(len(substrings)), substrings):
            if (substring.lower() in content.lower()) and idx != sample_idx: # It cannot be compared to itself.
                        print(f"valid matching between current idx:{idx} sampe_idx:{sample_idx}") # TEST
                        matching_list.append(substring)           

    is_duplicate = len(matching_list)>0       
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

# class Meta:
#     def __init__(self) -> None:
#         self.meta_dict = dict()
    
#     def update(self, lang: str, filter_reason: str):
#         if lang not in self.meta_dict:
#             self.meta_dict[lang] = {}
#         if filter_reason not in self.meta_dict[lang]:
#             self.meta_dict[lang][filter_reason] = 0
#         self.meta_dict[lang][filter_reason] += 1

class SubstringFilterer(object):
    def __init__(
            self,
            output_dir: str,
            filter_out: dict,
            ds_name:str,
            column:str = "",
            tmp_meta_dir = None,
            data_dir = None
    ) -> None:
        self.output_dir = output_dir
        self.tmp_meta_dir = tmp_meta_dir if tmp_meta_dir is not None else f"{output_dir}/tmp/meta"
        self.data_dir = data_dir if data_dir is not None else f"{output_dir}/data"
        os.makedirs(self.tmp_meta_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)
        self.ds_name = ds_name
        # Save benchmark data
        self.excluded_data_cache = os.path.join(self.output_dir, "excluded-data.json")
        self.benchmarks_cache = os.path.join(output_dir, f"benchmarks-{self.ds_name}.json") # saves merged dataset, but in different format
        self.filter_out = filter_out
        dump_benchmarks(self.benchmarks_cache, self.filter_out)
        self.column = column
    
    def _find_duplicate(self, sample, sample_idx, column):
        is_duplicate, matched_substrings = find_substrings(sample=sample, sample_idx=sample_idx, column=column, filter_out=self.filter_out) 
        return is_duplicate, matched_substrings

    """
    -batch: a dictionary with a column name as key, and a list of batch values as values associted to the list
    {'input':['input1', 'input2', 'inputN']
    'output':['output1', 'output2', 'outputN']
    'task_id':[...],
    ...}
    -idx: list of idexes of elements from original data that are part of the batch
    -features: column names
    -res: output dictionary, it has the same structure than batch dictionary
    """

    def _find_duplicates(self, batch: dict, idx):
        # meta = Meta()

        excluded_data = []
        duplicates_idx = []
        
        features = batch.keys()
        res = {k: [] for k in features}
            
        # TEST
        # print(f"batch keys: {features}")
        # print(f"# of keys in batch: {len(batch.items())}")
        print(f"Size of list of idxs: {len(idx)}")
        print(f"idx from {idx[0]} to {idx[len(idx)-1]}")
        print(f"# of Inputs in batch: {len(batch['input'])}")
        print(f"# of Outputs in batch: {len(batch['output'])}")
        # print(f"First Input in batch: {batch['input'][0]}")
        # print(f"First Output in batch: {batch['output'][0]}")
        
        i = 0 
        for sample in zip(*[batch[k] for k in features]):
            sample = {k: v for k, v in zip(features, sample)}
            is_duplicate, matched_substrings = self._find_duplicate(sample=sample, sample_idx=idx[i],column=self.column)

            if is_duplicate: 
                # TEST
                # print(f"SAMPLE {idx[i]}: {sample}") 
                # print(f"MATCHING INSTANCES:")  
                # for match in matched_substrings:
                #     print(f"{match}")

                # Track duplicates original idx
                duplicates_idx.append(idx[i])
                # Track data to be excluded:
                duplicate_info = {'duplicate_idx':idx[i], 'matches':matched_substrings}
                excluded_data.append(duplicate_info)
            else:
                # Add non duplicate to output
                for k in features:
                    res[k].append(sample[k])
            i+=1
        
        print(f"# of duplicates in batch: {len(duplicates_idx)}")

        # Record Meta
        # with open(os.path.join(self.tmp_meta_dir, f"{idx[0]}-{idx[-1]}-meta.json"), "w") as f:
        #     json.dump(meta.meta_dict, f)
        with open(os.path.join(self.tmp_meta_dir, f"{idx[0]}-{idx[-1]}-excluded-data.json"), "w") as f:
            json.dump(excluded_data, f, indent=4)
        return res

    def find_duplicates(self, ds, num_proc, batch_size):
        filtered = ds.map(
            self._find_duplicates,
            batched=True,
            batch_size=batch_size, # default batch_size=10,000
            with_indices=True, # Provide example indices to function. Note that in this case the signature of function should be def function(example, idx[, rank]): ....
            num_proc=num_proc, # default num_proc=200 / Max number of processes when generating cache. Already cached shards are loaded sequentially.
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
        shard_dataset(filtered, SHARD_SIZE, self.data_dir, num_proc=16, name=name) #TO CHECK: num_proc is always 16 for sharding
    
    def run(self, dataset, num_proc, batch_size): 
        filtered = self.find_duplicates(dataset, num_proc, batch_size)
        print("TOTAL ROWS IN DEDUP DATASET: ", len(filtered['train']))
        # print(filtered['train']['input']) #TEST
        # Finalize meta-data
        # self.finalize()
        # Save filtered dataset.
        self.save(filtered['train'], num_proc, name=self.ds_name) 

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

def merged_ds_strings(ds_list:list, c_input:str, c_output:str):      
    return [f"{sample[c_input]} {sample[c_output]}" for sample in ds_list]

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
        # default=1,
        default=os.cpu_count(), #96
        help="Number of processes"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=0,
        help="Size of batches passed to Dataset.map"
    )
    parser.add_argument(
        "--ds-output-name",
        required=True,
        type=str,
        help="Name to identify output dataset"
    )
    parser.add_argument(
        "--column-to-compare",
        type=str,
        default="", # It will take input and output
        help="Column for comparison."
    )
    return parser.parse_args()


def main():
    args = arguments()
    
    # Load merged dataset
    merged_ds_as_list = load_ds(file_path=args.dataset_path)
    # Convert merged dataset to HF datasets format
    ds = convert_jsonarray_to_hf_dataset(ds_list=merged_ds_as_list)

    # Create benchmark ds
    if len(args.column_to_compare)>0:
        if 'input' in args.column_to_compare:
        # if 'text' in args.column_to_compare:
            FILTER_OUT = {
                "merged_inputs": merged_ds_strings(ds_list=merged_ds_as_list, column="input")
                # "merged_inputs": merged_ds_strings(ds_list=merged_ds_as_list, column="text"),
            }
        elif 'output' in args.column_to_compare:
            FILTER_OUT = {
            "merged_outputs": merged_ds_strings(ds_list=merged_ds_as_list, column="output")
            # "merged_outputs": merged_ds_strings(ds_list=merged_ds_as_list, column="code")
            }
        else:
            raise Exception("Invalid column option for comparison. Valid options: 'input', 'output', '' (empty option will use 'input' and 'output')")
    else:
        FILTER_OUT = {
            "merged_instances": merged_ds_strings(ds_list=merged_ds_as_list, c_input="input", c_output="output")
            # "merged_inputs": merged_ds_strings(ds_list=merged_ds_as_list, column="input"),
            # "merged_outputs": merged_ds_strings(ds_list=merged_ds_as_list, column="output")
            # "merged_inputs": merged_ds_strings(ds_list=merged_ds_as_list, column="text"),
            # "merged_outputs": merged_ds_strings(ds_list=merged_ds_as_list, column="code")
        }   

    # TEST
    total_rows = len(ds['train']) 
    print(f"TOTAL ROWS IN MERGED DATASET: {total_rows}") 

    filterer = SubstringFilterer(
        output_dir=args.output_dir,
        filter_out=FILTER_OUT,
        ds_name=args.ds_output_name,
        column = args.column_to_compare
    )

    # Run filtering
    filterer.run(dataset=ds, num_proc=args.num_proc, batch_size=args.batch_size)

if __name__ == "__main__":
    start_time = time.time()
    main()
    final_time = time.time() - start_time
    print(f'All Computation complete, total run took {final_time:.2f}s')

