
import torch.nn as nn
import torch
from transformers import PreTrainedTokenizer,TrainingArguments, PreTrainedTokenizerBase

import numpy as np

from datasets import load_dataset, Dataset
from typing import Dict, Union, Optional,List
from dataclasses import dataclass
from typing import Dict, Union, Optional
from transformers.file_utils import PaddingStrategy


from config import DataTrainingArguments



class MLP(nn.Module):
    def _init_(self, input_dim, hid1,output_dim):
        super().__init__()
        self.layer1 = nn.Linear(input_dim,hid1)
        self.layer2 = nn.Linear(hid1,hid1)

    def forward(self, x1, x2):
        x = torch.cat(x1, x2)
        x = torch.relu(self.layer1(x1.float()))
        x = torch.relu(self.layer2(x))
        return torch.nn.functional.softmax(x)



def load_datas(datafiles: Dict[str, str],cache_dir: str = None, max_train: int = None, max_eval: int = None):
    # for text+other feature
    # train json valid json - > dataset
    datasets = load_dataset('json', data_files= datafiles, cache_dir =cache_dir)

    if max_train is not None:
        datasets['train'] = datasets['train'].select(range(max_train))
    if max_eval is not None:
        datasets['validation'] = datasets['validation'].select(range(max_eval))

    return datasets

def load_data(path: str, cache_dir: str = None, max_train_samples: int = None) -> Dataset:
    # one json - > dataset
    dataset = load_dataset("json", data_files=path, cache_dir=cache_dir)
    if max_train_samples is not None:
        return dataset.select(range(max_train_samples))
    return dataset

def encode_fn(mask_index:int,mask_value:int, samples: dict, tokenizer:PreTrainedTokenizer, max_length: int = 512, padding: bool = False):
    # nl feature
    nl_1 = [feature.lower() for feature in samples['text_feature1']]
    nl_2 = [feature.lower() for feature in samples['text_feature2']]
    # other feature
    other_1 = [feature for feature in samples['other_feature1']]
    other_2 = [feature for feature in samples['other_feature2']]
    # category feature
    categeory_1 = [feature for feature in samples['category']]
    category_2 = categeory_1
    # concat other feature and category feature
    category_1 = np.array(categeory_1).reshape(-1,1)
    category_2 = category_1
    other_1 = np.hstack((np.array(other_1),category_1))
    other_2 = np.hstack((np.array(other_2),category_2))
    other_1 = other_1.tolist()
    other_2 = other_2.tolist()


    nls= package_instance(nl_1,nl_2)
    nl_instances = unpack_instance(nls)

    # if nl_1 ='[UNK] and nl_2 ='[UNK], nl_feature_tokens = [1] else [0]
    nl_feature_tokens = []
    for item in nls:
        if item[0]=='[UNK]' and item[1]=='[UNK]' or item[0] == '[unk]' and  item[1]=='[unk]':
            nl_feature_tokens.append(1)
        else:
            nl_feature_tokens.append(0)


    tokenized_examples = tokenizer(
        nl_instances,
        #second_sentences,
        truncation=True,
        max_length=max_length,
        padding='max_length' if padding else False
    )

    tokenize_results = {}  
    # each add one key = [input_ids, type, mask]
    tokenize_results.update({
        k:[v[i:i+2] for i in range(0,len(v),2)] # [1000,2,*]
        for k,v in tokenized_examples.items()
    })



    others = package_instance(other_1,other_2)

    #for test
    assert len(other_1) == len(other_2)
    for x, y in zip(other_1,other_2):
        assert len(x) == len(y)

    assert len(nl_1) == len(nl_2)
    for x, y in zip(nl_1,nl_2):
        assert len([x]) == len([y])


    tokenize_results['other_features'] = mask_feature(others, mask_index, mask_value)
    tokenize_results['nl_feature_tokens'] = nl_feature_tokens #[1000,2] #add nls to identify ['[UNK]', '[UNK]']

    if 'label' in samples.keys():
        tokenize_results['labels'] = [label for label in samples['label']]

    return tokenize_results

def encode_fn_new(mask_index:int,mask_value:int, samples: dict, tokenizer:PreTrainedTokenizer, max_length: int = 512, padding: bool = False):
    # nl feature
    nl_1 = [feature.lower() for feature in samples['text_feature1']]
    nl_2 = [feature.lower() for feature in samples['text_feature2']]
    # other feature
    other_1 = [feature[0] for feature in samples['other_feature1']]
    other_2 = [feature[0] for feature in samples['other_feature2']]


    # text_full_feature
    txt_1 = []
    txt_2 = []
    for idx in range(len(nl_1)):
        txt_1.append(nl_1[idx] + "\n"+ other_1[idx])
        txt_2.append(nl_2[idx] + "\n" + other_2[idx])
        



    nls= package_instance(txt_1,txt_2)
    nl_instances = unpack_instance(nls)

    # if nl_1 ='[UNK] and nl_2 ='[UNK], nl_feature_tokens = [1] else [0]
    nl_feature_tokens = []
    for item in nls:
        if item[0]=='[UNK]' and item[1]=='[UNK]' or item[0] == '[unk]' and  item[1]=='[unk]':
            nl_feature_tokens.append(1)
        else:
            nl_feature_tokens.append(0)


    tokenized_examples = tokenizer(
        nl_instances,
        #second_sentences,
        truncation=True,
        max_length=max_length,
        padding='max_length' if padding else False
    )

    tokenize_results = {}  
    # each add one key = [input_ids, type, mask]
    tokenize_results.update({
        k:[v[i:i+2] for i in range(0,len(v),2)] # [1000,2,*]
        for k,v in tokenized_examples.items()
    })



    tokenize_results['nl_feature_tokens'] = nl_feature_tokens #[1000,2] #add nls to identify ['[UNK]', '[UNK]']

    if 'label' in samples.keys():
        tokenize_results['labels'] = [label for label in samples['label']]

    return tokenize_results


def encode_fn_only_text(mask_index: int, mask_value: int, samples: dict, tokenizer: PreTrainedTokenizer, max_length: int = 512,
              padding: bool = False):
    # nl feature
    nl_1 = [feature.lower() for feature in samples['text_feature1']]
    nl_2 = [feature.lower() for feature in samples['text_feature2']]



    nls = package_instance(nl_1, nl_2)
    nl_instances = unpack_instance(nls)

    # if nl_1 ='[UNK] and nl_2 ='[UNK], nl_feature_tokens = [1] else [0]
    nl_feature_tokens = []
    for item in nls:
        if item[0] == '[UNK]' and item[1] == '[UNK]' or item[0] == '[unk]' and item[1] == '[unk]':
            nl_feature_tokens.append(1)
        else:
            nl_feature_tokens.append(0)

    tokenized_examples = tokenizer(
        nl_instances,
        # second_sentences,
        truncation=True,
        max_length=max_length,
        padding='max_length' if padding else False
    )

    tokenize_results = {}
    # each add one key = [input_ids, type, mask]
    tokenize_results.update({
        k: [v[i:i + 2] for i in range(0, len(v), 2)]  # [1000,2,*]
        for k, v in tokenized_examples.items()
    })



    assert len(nl_1) == len(nl_2)
    for x, y in zip(nl_1, nl_2):
        assert len([x]) == len([y])

    # tokenize_results['other_features'] = others #[1000,2,*]
    # for ablation study
    tokenize_results['nl_feature_tokens'] = nl_feature_tokens  # [1000,2] #add nls to identify ['[UNK]', '[UNK]']

    if 'label' in samples.keys():
        tokenize_results['labels'] = [label for label in samples['label']]

    return tokenize_results

def mask_feature(features: List[List], idx, mask_value):
    # for ablation study
    if idx == -1:
        return features
    else:
        for item in features:
            item[0][idx] = mask_value
            item[1][idx] = mask_value

        return features

@dataclass
class DataCollatorForMultipleChoice:

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features):
        labels = self.extract(features, "labels", pop=True)
        self.extract(features, 'label', pop=True)
        nl_feature_tokens = self.extract(features,'nl_feature_tokens',pop=True)
        features_new = {} # [400,2,*]
        for key in features[0].keys():
            # text features -> list to dict
            features_new[key] = sum(self.extract(features, key), [])

        batch = self.tokenizer.pad(
            #features,
            features_new,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )


        batch_result = {k: v for k, v in batch.items()}
        batch_result["labels"] = torch.tensor(labels, dtype=torch.int64) if labels is not None else None
        # batch_result['nl_feature_tokens']=torch.tensor(nl_feature_tokens, dtype=torch.int64) if nl_feature_tokens is not None else None
        batch_result['nl_feature_tokens'] = nl_feature_tokens
        return batch_result
    
    @classmethod
    def extract(cls, features: List[Dict[str, object]], key: str, pop: bool = False):
        assert key in features[0].keys()
        result = []
        for feature in features:
            result.append(feature[key])
            if pop:
                feature.pop(key)
        return result


def data_collator(targ: TrainingArguments, tokenizer: PreTrainedTokenizer = None):
    pad_to_multiple_of = 8 if targ.fp16 else None
    return DataCollatorForMultipleChoice(tokenizer, pad_to_multiple_of=pad_to_multiple_of)


def define_encoding_process(
        key: str, dataset: Dataset, tokenizer: PreTrainedTokenizer,
        targ: TrainingArguments, darg: DataTrainingArguments, mask_index, mask_value
):
    with targ.main_process_first(desc=f"{key} dataset map pre-processing"):
        dataset = dataset.map(
            lambda x: encode_fn_only_text(mask_index, mask_value, x, tokenizer, darg.max_seq_length, darg.pad_to_max_length),
            batched=True,
            num_proc=darg.preprocessing_num_workers,
            load_from_cache_file=not darg.overwrite_cache,
        )
    return dataset

def package_instance(nls_1,nls_2):
    nls = []
    for index in range(len(nls_1)):
        nl  = [nls_1[index],nls_2[index]]
        nls.append(nl)
    return nls

def unpack_instance(nls):
    nl_instances = []
    # nls_1 = []
    # nls_2 = []
    for index in range(len(nls)):
        nl_instances.append(nls[index][0])
        nl_instances.append(nls[index][1])
        # nls_1.append(nls[index][0])
        # nls_2.append(nls[index][1])
    # return nl_instances,nls_1,nls_2
    return nl_instances
