# coding=utf-8
import logging
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "2"
import sys

import datasets
from transformers import (
    HfArgumentParser,
    TrainingArguments,
    Trainer
)
from transformers.trainer_utils import get_last_checkpoint, set_seed
from transformers.utils import logging as tl

import preprocess_trainBert_cosine as preprocess
import run as run
from config import (
    ModelArguments,
    DataTrainingArguments
)
from evaluation import compute_metrics,compute_metrics_cosine
from model_cosine import init_model
import time 
import torch
import numpy as np
import random



logger = logging.getLogger(__name__)

# get current time
def get_time():
    current_time = time.strftime('%Y%m%d%H%M',time.localtime())

    return current_time

seed = 42


torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# # once for a machine
# cache_dir = '/Users/Migration/model/ori_bert_model/'

# def download_plm(cache_dir):
#     os.makedirs(cache_dir, exist_ok=True)
#     tokenizer = BertTokenizer.from_pretrained(
#         "bert-base-uncased",
#         cache_dir=cache_dir,
#         use_fast=True,
#     )
#     model = BertModel.from_pretrained(
#         "bert-base-uncased",
#         cache_dir=cache_dir,
#     )
#     model.resize_token_embeddings(len(tokenizer))
#     torch.save(model, os.path.join(cache_dir, 'model.pth'))
#     torch.save(tokenizer, os.path.join(cache_dir, 'tokenizer.pth'))
#     return model, tokenizer






if __name__ == '__main__':
    start_time = time.time()
    print("start time",start_time)
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        m, d, a = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1]))
    else:
        m, d, a = parser.parse_args_into_dataclasses()

    model_args: ModelArguments = m
    data_args: DataTrainingArguments = d
    training_args: TrainingArguments = a

    current_time = get_time()
    training_args.output_dir = os.path.join(training_args.output_dir, f'{current_time}')
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    log_level = training_args.get_process_log_level()
    datasets.utils.logging.set_verbosity(log_level)
    tl.set_verbosity(log_level)
    tl.enable_default_handler()
    tl.enable_explicit_format()

    model, tokenizer = init_model(logger, model_args)
    set_seed(training_args.seed)

    data_files = {}
    if data_args.train_file is not None:
        data_files['train'] = data_args.train_file
    if data_args.validation_file is not None:
        data_files["validation"] = data_args.validation_file
    if data_args.test_file is not None:
        data_files['test'] = data_args.test_file

    raw_datasets = preprocess.load_datas(
        data_files, model_args.cache_dir,
        data_args.max_train_samples,
        data_args.max_eval_samples
    )

    train_dataset, eval_dataset, testset = None, None, None
    if training_args.do_train:
        train_dataset = preprocess.define_encoding_process(
            'train', raw_datasets['train'],
            tokenizer, training_args, data_args, model_args.mask_index,model_args.mask_value
        )

    if training_args.do_eval:
        eval_dataset = preprocess.define_encoding_process(
            'validation', raw_datasets['validation'],
            tokenizer, training_args, data_args,model_args.mask_index,model_args.mask_value)

    if training_args.do_predict:
        testset = preprocess.define_encoding_process(
            'test', raw_datasets['test'],
            tokenizer, training_args, data_args,model_args.mask_index,model_args.mask_value)

    data_collator = preprocess.data_collator(
        training_args, tokenizer=tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics = compute_metrics_cosine,
    )

    if training_args.do_train:
        run.train(trainer, training_args, last_checkpoint)
        end_time = time.time()
        print("end time", time.time())
        print("last time", end_time-start_time)
        run.log(start_time,end_time,training_args,data_args)
        # run.log(data_args.train_file,data_args.valid_file,data_args.test_file)

    if training_args.do_eval:
        run.evaluate(trainer, training_args,model_args.runtime_environment)
        end_time = time.time()
        print("end time", time.time())
        print("last time", end_time-start_time)


    if training_args.do_predict:
        # run.predict(trainer, training_args, testset, training_args.output_dir)
        run.predict_cosine(trainer, training_args, testset, training_args.output_dir,data_args.test_file,model_args.runtime_environment, data_args.test_file)
        end_time = time.time()
        print("end time", time.time())
        print("last time", end_time-start_time)

