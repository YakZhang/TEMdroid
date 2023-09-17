import numpy as np
from transformers.trainer_pt_utils import (
    log_metrics,
    save_metrics,
    save_state
)
from transformers import Trainer, TrainingArguments
from datasets import Dataset
import torch
import jsonlines
import pandas as pd
from logger import Log
import os
import time



def get_mrr_top1(df_predict):
    top1_num = 0
    mrr_score = 0
    gt_num = 0
    groups = df_predict.groupby(['widget1'])
    for group in groups:
        df_group = group[1]
        df_new_group = df_group.sort_values("predict_score", ascending=False)
        # calculate top1
        if df_new_group.iloc[0].at['label'] == 1:
            top1_num += 1
        # calculate mrr
        df_gt_line = df_new_group.query("label==1").iloc[0]
        df_gt_predict_score = df_gt_line.at['predict_score']
        sorted_predict_scores = df_new_group['predict_score'].tolist()
        gt_index = sorted_predict_scores.index(df_gt_predict_score)
        mrr_score += 1 / (gt_index+1)
        gt_num += 1
    print("top1 group", top1_num/gt_num)
    print("mrr score", mrr_score/gt_num)
    return top1_num, mrr_score, gt_num




# Logger
def get_log(test_category,log_save_path):
    log = Log(log_save_path, log_save_path)
    logger = log.Logger
    return logger


def train(trainer: Trainer, train_args: TrainingArguments, last_checkpoint: str = None):
    checkpoint = None
    if train_args.resume_from_checkpoint is not None:
        checkpoint = train_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint

    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model()  # Saves the tokenizer too for easy upload

    metrics: dict[str, float] = train_result.metrics
    train_set: Dataset = trainer.train_dataset
    metrics["train_samples"] = len(train_set)

    log_metrics(trainer, "train", metrics)
    save_metrics(trainer, "train", metrics)
    save_state(trainer)
    print('output_dir',train_args.output_dir)

def log(start_time, end_time,train_args:TrainingArguments,data_args):
    log_save_path = os.path.join(train_args.output_dir,'train_log.log')
    logger = get_log("", log_save_path)
    logger.info("the ablation study of the siamebe newok without share weight")
    logger.info("start_time"+str(start_time)+"second")
    logger.info("end_time"+str(end_time)+"second")
    last_time = end_time - start_time
    logger.info("last_time"+str(last_time)+"second")
    logger.info("train_file:"+data_args.train_file)
    logger.info("valid_file:"+data_args.validation_file)
    logger.info("test_file:"+data_args.test_file)

        



def evaluate(trainer: Trainer, train_args: TrainingArguments,runtime_environment):
    if runtime_environment == 'gpu':
        if not train_args.do_train and train_args.resume_from_checkpoint:
            state_dict = torch.load(os.path.join(
                train_args.resume_from_checkpoint, "pytorch_model.bin"
            ))
            trainer.model.load_state_dict(state_dict)

    else:
        # for cpu only
        if not train_args.do_train and train_args.resume_from_checkpoint:
            state_dict = torch.load(os.path.join(
                train_args.resume_from_checkpoint, "pytorch_model.bin"
            ),map_location="cpu")
            trainer.model.load_state_dict(state_dict)

    metrics = trainer.evaluate()
    eval_set: Dataset = trainer.eval_dataset
    metrics["eval_samples"] = len(eval_set)

    log_metrics(trainer, "eval", metrics)
    save_metrics(trainer, "eval", metrics)
    print('output_dir',train_args.output_dir)


def predict(trainer: Trainer, train_args: TrainingArguments, testset: Dataset,output_dir):
    if not train_args.do_train and train_args.resume_from_checkpoint:
        state_dict = torch.load(os.path.join(
            train_args.resume_from_checkpoint, 'pytorch_model.bin'
        ))
        trainer.model.load_state_dict(state_dict)

    predict_data, labels,_, = trainer.predict(testset) # the outputs of trainer.predict are predictions, label_id, metric(prediction, label_id)
    if len(predict_data) == 2: # add features in predictions[1]
        predictions = predict_data[0]
        features = predict_data[1]
    else:
        predictions = predict_data
    prediction_indexs = np.argmax(predictions, axis=-1).tolist() # index
    ori_predictions = predictions[:,1]


    predict_label_path = os.path.join(output_dir,'predict_label.txt')
    with open(predict_label_path, 'w') as f:
        pred = ' '.join([str(p) for p in prediction_indexs])
        f.write(pred)
    predict_score_path = os.path.join(output_dir,'predict_score.txt')
    with open(predict_score_path, 'w') as f:
        pred = ' '.join([str(p) for p in ori_predictions])
        f.write(pred)
    
    # predict_map csv
    json_path = 'a.b.json'
    ori_file = []
    with open(json_path, 'r+', encoding='utf8') as load_f:
        ori_file_iter = jsonlines.Reader(load_f)
        for item in ori_file_iter:
            ori_file.append(item)
    df = pd.DataFrame(columns=['widget1','widget2','predict_label','predict_score','label','group_index','padding_index'],index=[])
    line_index= 0
    for item in ori_file:
        df_line = pd.DataFrame({
            'widget1':item['widget1'],
            'widget2':item['widget2'],
            'predict_label':prediction_indexs[line_index],
            'predict_score':ori_predictions[line_index],
            'label':item['label'],
            'group_index':item['group'],
            'padding_index':item['padding']
        },index=[1])
        df = df.append(df_line,ignore_index=True)
        line_index = line_index+1
    csv_save_path = os.path.join(output_dir,'predict_map.csv')
    df.to_csv(csv_save_path)


    print('output_dir',train_args.output_dir)
    print("finish write")
    
def predict_cosine(trainer: Trainer, train_args: TrainingArguments, testset: Dataset,output_dir, testpath, runtime_environment,test_file):    
    if runtime_environment == 'gpu':
        if not train_args.do_train and train_args.resume_from_checkpoint:
            state_dict = torch.load(os.path.join(
                train_args.resume_from_checkpoint, 'pytorch_model.bin'
            ))
            trainer.model.load_state_dict(state_dict)
    else:
        # for cpu only
        if not train_args.do_train and train_args.resume_from_checkpoint:
            state_dict = torch.load(os.path.join(
                train_args.resume_from_checkpoint, 'pytorch_model.bin'
            ), map_location = "cpu")
            trainer.model.load_state_dict(state_dict)


    print("start predict")
    print("start predict time",time.time())
    predictions, labels,_, = trainer.predict(testset) # the outputs of trainer.predict are predictions, label_id, metric(prediction, label_id)
            


    predicted = (predictions > 0.6).astype(np.float32) # True->1


    predict_label_path = os.path.join(output_dir,'predict_label.txt')
    with open(predict_label_path, 'w') as f:
        pred = ' '.join([str(p) for p in predicted])
        f.write(pred)
    predict_score_path = os.path.join(output_dir,'predict_score.txt')
    with open(predict_score_path, 'w') as f:
        pred = ' '.join([str(p) for p in predictions])
        f.write(pred)

    # predict_map csv
    print("start write csv")
    print("start write csv time",time.time())

    json_path = testpath
    ori_file = []
    with open(json_path, 'r+', encoding='utf8') as load_f:
        ori_file_iter = jsonlines.Reader(load_f)
        for item in ori_file_iter:
            ori_file.append(item)
            
    widget1_list = []
    widget2_list = []
    text_feature1_list = []
    text_feature2_list = []
    label_list = []
    tgt_x_start_list = []
    tgt_x_end_list = []
    tgt_y_start_list = []
    tgt_y_end_list = []
    tgt_resource_list = []
    tgt_content_list = []
    tgt_text_list = []
    other_feature1_list = []
    other_feature2_list = []
    predict_label_list = predicted
    predict_score_list = predictions
    widget1_class_list = []
    widget2_class_list = []
    if 'tgt_x_start' in ori_file[0]:
        line_index= 0
        for item in ori_file:
            widget1_list.append(item['widget1'])
            widget2_list.append(item['widget2'])
            text_feature1_list.append(item['text_feature1'])
            text_feature2_list.append(item['text_feature2'])
            other_feature1_list.append(str(item['other_feature1']))
            other_feature2_list.append(str(item['other_feature2']))
            label_list.append(item['label'])
            tgt_x_start_list.append(item['tgt_x_start'])
            tgt_x_end_list.append(item['tgt_x_end'])
            tgt_y_start_list.append(item['tgt_y_start'])
            tgt_y_end_list.append(item['tgt_y_end'])
            tgt_resource_list.append(item['tgt_resource'])
            tgt_content_list.append(item['tgt_content'])
            tgt_text_list.append(item['tgt_text'])
            line_index = line_index+1
        df = pd.DataFrame({'widget1':widget1_list,'widget2':widget2_list,'text_feature1':text_feature1_list,'text_feature2':text_feature2_list,'other_feature1':other_feature1_list,
                           'other_feature2':other_feature2_list,'predict_label':predict_label_list,"predict_score":predict_score_list, 'label':label_list,
                           'tgt_x_start':tgt_x_start_list,'tgt_x_end':tgt_x_end_list,'tgt_y_start':tgt_y_start_list,"tgt_y_end":tgt_y_end_list,
                           'tgt_resource':tgt_resource_list,'tgt_content':tgt_content_list, 'tgt_text':tgt_text_list})

    elif 'other_feature1' in ori_file[0]:
        line_index= 0
        for item in ori_file:
            widget1_list.append(item['widget1'])
            widget2_list.append(item['widget2'])
            text_feature1_list.append(item['text_feature1'])
            text_feature2_list.append(item['text_feature2'])
            other_feature1_list.append(str(item['other_feature1']))
            other_feature2_list.append(str(item['other_feature2']))
            label_list.append(item['label'])
            line_index = line_index+1
        df = pd.DataFrame({'widget1':widget1_list,'widget2':widget2_list,'text_feature1':text_feature1_list,'text_feature2':text_feature2_list,'other_feature1':other_feature1_list,
                           'other_feature2':other_feature2_list,'predict_label':predict_label_list,"predict_score":predict_score_list, 'label':label_list,
                           'tgt_resource':tgt_resource_list,'tgt_content':tgt_content_list, 'tgt_text':tgt_text_list})
    else:
        line_index= 0
        for item in ori_file:
            widget1_list.append(item['widget1'])
            widget2_list.append(item['widget2'])
            text_feature1_list.append(item['text_feature1'])
            text_feature2_list.append(item['text_feature2'])
            label_list.append(item['label'])
            tgt_resource_list.append(item['tgt_resource'])
            tgt_content_list.append(item['tgt_content'])
            tgt_text_list.append(item['tgt_text'])
            if 'widget1_class' in item:
                widget1_class_list.append(item['widget1_class'])
            else :
                widget1_class_list.append(item['src_class'])
            if 'widget2_class' in item:
                widget2_class_list.append(item['widget2_class'])
            else:
                widget2_class_list.append(item['tgt_class'])
            line_index = line_index+1
        df = pd.DataFrame({'widget1':widget1_list,'widget2':widget2_list,'text_feature1':text_feature1_list,'text_feature2':text_feature2_list,
                           'predict_label':predict_label_list,"predict_score":predict_score_list, 'label':label_list,
                           'tgt_resource':tgt_resource_list,'tgt_content':tgt_content_list, 'tgt_text':tgt_text_list,'widget1_class':widget1_class_list,'widget2_class':widget2_class_list})

    # print mrr top1
    get_mrr_top1(df)
    csv_save_path = os.path.join(output_dir,'predict_map.csv')
    df.to_csv(csv_save_path)
    test_file_revise_path = test_file.replace(".jsonl","") + "_output.csv"
    df.to_csv(test_file_revise_path)
    print("end write csv time",time.time())
    print('output_dir',train_args.output_dir)
    print('output_dir',test_file_revise_path)
    print("finish write")
    
def predict_class(trainer: Trainer, train_args: TrainingArguments, testset: Dataset,output_dir,testpath):
    # for multiclass
    if not train_args.do_train and train_args.resume_from_checkpoint:
        state_dict = torch.load(os.path.join(
            train_args.resume_from_checkpoint, 'pytorch_model.bin'
        ))
        trainer.model.load_state_dict(state_dict)

    predictions, labels,_, = trainer.predict(testset) # the outputs of trainer.predict are predictions, label_id, metric(prediction, label_id)
    prediction_indexs = np.argmax(predictions, axis=-1).tolist() # index
    ori_predictions = []
    for idx in range(len(predictions)):
        prediction_idx = prediction_indexs[idx]
        ori_prediction = predictions[idx][prediction_idx]
        ori_predictions.append(ori_prediction)




    predict_label_path = os.path.join(output_dir,'predict_label.txt')
    with open(predict_label_path, 'w') as f:
        pred = ' '.join([str(p) for p in prediction_indexs])
        f.write(pred)
    predict_score_path = os.path.join(output_dir,'predict_score.txt')
    with open(predict_score_path, 'w') as f:
        pred = ' '.join([str(p) for p in ori_predictions])
        f.write(pred)
    
    # predict_map csv
    json_path = testpath
    ori_file = []
    with open(json_path, 'r+', encoding='utf8') as load_f:
        ori_file_iter = jsonlines.Reader(load_f)
        for item in ori_file_iter:
            ori_file.append(item)
    df = pd.DataFrame(columns=['widget','text_feature','other_feature','predict_label','predict_score','label'],index=[])
    line_index= 0
    for item in ori_file:
        df_line = pd.DataFrame({
            'widget':item['widget'],
            'text_feature':item['text_feature'],
            'other_feature':str(item['other_feature']),
            'predict_label':prediction_indexs[line_index],
            'predict_score':ori_predictions[line_index],
            'label':item['label'],
        },index=[1])
        df = df.append(df_line,ignore_index=True)
        line_index = line_index+1
    csv_save_path = os.path.join(output_dir,'predict_map.csv')
    df.to_csv(csv_save_path)
    print('output_dir',train_args.output_dir)
    print("finish write")
