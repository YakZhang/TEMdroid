from stage2.get_same_state import get_all_file
from BERT.get_widget_feature_trainBert import get_widget_from_json_xml
import os
import pandas as pd
import time
import json
import jsonlines


def get_widget_pair_json_xml(src_path, tgt_path, feature_save_path_prefix):
    # get src_file_path
    file_List = []
    get_all_file(src_path, file_List)
    src_file_List = []
    for file_path in file_List:
        if '_add_feature.csv' in file_path:
            src_file_List.append(file_path)

    # get tgt_file_path
    file_List = []
    get_all_file(tgt_path, file_List)
    tgt_file_List = []
    for file_path in file_List:
        if 'states' in file_path and '.xml' in file_path:  # states/*.json
            tgt_file_List.append(file_path)

    end_time = time.strftime('%Y%m%d%H%M', time.localtime())

    # generate all pair for each src and each tgt
    for src_file_path in src_file_List:
        src_app_name = src_file_path.split("/")[-1].split("_")[0]
        for tgt_file_path in tgt_file_List:
            if src_app_name in tgt_file_path:
                continue
            tgt_prefix = tgt_file_path.replace(tgt_file_path.split("/")[-1],"")
            tgt_app_name = tgt_prefix.split("/")[-3]
            if '351' in tgt_file_path:
                print("here")
            fold_name = choose_fold(tgt_app_name)
            feature_save_full_path_prefix = feature_save_path_prefix + fold_name + '/'
            feature_save_full_path = feature_save_full_path_prefix + src_app_name + "_" + tgt_app_name + "_test_data_" + end_time + ".jsonl"
            tgt_screen_file = ''
            id = -1
            src_prefix = src_file_path.replace(src_file_path.split("/")[-1],"")
            src_name = src_file_path.split("/")[-1]
            src_event_id = -1
            tgt_file_symbol = 'xml'
            feature_save_path = None
            tgt_event_id = None
            get_widget_from_json_xml(tgt_file_path, tgt_screen_file,id,src_prefix, src_name,feature_save_path,src_event_id,tgt_file_symbol,feature_save_full_path,tgt_event_id)

def get_widget_pair_xml(groundtruth_file_path, widget_path_prefix, feature_save_path_prefix):
    
    df_groundtruth = pd.read_csv(groundtruth_file_path)
    groups = df_groundtruth.groupby(['aid_from','aid_to'])
    index = 0
    for group in groups:
        src_app = group[0][0]
        tgt_app = group[0][1]
        if 'a33'in src_app or 'a43' in src_app or 'a33' in tgt_app or 'a43' in tgt_app:
            continue
        tgt_prefix = tgt_app[0:2] + '/'
        print("src_app",src_app)
        print("tgt_app",tgt_app)
        print("index",index)
        end_time = time.strftime('%Y%m%d%H%M',time.localtime())
        fold_name = choose_fold(tgt_app)
        feature_save_full_path_prefix = feature_save_path_prefix + fold_name + '/'

        feature_save_full_path = feature_save_full_path_prefix + src_app+"_"+tgt_app+"_test_data_"+end_time+ ".jsonl"
        df_group = group[1]
        for idx in range(len(df_group)):
            function = df_group.iloc[idx].at['function']
            src_event_index = df_group.iloc[idx].at['step_from']
            tgt_event_index = None
            # for -1 pair
            if df_group.iloc[idx].at['step_to'] == -1:
                tgt_event_index = get_false_corresponding_tgt_event(df_group,idx)
            else:
                tgt_event_index = df_group.iloc[idx].at['step_to']
            # src_path = get_app_path(src_app,widget_path_prefix)
            # tgt_path = get_app_path(tgt_app,widget_path_prefix)
            src_path = widget_path_prefix + src_app[0:2]+"/"+ src_app+"_revise_add_feature.csv"
            tgt_path = widget_path_prefix + tgt_app[0:2]+"/"+ tgt_app+"_revise_add_feature.csv"
            src_event_id = function[0]+function[2]+"-"+str(src_event_index) # e.g., b1-0
            tgt_event_id = function[0]+function[2]+"-"+str(tgt_event_index)
            df_tgt = pd.read_csv(tgt_path)
            df_src = pd.read_csv(src_path)
            tgt_screen_file = None
            df_tgt_line = get_event_id(df_tgt,tgt_event_index,function)
            df_src_line =get_event_id(df_src,src_event_index,function)
            if len(df_src_line) == 0 or len(df_tgt_line)==0:

                continue
            # for SYS_EVENT:
            tgt_type = df_tgt_line.iloc[0].at['type']
            if tgt_type == 'SYS_EVENT':
                continue
            tgt_screen_file = df_tgt_line.iloc[0].at['state_name']
            id = -1
            src_name = src_path.split('/')[-1]
            src_prefix = src_path.replace(src_path.split("/")[-1],"")
            tgt_file_symbol = 'xml'
            feature_save_path = None
            get_widget_from_json_xml(tgt_prefix, tgt_screen_file,id,src_prefix, src_name,feature_save_path,src_event_id,tgt_file_symbol,feature_save_full_path,tgt_event_id)
        index += 1

def get_event_id(df,event_index,function):
    if function[2] == '1':
        df_line = df.query('b1==@event_index')
        return df_line
    else:
        df_line = df.query('b2==@event_index')
        return df_line


def choose_fold(tgt_app_name):
    if tgt_app_name[2] == '1':
        return 'fold_1'
    elif tgt_app_name[2] == '2':
        return 'fold_2'
    elif tgt_app_name[2] == '3':
        return 'fold_3'
    elif tgt_app_name[2] == '4':
        return 'fold_4'
    elif tgt_app_name[2]=='5':
        return 'fold_5'








def get_true_false_pair(predict_result_save_path, groundtruth_file_path, widget_path_prefix, true_false_feature_save_path, threahold):
    predict_csv_List = []
    df_groundtruth = pd.read_csv(groundtruth_file_path)
    get_all_file(predict_result_save_path, predict_csv_List)
    index = 0
    true_feature_num = 0
    false_feature_num = 0
    false_fp_num = 0 # -1 fp num
    true_data_list = []
    false_data_list = []

    time = predict_csv_List[0].split("/")[-1].split("_")[-2] 
    for predict_csv_path in predict_csv_List:
        print(index)
        print(predict_csv_path)
        if 'a11_a13' in predict_csv_path:
            print("here")
        df_predict =pd.read_csv(predict_csv_path)
        groups = df_predict.groupby(['widget1'])
        for group in groups:
            df_group = group[1]
            widget1 = df_group.iloc[0].at['widget1']
            widget2 = df_group.iloc[0].at['widget2']
            src_app_name = widget1.split("/")[0]
            src_event_id = widget1.split(":")[0].split("/")[1] # b2-3
            src_function = src_event_id.split('-')[0] # b2
            src_event_index = int(src_event_id.split("-")[1])
            tgt_app_name = widget2.split("/")[0]
            src_function_full = src_function[0]+src_app_name[1]+src_function[1] # b12
            tgt_event_index = int(widget2.split(":")[1].split("-")[1].split("/")[0])
            # tgt_event_index = df_groundtruth.query("aid_from==@src_app_name and aid_to == @tgt_app_name and step_from==@src_event_index and function==@src_function_full").iloc[0].at['step_to']
            tgt_path = widget_path_prefix + tgt_app_name[0:2]+"/"+ tgt_app_name+"_revise_add_feature.csv"
            df_tgt = pd.read_csv(tgt_path)
            df_tgt_line = get_event_id(df_tgt,tgt_event_index,src_function_full)
            gt_tgt_resource = df_tgt_line.iloc[0].at['add_id']
            gt_tgt_content = df_tgt_line.iloc[0].at['content_desc']
            gt_tgt_text = df_tgt_line.iloc[0].at['text']
            true_predict_score = None
            df_groundtruth_pair = None

            df_line = df_groundtruth.query("aid_from==@src_app_name and aid_to == @tgt_app_name and step_from==@src_event_index and function==@src_function_full and step_to==@tgt_event_index")
            if len(df_line) == 1:
                if gt_tgt_resource==gt_tgt_resource:
                    gt_tgt_resource= gt_tgt_resource.split("id/")[1]
                    df_groundtruth_pair = df_group.query("tgt_resource==@gt_tgt_resource")
                    if len(df_groundtruth_pair) > 1:
                        if gt_tgt_text == gt_tgt_text:
                            df_groundtruth_pair = df_group.query("tgt_resource==@gt_tgt_resource and tgt_text == @gt_tgt_text")
                            assert len(df_groundtruth_pair) == 1
                        elif gt_tgt_content == gt_tgt_content:
                            df_groundtruth_pair = df_group.query(
                                "tgt_resource==@gt_tgt_resource and tgt_content == @gt_tgt_content")
                            assert len(df_groundtruth_pair) == 1
                elif gt_tgt_content == gt_tgt_content:
                    df_groundtruth_pair = df_group.query("tgt_content==@gt_tgt_content")
                    if len(df_groundtruth_pair) > 1:
                        print("more than one sample using same tgt_content")

                # get true sample and false sample
                row_index = get_row_index_from_df(df_groundtruth_pair)[0]
                df_predict.loc[row_index,'correct_label'] = 1
                true_feature_num += 1
                ori_tgt_resource = df_predict.loc[row_index,'tgt_resource']
                tgt_resource = None
                tgt_content = None
                tgt_text = None
                if ori_tgt_resource == ori_tgt_resource:
                    tgt_resource = ori_tgt_resource
                else:
                    tgt_resource = ""
                ori_tgt_content = df_predict.loc[row_index,'tgt_content']
                if ori_tgt_content == ori_tgt_content:
                    tgt_content = ori_tgt_content
                else:
                    tgt_content = ""
                ori_tgt_text = df_predict.loc[row_index,'tgt_text']
                if ori_tgt_text == ori_tgt_text:
                    tgt_text = ori_tgt_text
                else:
                    tgt_text = ""
                true_data = {
                    "widget1":df_predict.loc[row_index,'widget1'],
                    "widget2":df_predict.loc[row_index,'widget2'],
                    "text_feature1":df_predict.loc[row_index,'text_feature1'],
                    "other_feature1":df_predict.loc[row_index,'other_feature1'],
                    "text_feature2":df_predict.loc[row_index,'text_feature2'],
                    "other_feature2":df_predict.loc[row_index,'other_feature2'],
                    "tgt_resource":tgt_resource,
                    "tgt_content":tgt_content,
                    "tgt_text":tgt_text,
                    'label':1
                }
                true_data_list.append(json.dumps(true_data)+"\n")
                true_predict_score = df_groundtruth_pair.iloc[0].at['predict_score']
                for idx in range(len(df_group)):
                    predict_score = df_group.iloc[idx].at['predict_score']
                    if predict_score > true_predict_score:
                        row_index = get_row_index_from_series(df_group.iloc[idx])
                        df_predict.loc[row_index, 'correct_label'] = 0
                        tgt_resource = None
                        tgt_content = None
                        tgt_text = None
                        ori_tgt_resource = df_predict.loc[row_index, 'tgt_resource']
                        if ori_tgt_resource == ori_tgt_resource:
                            tgt_resource = ori_tgt_resource
                        else:
                            tgt_resource = ""
                        ori_tgt_content = df_predict.loc[row_index, 'tgt_content']
                        if ori_tgt_content == ori_tgt_content:
                            tgt_content = ori_tgt_content
                        else:
                            tgt_content = ""
                        ori_tgt_text = df_predict.loc[row_index, 'tgt_text']
                        if ori_tgt_text == ori_tgt_text:
                            tgt_text = ori_tgt_text
                        else:
                            tgt_text = ""
                        false_feature_num += 1
                        false_data = {
                            "widget1": df_predict.loc[row_index, 'widget1'],
                            "widget2": df_predict.loc[row_index, 'widget2'],
                            "text_feature1": df_predict.loc[row_index, 'text_feature1'],
                            "other_feature1": df_predict.loc[row_index, 'other_feature1'],
                            "text_feature2": df_predict.loc[row_index, 'text_feature2'],
                            "other_feature2": df_predict.loc[row_index, 'other_feature2'],
                            "tgt_resource": tgt_resource,
                            "tgt_content": tgt_content,
                            "tgt_text": tgt_text,
                            'label': 0
                        }
                        false_data_list.append(json.dumps(false_data) + "\n")

            else:
                for idx in range(len(df_group)):
                    predict_score = df_group.iloc[idx].at['predict_score']
                    if predict_score > threahold:
                        row_index = get_row_index_from_series(df_group.iloc[idx])
                        false_fp_num += 1
                        false_feature_num += 1
                        tgt_resource = None
                        tgt_content = None
                        tgt_text = None
                        ori_tgt_resource = df_predict.loc[row_index, 'tgt_resource']
                        if ori_tgt_resource == ori_tgt_resource:
                            tgt_resource = ori_tgt_resource
                        else:
                            tgt_resource = ""
                        ori_tgt_content = df_predict.loc[row_index, 'tgt_content']
                        if ori_tgt_content == ori_tgt_content:
                            tgt_content = ori_tgt_content
                        else:
                            tgt_content = ""
                        ori_tgt_text = df_predict.loc[row_index, 'tgt_text']
                        if ori_tgt_text == ori_tgt_text:
                            tgt_text = ori_tgt_text
                        else:
                            tgt_text = ""
                        false_data = {
                            "widget1": df_predict.loc[row_index, 'widget1'],
                            "widget2": df_predict.loc[row_index, 'widget2'],
                            "text_feature1": df_predict.loc[row_index, 'text_feature1'],
                            "other_feature1": df_predict.loc[row_index, 'other_feature1'],
                            "text_feature2": df_predict.loc[row_index, 'text_feature2'],
                            "other_feature2": df_predict.loc[row_index, 'other_feature2'],
                            "tgt_resource": tgt_resource,
                            "tgt_content": tgt_content,
                            "tgt_text": tgt_text,
                            'label': 0
                        }
                        false_data_list.append(json.dumps(false_data) + "\n")


        df_predict.to_csv(predict_csv_path)

        index += 1
    print("true_feature_num",true_feature_num)
    print("false_feature_num",false_feature_num)
    print("false fp num",false_fp_num)

    true_data_path = true_false_feature_save_path+"true_data_"+time+".jsonl"
    false_data_path = true_false_feature_save_path+"false_data_"+time+".jsonl"
    with open(true_data_path, 'w+') as jsonl:
        jsonl.writelines(true_data_list)
    print("true_data_path", true_data_path)
    with open(false_data_path, 'w+') as jsonl:
        jsonl.writelines(false_data_list)
    print("false_data_path", false_data_path)



def get_row_index_from_df(df_groundtruth_pair):
    index = df_groundtruth_pair.index.tolist()
    return index

def get_row_index_from_series(series):
    index = series.name
    return index


def revise_ori_widget_pair(ori_widget_pair_path, true_feature_label_path,feature_save_path_prefix):
    # given ori_widget_pair_path and true_feature_label_path
    # revised the label of the ori_widget_pairs according to true_feature_label

    # get ori_widget_file_path
    file_List = []
    get_all_file(ori_widget_pair_path, file_List)
    ori_widget_file_List = []
    time = true_feature_label_path.split("/")[-1].split("_")[-1]
    for file_path in file_List:
        if '_test_data_' in file_path:
            ori_widget_file_List.append(file_path)

    # get true widget 1 and widget2
    true_key = set()
    with open(true_feature_label_path, 'r+', encoding='utf8') as f:
        for item in jsonlines.Reader(f):
            widget1 = item['widget1']
            widget2 = item['widget2']
            true_key.add(widget1+"-"+widget2)
    print("true_key_num",len(true_key))

    true_sample_num = 0
    # generate full true_sample.jsonl false_sample.jsonl
    true_sample_list = []
    false_sample_list = []


    for widget_file_path in ori_widget_file_List:
        new_widget_pairs = []
        with open(widget_file_path, 'r+', encoding='utf8') as f:
            for item in jsonlines.Reader(f):
                item['label'] = 0  # default = 0
                candidate_widget1 = item['widget1']
                candidate_widget2 = item['widget2']
                candidate_key = candidate_widget1 + "-" + candidate_widget2
                if candidate_key in true_key:
                    item['label'] = 1
                    true_sample_num += 1
                    true_sample_list.append(json.dumps(item)+"\n")
                else:
                    false_sample_list.append(json.dumps(item)+"\n")
                new_widget_pairs.append(json.dumps(item)+"\n")

        # save the revised widget_feature_label in the feature_save_path
        feature_save_path = widget_file_path
        with open(feature_save_path, 'w') as jsonl:
            jsonl.writelines(new_widget_pairs)

    # save the full true widget_feature_label
    true_feature_save_path = feature_save_path_prefix + 'true_full_data_'+time
    with open(true_feature_save_path, 'w') as jsonl:
        jsonl.writelines(true_sample_list)
    print("true_feature_save_path", true_feature_save_path)

    # save the full false feature label
    false_feature_save_path = feature_save_path_prefix + 'false_full_data_'+time
    with open(false_feature_save_path, 'w') as jsonl:
        jsonl.writelines(false_sample_list)
    print("total false_feature_save_path", false_feature_save_path)



    print("true_sample_num",true_sample_num)


def get_false_widget_pair_xml(groundtruth_file_path, widget_path_prefix, feature_save_path_prefix):
    # generate all pair for each src and corresponding tgt
    df_groundtruth = pd.read_csv(groundtruth_file_path)
    groups = df_groundtruth.groupby(['aid_from', 'aid_to','function'])
    index = 0
    for group in groups:
        src_app = group[0][0]
        tgt_app = group[0][1]
        if 'a33' in src_app or 'a43' in src_app or 'a33' in tgt_app or 'a43' in tgt_app:
            continue
        tgt_prefix = tgt_app[0:2] + '/'
        print("src_app", src_app)
        print("tgt_app", tgt_app)
        print("index", index)
        end_time = time.strftime('%Y%m%d%H%M', time.localtime())
        fold_name = choose_fold(tgt_app)
        feature_save_full_path_prefix = feature_save_path_prefix + fold_name + '/'

        feature_save_full_path = feature_save_full_path_prefix + src_app + "_" + tgt_app + "_test_data_" + end_time + ".jsonl"
        df_group = group[1]
        for idx in range(len(df_group)):
            if df_group.iloc[idx].at['step_to'] != -1:
                continue
            # only tgt_app = -1
            function = df_group.iloc[idx].at['function']
            src_event_index = df_group.iloc[idx].at['step_from']
            tgt_event_index = get_false_corresponding_tgt_event(df_group,idx)
            # src_path = get_app_path(src_app,widget_path_prefix)
            # tgt_path = get_app_path(tgt_app,widget_path_prefix)
            src_path = widget_path_prefix + src_app[0:2] + "/" + src_app + "_revise_add_feature.csv"
            tgt_path = widget_path_prefix + tgt_app[0:2] + "/" + tgt_app + "_revise_add_feature.csv"
            src_event_id = function[0] + function[2] + "-" + str(src_event_index)  # e.g., b1-0
            tgt_event_id = function[0] + function[2] + "-" + str(tgt_event_index)
            df_tgt = pd.read_csv(tgt_path)
            df_src = pd.read_csv(src_path)
            tgt_screen_file = None
            df_tgt_line = get_event_id(df_tgt, tgt_event_index, function)
            df_src_line = get_event_id(df_src, src_event_index, function)
            if len(df_src_line) == 0 or len(df_tgt_line) == 0:
                continue
            # for SYS_EVENT:
            tgt_type = df_tgt_line.iloc[0].at['type']
            if tgt_type == 'SYS_EVENT':
                continue
            tgt_screen_file = df_tgt_line.iloc[0].at['state_name']
            id = -1
            src_name = src_path.split('/')[-1]
            src_prefix = src_path.replace(src_path.split("/")[-1], "")
            tgt_file_symbol = 'xml'
            feature_save_path = None
            get_widget_from_json_xml(tgt_prefix, tgt_screen_file, id, src_prefix, src_name, feature_save_path,
                                     src_event_id, tgt_file_symbol, feature_save_full_path,tgt_event_id)
        index += 1

def get_not_consistent_widget_pair_xml(groundtruth_file_path, widget_path_prefix, ori_tgt_prefix,feature_save_path_prefix):
    df_groundtruth = pd.read_csv(groundtruth_file_path)
    index = 0
    for idx in range(len(df_groundtruth)):
        detect = df_groundtruth.loc[idx,'detect']
        if detect != detect:
            continue
        detect_list = detect.split(" ")
        end_time = time.strftime('%Y%m%d%H%M', time.localtime())
        for detect_item in detect_list:
            function = df_groundtruth.loc[idx,'function']
            src_app = df_groundtruth.loc[idx,'aid_from']
            tgt_app = df_groundtruth.loc[idx,'aid_to']
            src_event_index = df_groundtruth.loc[idx,'step_from']
            tgt_event_index = int(detect_item.split("-")[1]) #0-0
            groundtruth_tgt_event_index = df_groundtruth.loc[idx,'step_to']
            src_path = widget_path_prefix + src_app[0:2] + "/" + src_app + "_revise_add_feature.csv"
            tgt_path = widget_path_prefix + tgt_app[0:2] + "/" + tgt_app + "_revise_add_feature.csv"
            src_event_id = function[0] + function[2] + "-" + str(src_event_index)  # e.g., b1-0
            tgt_event_id = function[0] + function[2] + "-" + str(tgt_event_index)
            df_tgt = pd.read_csv(tgt_path)
            df_src = pd.read_csv(src_path)
            tgt_screen_file = None
            df_tgt_line = get_event_id(df_tgt, tgt_event_index, function)
            df_src_line = get_event_id(df_src, src_event_index, function)
            df_groundtruth_tgt_line = get_event_id(df_tgt, groundtruth_tgt_event_index, function)
            if len(df_src_line) == 0 or len(df_tgt_line) == 0 or len(df_groundtruth_tgt_line)==0:
                continue
            if df_tgt_line.iloc[0].at['state_name'] == df_groundtruth_tgt_line.iloc[0].at['state_name']:
                continue
            # for SYS_EVENT:
            tgt_type = df_tgt_line.iloc[0].at['type']
            if tgt_type == 'SYS_EVENT':
                continue
            tgt_screen_file = df_tgt_line.iloc[0].at['state_name']
            id = -1
            src_name = src_path.split('/')[-1]
            src_prefix = src_path.replace(src_path.split("/")[-1], "")
            tgt_file_symbol = 'xml'
            feature_save_path = None
            fold_name = choose_fold(tgt_app)

            tgt_prefix = ori_tgt_prefix + tgt_app[0:2] + '/'
            feature_save_full_path_prefix = feature_save_path_prefix + fold_name + '/'
            feature_save_full_path = feature_save_full_path_prefix + src_app + "_" + tgt_app + "_test_data_" + end_time + ".jsonl"
            get_widget_from_json_xml(tgt_prefix, tgt_screen_file, id, src_prefix, src_name, feature_save_path,
                                     src_event_id, tgt_file_symbol, feature_save_full_path, tgt_event_id)
            index += 1
            if index == 73:
                print("here")
            print(index)

def get_not_consistent_widget_pairs(groundtruth_file_path, widget_path_prefix, ori_tgt_prefix,feature_save_path_prefix):
    df_groundtruth = pd.read_csv(groundtruth_file_path,encoding='latin-1')

    groups = df_groundtruth.groupby(['aid_from', 'aid_to','function'])
    for group in groups:
        df_group = group[1]
        src_app = group[0][0]
        tgt_app = group[0][1]
        if src_app == 'a21' and tgt_app == 'a23':
            print("here")
        end_time = time.strftime('%Y%m%d%H%M', time.localtime())
        for index in range(len(df_group)):
            detect_item_ori = df_group.iloc[index].at['detect']
            if detect_item_ori == detect_item_ori:
                function = df_group.iloc[index].at['function']
                start_src = df_group.iloc[index].at['step_from']
                end_src = df_group.iloc[len(df_group)-1].at['step_from']
                detect_items = detect_item_ori.split(" ")
                for detect_item in detect_items:
                    tgt_event_index = int(detect_item.split("-")[1])  # 0-0
                    groundtruth_tgt_event_index = df_group.iloc[index].at['step_to']
                    for src_event_index in range(start_src,end_src+1):
                        src_path = widget_path_prefix + src_app[0:2] + "/" + src_app + "_revise_add_feature.csv"
                        tgt_path = widget_path_prefix + tgt_app[0:2] + "/" + tgt_app + "_revise_add_feature.csv"
                        src_event_id = function[0] + function[2] + "-" + str(src_event_index)  # e.g., b1-0
                        tgt_event_id = function[0] + function[2] + "-" + str(tgt_event_index)
                        df_tgt = pd.read_csv(tgt_path)
                        df_src = pd.read_csv(src_path)
                        tgt_screen_file = None
                        df_tgt_line = get_event_id(df_tgt, tgt_event_index, function)
                        df_src_line = get_event_id(df_src, src_event_index, function)
                        df_groundtruth_tgt_line = get_event_id(df_tgt, groundtruth_tgt_event_index, function)
                        if len(df_src_line) == 0 or len(df_tgt_line) == 0 or len(df_groundtruth_tgt_line) == 0:
                            continue
                        if df_tgt_line.iloc[0].at['state_name'] == df_groundtruth_tgt_line.iloc[0].at['state_name']:
                            continue
                        # for SYS_EVENT:
                        tgt_type = df_tgt_line.iloc[0].at['type']
                        src_type = df_src_line.iloc[0].at['type']
                        if tgt_type == 'SYS_EVENT' or src_type == 'SYS_EVENT':
                            continue
                        tgt_screen_file = df_tgt_line.iloc[0].at['state_name']
                        id = -1
                        src_name = src_path.split('/')[-1]
                        src_prefix = src_path.replace(src_path.split("/")[-1], "")
                        tgt_file_symbol = 'xml'
                        feature_save_path = None
                        fold_name = choose_fold(tgt_app)

                        tgt_prefix = ori_tgt_prefix + tgt_app[0:2] + '/'
                        feature_save_full_path_prefix = feature_save_path_prefix + fold_name + '/'
                        feature_save_full_path = feature_save_full_path_prefix + src_app + "_" + tgt_app + "_test_data_" + end_time + ".jsonl"
                        get_widget_from_json_xml(tgt_prefix, tgt_screen_file, id, src_prefix, src_name, feature_save_path,
                                                 src_event_id, tgt_file_symbol, feature_save_full_path, tgt_event_id)



def get_false_corresponding_tgt_event(df_group,group_row_idx):
    for idx in range(len(df_group)):
        if idx <= group_row_idx:
            continue
        tgt_event_index = df_group.iloc[idx].at['step_to']
        if tgt_event_index != -1:
            return tgt_event_index

def combine_json_file(json_path,json_save_path_prefix):
    file_List = []
    item_list = []
    get_all_file(json_path, file_List)
    for json_file_path in file_List:
        with open(json_file_path, 'r+', encoding='utf8') as f:
            for item in jsonlines.Reader(f):
                item_list.append(json.dumps(item)+"\n")
    fold_num = file_List[0].split("/")[-1].split("_")[1][2]
    ori_time = file_List[0].split("/")[-1].split("_")[-1]
    json_save_path = json_save_path_prefix + "test_"+fold_num + "_data_"+ori_time
    with open(json_save_path , 'w') as jsonl:
        jsonl.writelines(item_list)


def delete_dumplicate_widget_pair(feature_save_path):
    file_List = []
    dumplicate_file_list = []
    get_all_file(feature_save_path, file_List)
    for feature_save_path in file_List:
        if 'without_dumplicate' not in feature_save_path:
            dumplicate_file_list.append(feature_save_path)
    for json_file_path in dumplicate_file_list:
        # get without_dumplicate_feature dict
        with open(json_file_path, 'r+', encoding='utf8') as f:
            item_num = 0
            without_dumplicate_feature_dict = dict()  # key is widget1 + tgt signature, value = [original_item, widget2 list]
            for item in jsonlines.Reader(f):
                widget1 = item['widget1']
                widget2 = item['widget2']
                widget2_class = item['widget2_class']
                tgt_resource = item['tgt_resource']
                tgt_text = item['tgt_text']
                tgt_content = item ['tgt_content']
                tgt_droidbot_signature = "[class]%s[resource_id]%s[text]%s[content]%s"%\
                                         (widget2_class,tgt_resource,tgt_text,tgt_content)
                key = widget1 + "-" +tgt_droidbot_signature
                widget2_list = []
                if key in without_dumplicate_feature_dict:
                    widget2_list = without_dumplicate_feature_dict[key][1]
                    widget2_list.append(widget2)
                else:
                    widget2_list.append(widget2)
                value = [item, widget2_list]
                without_dumplicate_feature_dict[key] = value
                item_num += 1
            # generate json_file according without_dumplicate_feature_dict
            item_list = []
            widget2_num =0
            for key in without_dumplicate_feature_dict:
                item = without_dumplicate_feature_dict[key][0]
                widget2_list = without_dumplicate_feature_dict[key][1]
                widget2_num += len(widget2_list)
                item['widget2_list'] = str(widget2_list)
                item_list.append(json.dumps(item)+"\n")
            assert widget2_num == item_num
            # save to the json_file
            feature_save_new_path = json_file_path.replace(".jsonl","_without_dumplicate.jsonl")
            print(feature_save_new_path)
            with open(feature_save_new_path, 'w') as jsonl:
                jsonl.writelines(item_list)



if __name__ == "__main__":
    """
    ** Set the appropriate parameters based on the directory name you created **
    """
    groundtruth_file_path = 'example/Train/groundtruth/groundtruth.csv'
    widget_path_prefix = 'example/Train/tgt_widget/'
    feature_save_path_prefix = 'example/Train/true_false_data/'
    """
    ****************************************************************************
    """
    get_widget_pair_xml(groundtruth_file_path, widget_path_prefix, feature_save_path_prefix)
    get_false_widget_pair_xml(groundtruth_file_path, widget_path_prefix, feature_save_path_prefix)

