
from stage2.get_same_state import get_all_file
import pandas as pd
import json
from oracleEvent.groundtruth_true_false_generation import get_row_index_from_series, get_row_index_from_df


def add_information_for_groundtruth(groundtruth_pair_path, dataset_path):
    df_groundtruth = pd.read_csv(groundtruth_pair_path)
    df_dataset = pd.read_csv(dataset_path)
    for idx in range(len(df_groundtruth)):
        src_app = df_groundtruth.loc[idx]['aid_from']
        tgt_app = df_groundtruth.loc[idx]['aid_to']
        tgt_app_full_name = src_app + "-" + tgt_app
        src_event_index = df_groundtruth.loc[idx]['step_from']
        tgt_event_index = df_groundtruth.loc[idx]['step_to']
        tgt_label = 'correct'
        df_line = df_dataset.query("src_event_index==@src_event_index and target_event_index==@tgt_event_index "
                         "and src_app == @src_app and target_app == @tgt_app_full_name and target_label == @tgt_label")
        tgt_text = df_line.iloc[0].at['target_text']
        tgt_resource = df_line.iloc[0].at['target_id']
        tgt_content = df_line.iloc[0].at['target_content_desc']
        tgt_neighbor = df_line.iloc[0].at['target_atm_neighbor']
        tgt_class = df_line.iloc[0].at['target_class']
        src_class = df_line.iloc[0].at['src_class']
        df_groundtruth.loc[idx,'tgt_resource'] = tgt_resource
        df_groundtruth.loc[idx, 'tgt_text'] = tgt_text
        df_groundtruth.loc[idx,'tgt_content'] = tgt_content
        df_groundtruth.loc[idx,'tgt_neighbor'] = tgt_neighbor
        df_groundtruth.loc[idx,'tgt_class'] = tgt_class
        df_groundtruth.loc[idx,'src_class'] = src_class
    df_groundtruth.to_csv(groundtruth_pair_path)


def get_true_hard_false_pair(predict_result_save_path, groundtruth_file_path, true_false_feature_save_path, threahold):

    predict_csv_List = []
    df_groundtruth = pd.read_csv(groundtruth_file_path)
    get_all_file(predict_result_save_path, predict_csv_List)
    index = 0
    true_feature_num = 0
    false_feature_num = 0
    false_fp_num = 0 # -1 fp num
    true_data_list = []
    false_data_list = []

    time = predict_csv_List[0].split("/")[-1].split("_")[3] 
    for predict_csv_path in predict_csv_List:
        print(index)
        print(predict_csv_path)
        df_predict =pd.read_csv(predict_csv_path)
        groups = df_predict.groupby(['widget1','widget2'])
        for group in groups:
            df_group = group[1]
            widget1 = df_group.iloc[0].at['widget1']
            widget2 = df_group.iloc[0].at['widget2']
            src_app_name = widget1.split("/")[0]
            src_event_id = widget1.split(":")[0].split("/")[1] # b2-3
            src_function = src_event_id.split('-')[0] # b2
            src_event_index = int(src_event_id.split("-")[1])
            tgt_app_name = widget2.split("/")[0]
            tgt_event_index = int(widget2.split(":")[0].split("-")[1])
            # tgt_event_index = df_groundtruth.query("aid_from==@src_app_name and aid_to == @tgt_app_name and step_from==@src_event_index and function==@src_function_full").iloc[0].at['step_to']
            true_predict_score = None
            df_groundtruth_pair = None
            if widget1 == 'Shop1/b1-1:' and widget2 == 'Shop3/b1-0:':
                print("here")
            if src_function == 'b2':
                src_app_name = src_app_name + 'b2'
                tgt_app_name = tgt_app_name + 'b2'


            df_line = df_groundtruth.query("aid_from==@src_app_name and aid_to == @tgt_app_name and step_from==@src_event_index and function==@src_function and step_to==@tgt_event_index")
            if len(df_line) == 1:

                df_groundtruth_pair = df_group.query("label==1")
                if len(df_groundtruth_pair) > 1:
                    print("here")
                assert len(df_groundtruth_pair) == 1


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
                    "text_feature2":df_predict.loc[row_index,'text_feature2'],
                    "tgt_resource":tgt_resource,
                    "tgt_content":tgt_content,
                    "tgt_text":tgt_text,
                    'widget1_class': df_predict.loc[row_index, 'widget1_class'],
                    'widget2_class': df_predict.loc[row_index, 'widget2_class'],
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
                        text_feature2 = df_predict.loc[row_index, 'text_feature2']
                        if text_feature2!=text_feature2:
                            text_feature2 = ''
                        false_data = {
                            "widget1": df_predict.loc[row_index, 'widget1'],
                            "widget2": df_predict.loc[row_index, 'widget2'],
                            "text_feature1": df_predict.loc[row_index, 'text_feature1'],
                            "text_feature2": text_feature2,
                            "tgt_resource": tgt_resource,
                            "tgt_content": tgt_content,
                            "tgt_text": tgt_text,
                            'widget1_class': df_predict.loc[row_index, 'widget1_class'],
                            'widget2_class': df_predict.loc[row_index, 'widget2_class'],
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
                        text_feature2 = df_predict.loc[row_index, 'text_feature2']
                        if text_feature2!=text_feature2:
                            text_feature2 = ''
                        false_data = {
                            "widget1": df_predict.loc[row_index, 'widget1'],
                            "widget2": df_predict.loc[row_index, 'widget2'],
                            "text_feature1": df_predict.loc[row_index, 'text_feature1'],
                            "text_feature2": text_feature2,
                            "tgt_resource": tgt_resource,
                            "tgt_content": tgt_content,
                            "tgt_text": tgt_text,
                            'widget1_class':df_predict.loc[row_index, 'widget1_class'],
                            'widget2_class':df_predict.loc[row_index, 'widget2_class'],
                            'label': 0
                        }
                        false_data_list.append(json.dumps(false_data) + "\n")


        # df_predict.to_csv(predict_csv_path)

        index += 1
    print("true_feature_num",true_feature_num)
    print("hard_false_feature_num",false_feature_num)
    print("false fp num",false_fp_num)

    true_data_path = true_false_feature_save_path+"true_data_"+time+".jsonl"
    false_data_path = true_false_feature_save_path+"hard_false_data_"+time+".jsonl"
    with open(true_data_path, 'w+') as jsonl:
        jsonl.writelines(true_data_list)
    print("true_data_path", true_data_path)
    with open(false_data_path, 'w+') as jsonl:
        jsonl.writelines(false_data_list)
    print("hard_false_data_path", false_data_path)

