# coding=utf-8



import json
import pandas as pd
import numpy as np
from stage2.get_same_state import get_all_file

b1 = {'b11','b21','b31','b41','b51'}
b2 = {'b12','b22','b32','b42','b52'}


def parse_json(json_path,csv_save_path):
    f = open(json_path)
    data = json.load(f)
    index_b1 = 0
    index_b2 = 0
    df = pd.DataFrame(columns = ['id','class','current Activity','type','text','content','clickable',
                                 'password','parent_text','sibling_text','package','ignorable','action','value',
                                 'wait_time','oracle_desc','oracle_id','oracle_xpath','oracle_text','b1','b2'],index=[])
    file_name = json_path.split("/")[-1].replace(".json", "")
    test_name = file_name.split("_")[1]
    for event in data:
        type = event['event_type']
        if type == "SYS_EVENT":
            if test_name in b1:
                df_line = pd.DataFrame(
                    {   'b1':index_b1,
                        'type':type,
                    },index=[0]
                )

            else:
                df_line = pd.DataFrame(
                    {
                        'b2':index_b2,
                        'type':type,
                    },index=[0]
                )
            df = df.append(df_line,ignore_index=True)
        if type == "gui" or type == "oracle":
            event_class = event['class']
            id = event['resource-id']
            text = event['text']
            content = event['content-desc']
            clickable = event['clickable']
            password = event['password']
            parent_text = event['parent_text']
            sibling_text = event['sibling_text']
            package = event['package']
            activity = event['activity']
            # ignorable = event['ignorable'] # not all have ignorable attribute
            ignorable = np.nan
            if 'ignorable' in event:
                ignorable = event['ignorable']
            action_full = event['action']
            action = action_full[0]
            value = np.nan
            oracle_desc = np.nan
            oracle_id = np.nan
            oracle_xpath = np.nan
            wait_time = np.nan
            if action == "click":
                value = np.nan
            if action == "send_keys_and_enter":
                value = action_full[1]
            if action == "wait_until_element_presence":
                wait_time = action_full[1]
                oracle_desc = action_full[2]
                if oracle_desc == "xpath":
                    oracle_xpath = action_full[3]
                if oracle_desc == "id":
                    oracle_id = action_full[3]
            if action == "send_keys_and_hide_keyboard":
                value = action_full[1]
            if action == "wait_until_text_presence":
                wait_time = action_full[1]
                oracle_desc = action_full[2] # text
                oracle_value = action_full[3]
            if action == "wait_until_text_invisible":
                wait_time = action_full[1]
                oracle_desc = action_full[2] # text
                oracle_value = action_full[3]
            if action == "swipe_right":
                value = np.nan
            if action == "send_keys":
                value = action_full[1]
            if action == "clear_and_send_keys":
                value = action_full[1]
            if action == "clear_and_send_keys_and_hide_keyboard":
                value = action_full[1]
            else:
                print("special action",action)
            if test_name in b1:
                df_line = pd.DataFrame(
                    {
                        'id':id,
                        'class':event_class,
                        'current Activity':activity,
                        'type':type,
                        'text':text,
                        'content':content,
                        'clickable':clickable,
                        'password':password,
                        'parent_text':parent_text,
                        'sibling_text':sibling_text,
                        'package':package,
                        'ignorable':ignorable,
                        'action':action,
                        'value':value,
                        'wait_time':wait_time,
                        'oracle_desc':oracle_desc,
                        'oracle_id':oracle_id,
                        'oracle_xpath':oracle_xpath,
                        'oracle_text':text,
                        'b1':index_b1,
                    },index=[0]
                )
            else:
                df_line = pd.DataFrame(
                    {
                        'id':id,
                        'class':event_class,
                        'current Activity':activity,
                        'type':type,
                        'text':text,
                        'content':content,
                        'clickable':clickable,
                        'password':password,
                        'parent_text':parent_text,
                        'sibling_text':sibling_text,
                        'package':package,
                        'ignorable':ignorable,
                        'action':action,
                        'value':value,
                        'wait_time':wait_time,
                        'oracle_desc':oracle_desc,
                        'oracle_id':oracle_id,
                        'oracle_xpath':oracle_xpath,
                        'oracle_text':text,
                        'b2':index_b2,
                    },index=[0]
                )
            df = df.append(df_line,ignore_index=True)
        else:
            print("special type",type)
        if test_name in b1:
            index_b1 = index_b1 + 1
        else:
            index_b2 = index_b2 + 1
    csv_name = json_path.split("/")[-1].replace(".json",".csv")
    df.to_csv(csv_save_path+csv_name)


def postprocess_groundtruth(groundtruth_path_prefix,csv_save_path):
    file_List = []
    get_all_file(groundtruth_path_prefix, file_List)
    file_pairs = dict() #a11_b11.csv and a11_b12.csv are the pair
    for index in range(len(file_List)):
        file_name = file_List[index].split("/")[-1]
        if '.DS_Store' in file_name:
            continue
        name_prefix = file_name.split("_")[0]
        if name_prefix in file_pairs:
            value = file_pairs[name_prefix]
            file_pairs[name_prefix]=[value[0],file_List[index]]
        else:
            value =[file_List[index]]
            file_pairs[name_prefix] = value
    for file_key in file_pairs:
        file_first = file_pairs[file_key][0]
        file_second = file_pairs[file_key][1]
        df_first = pd.read_csv(file_first)
        df_second = pd.read_csv(file_second)
        df = pd.concat([df_first,df_second]).drop_duplicates()
        csv_name = file_key+".csv"
        df.to_csv(csv_save_path+csv_name)

# delete column index for *.csv
def remove_duplicates(groundtruth_path_prefix):
    file_List = []
    get_all_file(groundtruth_path_prefix, file_List)
    for file_path in file_List:
        if '.DS_Store' in file_path:
            continue
        df_ori = pd.read_csv(file_path)
        df = df_ori.drop_duplicates()
        df.to_csv(file_path)



def main(json_file,csv_save_path):
    file_List = []
    get_all_file(json_file, file_List)
    for json_path in file_List:
        if '.DS_Store' in json_path:
            continue
        parse_json(json_path,csv_save_path)
    print("finish writing")







