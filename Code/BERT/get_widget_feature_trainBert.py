# coding=utf-8

"""
get the widget features from the xml files (benchmark)

this is for train bert
"""






from stage2.common import get_all_features, get_all_features_save_text, check_use_class
import os
from stage2.common import get_all_text_path, Blacklist
import pandas as pd
import json
from stage2.common import get_all_file
import time
import re

def get_feature_embedding(prefix,screen_file):
    ui_path_list = get_all_text_path(prefix,screen_file)
    blacklist = Blacklist()
    feature_map = get_all_features(ui_path_list, blacklist, prefix,screen_file)
    feature_modify_map = {}
    feature_key = []
    for key in feature_map:
        feature = feature_map[key]
        text_feature = feature[0].replace("_", " ").replace("/","").replace("\n\n","\n")
        text_feature = text_feature.strip()
        has_text_feature = 1
        if text_feature == "":
            has_text_feature = 0
            text_feature = "[UNK]"
        has_sibling_feature = feature[1][-1]
        has_other_feature = int(1 in feature[1][:-1])
        other_feature = feature[1][:-1]

        other_feature.append(has_sibling_feature)
        other_feature.append(has_text_feature)
        other_feature.append(has_other_feature)
        other_feature_num = len(other_feature)
        feature_modify = []
        feature_modify.append(text_feature)
        for feature in other_feature:
            feature_modify.append(feature)
        feature_modify_map[key] = feature_modify
        feature_key.append(key)
    return feature_modify_map, feature_key


def get_feature_embedding_revise(prefix,screen_file,id):
    ui_path_list = None
    if os.path.isdir(prefix+screen_file):
        ui_path_list = get_all_text_path(prefix,screen_file)
    else:
        ui_path_list = [prefix+screen_file]
    blacklist = Blacklist()
    feature_map = get_all_features(ui_path_list, blacklist, prefix,screen_file,id)
    feature_modify_map = {}
    feature_key = []
    for key in feature_map:
        feature = feature_map[key]
        text_feature = camel_case_split(feature[0])
        text_feature = text_feature.strip()
        has_text_feature = 1
        if text_feature == "":
            has_text_feature = 0
            text_feature = "[UNK]"
        other_feature = feature[1]

        other_feature.append(has_text_feature)
        other_feature_num = len(other_feature)
        feature_modify = []
        feature_modify.append(text_feature)
        for feature in other_feature:
            feature_modify.append(feature)
        feature_modify_map[key] = feature_modify
        feature_key.append(key)
    return feature_modify_map, feature_key



def get_feature_embedding_revise_json(prefix,screen_file,id):
    tgt_app_name = prefix.split("/")[-3]
    file_List = []
    get_all_file(prefix+screen_file,file_List)
    image_list = []
    for file in file_List:
        if file.endswith('.json'):
            image_list.append(file)
    feature_map = {}
    key_list = []
    for file_path in image_list:
        feature_list_by_widget = []
        with open(file_path,'r') as load_f:
            load_dict = json.load(load_f)
            views = load_dict['views']
            for view in views:
                view_class = view['class']
                class_checker = check_use_class(view_class)
                if class_checker == False:
                    continue
                view_str = view['view_str']
                text = ''
                if view['text'] != None:
                    text = view['text']
                content_desc = ''
                if view['content_description']!=None and view['content_description']!='':
                    content_desc = view['content_description']
                resource_id = ''
                if view['resource_id']!=None and view['resource_id']!='':
                    if 'id/' in view['resource_id']:
                        resource_id = view['resource_id'].split("id/")[1]
                    else:
                        resource_id = view['resource_id']
                text_feature_ori = text + ' ' + content_desc+ ' '+ resource_id
                text_feature = camel_case_split(text_feature_ori)

                widget_str = resource_id+"#"+ text +"#" + content_desc

                x_start = view['bounds'][0][0]
                x_end = view['bounds'][1][0]
                y_start = view['bounds'][0][1]
                y_end = view['bounds'][1][1]
                feature_list = [view_str,text_feature,resource_id, content_desc, text, x_start,x_end,y_start,y_end,widget_str]
                feature_list_by_widget.append(feature_list)
        for widget_index in range(len(feature_list_by_widget)):
            widget_information = feature_list_by_widget[widget_index]
            widget_str = widget_information[0]
            text_feature = widget_information[1]
            if 'uci homepage' in text_feature:
                print("here")
            if isinstance(widget_information[-1],str):
                other_feature = widget_information[2:-1]
            else:
                other_feature = widget_information[2:]
            key = ''
            key_prefix = tgt_app_name+"/"+file_path.split("/")[-1].replace(".json",".png").replace("state",'screen')
            key_suffix = widget_information[-1]
            key = key_prefix+":"+key_suffix

            key = key.strip()
            value = [text_feature, other_feature]
            feature_map[key] = value
            key_list.append(key)
    feature_modify_map = {}
    feature_key = []
    for key in feature_map:
        feature = feature_map[key]
        text_feature = feature[0].replace("_", " ").replace("/","").replace("\n\n","\n")
        text_feature = text_feature.strip()
        has_text_feature = 1
        if text_feature == "":
            has_text_feature = 0
            text_feature = "[UNK]"
        other_feature = feature[1]

        other_feature.append(has_text_feature)
        other_feature_num = len(other_feature)
        feature_modify = []
        feature_modify.append(text_feature)
        for feature in other_feature:
            feature_modify.append(feature)
        feature_modify_map[key] = feature_modify
        feature_key.append(key)
    return feature_modify_map, feature_key

def camel_case_split(identifier):

    special_word_dict = {
        'Sign_in':'signin',
        'sign_up':'signup',
        'sign_in':'signin',
        'Sign_up':'signup',
        'Log_In':'login',
        'Log_in':'login',
        '%':'percent',
        '#':'number',
        'et':'edit text',
        'btn':'button',
        'bt':'button',
        'tv':'text view',
        'fab':''
    }

    identifier = identifier.replace("\"","").replace("ICST","icst")
    for text in identifier.split(" "):
        if text in special_word_dict:
            identifier = identifier.replace(text,special_word_dict[text])
    splitted = re.sub('([A-Z][a-z]+)', r' \1', re.sub('([A-Z]+)', r' \1', identifier)).split()

    text_feature_without_camel_case = ''
    for revised_text in splitted:
        if '.com' in revised_text:
            revised_text = revised_text.split(".com")[0]+".com"
        if 'S.' in revised_text or 's.' in revised_text: # for a35
            revised_text = 'Smith'
        text_feature_without_camel_case += revised_text + " "
    text_feature = text_feature_without_camel_case.lower().replace("_", " ").replace("-", " ").replace("\b",
                                                                                                       " ").replace(
        "todo", "to do").replace("$","").replace(".0","").replace("sample.","sample").replace("to.", "to"). replace("do.", "do").strip()


    return text_feature



# other features are in the text
def get_feature_embedding_revise_save_all_text(prefix,screen_file,id):
    # unused
    ui_path_list = get_all_text_path(prefix,screen_file)
    blacklist = Blacklist()
    feature_map = get_all_features_save_text(ui_path_list, blacklist, prefix,screen_file,id)
    feature_modify_map = {}
    feature_key = []
    for key in feature_map:
        feature = feature_map[key]
        text_feature_ori = camel_case_split(feature[0])
        text_feature = feature[0].replace("_", " ").replace("/","").replace("\n\n","\n").replace("\b"," ")
        text_feature = text_feature.strip()
        other_feature = feature[1].replace("_", " ").replace("/","").replace("\n\n","\n")
        other_feature = other_feature.strip()
        feature_modify = []
        feature_modify.append(text_feature)
        feature_modify.append(other_feature)
        feature_modify_map[key] = feature_modify
        feature_key.append(key)
    return feature_modify_map, feature_key


def get_widget_from_json_xml(tgt_prefix, tgt_screen_file,id,src_prefix, src_name,feature_save_path,src_event_id,tgt_file_symbol,feature_save_full_path,tgt_event_id):
    """

    :param prefix: root_prefix (path)
    :param screen_file: xml_file_path
    :param id: the type for different text features (-1 is for all)
    :return:
    tgt_file_symbol == 'xml' or 'json'
    feature_save_full_path == full feature save path
    """
    tgt_feature_modify_map = None
    tgt_feature_key = None
    # # get target xml widgets
    if tgt_file_symbol == 'xml':
        tgt_feature_modify_map, tgt_feature_key = get_feature_embedding_revise(tgt_prefix,tgt_screen_file,id)
    # get target json widgets
    else:
        tgt_feature_modify_map, tgt_feature_key = get_feature_embedding_revise_json(tgt_prefix, tgt_screen_file, id)
    # get_source_event_widget
    # src_label_map = get_source_widget(source_widget_path,source_index)
    src_feature_modify_map = get_feature_embedding_from_csv(src_prefix, src_name,src_event_id)
    # widget1_list = []
    # feature1_list = []
    # for idx, row in src_feature_modify_map.iterrows():
    #     widget1 = row['widget1']
    #     feature1 = src_feature_modify_map[widget1]
    #     widget1_list.append(widget1)
    #     feature1_list.append(feature1)

    src_app_name = src_name.split("_")[0]
    tgt_full_path = tgt_prefix + tgt_screen_file
    if tgt_file_symbol == 'xml':
        tgt_app_name = tgt_full_path.split("/")[-2]
    else:
        tgt_app_name = tgt_full_path.split("/")[-3]
    data_list = []
    for widget1 in src_feature_modify_map:
        feature1 = src_feature_modify_map[widget1]
        other_feature1 =str_to_list(feature1)
        widget2_index = 0
        for widget2 in tgt_feature_modify_map:
            if src_app_name in widget2: # src and tgt are in the same app
                continue
            feature2=tgt_feature_modify_map[widget2]
            if feature2[0] == '[UNK]': # delete the none text features
                continue
            # for json file
            # add tgt_event_index in the widget2
            widget_2_revise = None
            if tgt_event_id is not None:
                widget_2_revise = widget2.split(":")[0]+":"+tgt_event_id+ widget2.split(":")[1] #a11/search.xml:b1-2/hierahchy/
            else:
                widget_2_revise = widget2
            if 'state' not in widget2:
                tgt_resource = feature2[1]
                tgt_content = feature2[2]
                tgt_text = feature2[3]
                tgt_x_start = feature2[4]
                tgt_x_end = feature2[5]
                tgt_y_start = feature2[6]
                tgt_y_end = feature2[7]
                data = {
                    "widget1":widget1,
                    "widget2":widget_2_revise,
                    "text_feature1":feature1[0],
                    "other_feature1":other_feature1,
                    "text_feature2":feature2[0],
                    "other_feature2":feature2[4:],
                    "tgt_resource":tgt_resource,
                    "tgt_content":tgt_content,
                    "tgt_text":tgt_text,
                    "tgt_x_start":tgt_x_start,
                    'tgt_x_end':tgt_x_end,
                    'tgt_y_start':tgt_y_start,
                    'tgt_y_end':tgt_y_end,
                    'label':1 # the label is a placeholder for fit the main_cosine.py
                }
                data_list.append(json.dumps(data)+"\n")

            else:
                tgt_resource = feature2[1]
                tgt_content = feature2[2]
                tgt_text = feature2[3]
                tgt_x_start = feature2[4]
                tgt_x_end = feature2[5]
                tgt_y_start = feature2[6]
                tgt_y_end = feature2[7]
                widget1_class = widget1.split(":")[1].split("/")[-1]
                widget2_class = widget2.split(":")[1].split("/")[-1]
                widget2_new = widget_2_revise.split(":")[0]+":"+str(widget2_index)
                if widget2_new == 'a41/states/xml_181340.xml:0' and 'a42/b2-0' in widget1:
                    print("here")
                data = {
                    "widget1":widget1.split(":")[0],
                    "widget2":widget_2_revise.split(":")[0]+":"+str(widget2_index),
                    "text_feature1":feature1[0],
                    "text_feature2":feature2[0],
                    "tgt_resource":tgt_resource,
                    "tgt_content":tgt_content,
                    "tgt_text":tgt_text,
                    'label':1, # the label is a placeholder
                    'widget1_class':widget1_class,
                    'widget2_class':widget2_class,
                }
                data_list.append(json.dumps(data)+"\n")
                widget2_index += 1




    if feature_save_full_path == None:
        end_time  = time.strftime('%Y%m%d%H%M',time.localtime())
        data_path = feature_save_path + src_app_name+"_"+tgt_app_name+"_test_data_"+end_time+ ".jsonl"
        with open(data_path, 'w') as jsonl:
            jsonl.writelines(data_list)
        print("data_path",data_path)
    if feature_save_path == None:
        with open(feature_save_full_path,'a+') as jsonl:
            jsonl.writelines(data_list)
        print("data_path", feature_save_full_path)

def str_to_list(src_feature):
    # example input = '[\\'\\naddToDoItemFAB\\n\\x0c\\', 6, 1, 0, 0, 1, 0, 0, 0, 0, 3, 0, 0, 1]'
    # output = [6, 1, 0, 0, 1, 0, 0, 0, 0, 3, 0, 0, 1]
    if len(src_feature) == 2:
        other_feature_str = src_feature[1].replace("[","").replace("]","")
        other_feature = other_feature_str.split(",")[1:]
        return other_feature
    else:
        return src_feature


def get_source_widget(source_widget_path,src_index):
    # src_index is not the index of the csv but the sequnece index (e.g., b1-0,b2-0)
    true_label_map = pd.DataFrame(columns=['widget1'], index=[])
    key_list = []
    df = pd.read_csv(source_widget_path)
    screen_x_files = ''
    xpath_x = ''
    if 'add-xpath' in df:
        xpath_x = df['add-xpath'].tolist()[src_index]
        screen_x_files = df['state-name'].tolist()[src_index]
    else:
        xpath_x = df['add_xpath'].tolist()[src_index]
        screen_x_files = df['state_name'].tolist()[src_index]
    screen_x = screen_x_files.strip().split("\t")[-1]  # the last screen can be match with xpath
    if ' ' in screen_x:  # space
        screen_x = screen_x.split(" ")[-1]
    widget1 = screen_x + ":" + xpath_x  # can change to entity1
    key_list.append(widget1)
    df_line = pd.DataFrame({
        'widget1': widget1
    }, index=[1])
    true_label_map = true_label_map.append(df_line, ignore_index=True)
    return true_label_map

def save_feature_for_source_widget(prefix,widget_csv_path,src_screen_file,id,src_screen_prefix):
    # input source_csv
    # process: [key,value] -- key = 'b1-0' value = ori_key (state_name + add_xpath)
    ui_path_list = []
    get_all_file(prefix+widget_csv_path,ui_path_list)
    src_feature_modify_map, src_feature_key = get_feature_embedding_revise(src_screen_prefix, src_screen_file, id)
    for idx in range(len(ui_path_list)):
        csv_path = ui_path_list[idx]
        if '.DS_Store' in csv_path:
            continue
        if 'groundtruth' in csv_path:
            continue
        if 'add_feature' in csv_path:
            continue

        print(csv_path)
        df = pd.read_csv(csv_path)
        src_widget_list_b1 = []
        src_widget_list_b2 = []
        for index in range(len(df)):
            if df.loc[index,'add_xpath']!=df.loc[index,'add_xpath']: # for SYS_EVENT and the event cannot find (np.nan)
                continue
            if df.loc[index,'b1'] ==df.loc[index,'b1']:  # np.nan
                src_index = df.loc[index,'b1']
                key = 'b1'+'-'+str(src_index)
                value = df.loc[index,'state_name'] + ":" + df.loc[index,'add_xpath']
                src_widget_list_b1.append([key,value])
            else:
                src_index = df.loc[index,'b2']
                key = 'b2' + '-' + str(src_index)
                value = df.loc[index, 'state_name'] + ":" + df.loc[index, 'add_xpath']
                src_widget_list_b2.append([key,value])
        for src_widget in src_widget_list_b1:
            key = src_widget[0]
            value = src_widget[1]
            if value == 'SYS_EVENT':
                continue
            feature = src_feature_modify_map[value]
            text_feature = feature[0]
            other_feature = feature[1:]
            b1_index= key.split("-")[1]
            row_index = df[df['b1'].isin([float(b1_index)])].index.tolist()[0]
            df.loc[row_index,'text_feature'] = text_feature
            df.loc[row_index,'other_feature'] = str(other_feature)
            df.loc[row_index,'x_start'] = feature[4]
            df.loc[row_index,'x_end'] = feature[5]
            df.loc[row_index,'y_start'] = feature[6]
            df.loc[row_index,'y_end'] = feature[7]
        for src_widget in src_widget_list_b2:
            key = src_widget[0]
            value = src_widget[1]
            if value == 'SYS_EVENT':
                continue
            feature = src_feature_modify_map[value]
            text_feature = feature[0]
            other_feature = feature[1:]
            b2_index= key.split("-")[1]
            row_index = df[df['b2'].isin([float(b2_index)])].index.tolist()[0]
            df.loc[row_index,'text_feature'] = text_feature
            df.loc[row_index,'other_feature'] = str(other_feature)
            df.loc[row_index,'x_start'] = feature[4]
            df.loc[row_index,'x_end'] = feature[5]
            df.loc[row_index,'y_start'] = feature[6]
            df.loc[row_index,'y_end'] = feature[7]
        csv_save_path = csv_path.replace(".csv","") + "_add_feature_1"+".csv"
        df.to_csv(csv_save_path)


def add_coordinate_on_groundtruth(groundtruth_path, src_csv_prefix, csv_save_path):
    # input: groundtruth_path and src_csv_path
    # output: groundtruth_path and coordinate
    df_groundtruth = pd.read_csv(groundtruth_path)
    for idx in range(len(df_groundtruth)):
        if idx == 739:
            print("here")
        if df_groundtruth.loc[idx, 'type'] == 'SYS_EVENT':
            continue
        if df_groundtruth.loc[idx, 'label'] == 0:
            continue
        src_app = df_groundtruth.loc[idx,'src_app']
        tgt_app = df_groundtruth.loc[idx, 'tgt_app']
        function = df_groundtruth.loc[idx, 'function']
        src_index = df_groundtruth.loc[idx, 'src_index']
        tgt_index = df_groundtruth.loc[idx, 'tgt_index']
        src_csv_path = src_csv_prefix + src_app[0:2] + "/" + src_app+"_revise_add_feature.csv"
        tgt_csv_path = src_csv_prefix + tgt_app[0:2] + "/" + tgt_app+"_revise_add_feature.csv"
        src_csv_coordinate_list = get_coordinate_from_csv(src_csv_path, function, src_index)
        tgt_csv_coordinate_list = get_coordinate_from_csv(tgt_csv_path, function, tgt_index)
        df_groundtruth.loc[idx,'src_x_start'] = src_csv_coordinate_list[0]
        df_groundtruth.loc[idx,'src_x_end'] = src_csv_coordinate_list[1]
        df_groundtruth.loc[idx,'src_y_start'] = src_csv_coordinate_list[2]
        df_groundtruth.loc[idx,'src_y_end'] = src_csv_coordinate_list[3]
        df_groundtruth.loc[idx,'tgt_x_start'] = tgt_csv_coordinate_list[0]
        df_groundtruth.loc[idx,'tgt_x_end'] = tgt_csv_coordinate_list[1]
        df_groundtruth.loc[idx,'tgt_y_start'] = tgt_csv_coordinate_list[2]
        df_groundtruth.loc[idx,'tgt_y_end'] = tgt_csv_coordinate_list[3]
    df_groundtruth.to_csv(csv_save_path)
    return df_groundtruth


def get_coordinate_from_csv(csv_path,function,event_index):
    df_csv = pd.read_csv(csv_path)
    if function[0] + function[2] == 'b1':
        df_line = df_csv[df_csv['b1'] == event_index]
        x_start = df_line.iloc[0].at['x_start']
        x_end = df_line.iloc[0].at['x_end']
        y_start = df_line.iloc[0].at['y_start']
        y_end = df_line.iloc[0].at['y_end']
        return [x_start, x_end,y_start,y_end]
    else:
        df_line = df_csv[df_csv['b2'] == event_index]
        x_start = df_line.iloc[0].at['x_start']
        x_end = df_line.iloc[0].at['x_end']
        y_start = df_line.iloc[0].at['y_start']
        y_end = df_line.iloc[0].at['y_end']
        return [x_start, x_end, y_start, y_end]














def get_feature_embedding_from_csv(csv_prefix,csv_name,event_id):
    # given csv
    # output dict (key = widget--; value = feature (text feature)
    df = pd.read_csv(csv_prefix+csv_name)
    app_name = csv_name.split("_")[0]
    src_feature_modify_map = dict()
    if event_id == -1: # all_csv
        for idx in range(len(df)):
            if df.loc[idx,'type'] == 'SYS_EVENT':
                continue
            widget_prefix = ''
            if df.loc[idx,'b1'] != df.loc[idx,'b1']:
                widget_num = df.loc[idx,'b2']
                if widget_num != widget_num:
                    continue
                widget_prefix = app_name+"/"+'b2-' + str(int(widget_num))
            else:
                widget_num = df.loc[idx,'b1']
                if widget_num != widget_num:
                    continue
                widget_prefix = app_name+"/"+'b1-'+str(int(widget_num))
            widget_xpath = df.loc[idx,'add_xpath']
            widget_key = widget_prefix+":"+widget_xpath
            text_feature_ori = df.loc[idx,'text_feature']
            text_feature = camel_case_split(text_feature_ori)
            text_feature = text_feature.strip()
            other_feature = df.loc[idx,'other_feature']
            widget_feature = [text_feature,other_feature]
            src_feature_modify_map[widget_key] = widget_feature
    else:
        # example event_id = 'b1-0'
        function = event_id.split("-")[0]
        e_id = int(event_id.split("-")[1])
        df_src_line = ''
        if function == 'b1':
            df_src_line = df[df['b1']==float(e_id)]
            widget_prefix = app_name+"/"+'b1-'+str(int(e_id))
        else:
            df_src_line = df[df['b2']==float(e_id)]
            widget_prefix = app_name+"/"+'b2-' + str(int(e_id))
        # # for SYS_EVENT:
        # idx = df_src_line.index.tolist()[0]
        # if df_src_line.loc[idx,'type'] == 'SYS_EVENT':
        #     continue
        idx = df_src_line.index.tolist()[0]
        widget_xpath = df_src_line.loc[idx,'add_xpath']
        widget_key = widget_prefix+":"+widget_xpath
        text_feature_ori = df_src_line.loc[idx,'text_feature']
        text_feature = camel_case_split(text_feature_ori)
        text_feature = text_feature.strip()
        other_feature = df_src_line.loc[idx,'other_feature']
        widget_feature = [text_feature,other_feature]
        src_feature_modify_map[widget_key] = widget_feature

    return src_feature_modify_map

