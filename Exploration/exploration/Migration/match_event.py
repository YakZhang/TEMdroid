


import pandas as pd
import numpy as np
import os
import copy
import heapq



def add_new_function(predict_map_path,new_function_threahold,predict_map_save_path):
    df_predict = pd.read_csv(predict_map_path)
    df_predict['ori_predict_score'] = copy.deepcopy(df_predict['predict_score'])
    for idx in range(len(df_predict)):
        ori_text_feature1 = df_predict.loc[idx,'text_feature1']
        ori_text_feature2 = df_predict.loc[idx,'text_feature2']
        ori_text_feature1 = ori_text_feature1.replace("btn","button").replace("login","log in")
        ori_text_feature2 = ori_text_feature2.replace("btn", "button").replace("login", "log in")

        text_feature1 = ori_text_feature1.strip().split(" ")
        text_feature2 = ori_text_feature2.strip().split(" ")
        text_feature1_copy = copy.deepcopy(text_feature1)
        ori_predict_score = df_predict.loc[idx,'ori_predict_score']
        common_words = list()
        for text in text_feature2:
            if len(text_feature1)>0 and text in text_feature1:
                common_words.append(text)
                text_feature1.remove(text)
        same_words_percentage = len(common_words)/len(text_feature1_copy)
        df_predict.loc[idx,'same_words_percentage'] = same_words_percentage
        revise_predict_score = same_words_percentage*new_function_threahold + ori_predict_score*(1-new_function_threahold)
        df_predict.loc[idx,'predict_score'] = revise_predict_score
    df_predict.to_csv(predict_map_save_path)
    return df_predict

def get_match_widget(threhold,predict_map_path,match_csv_path,back_count):

    # add new rule: if widget1_class include button, widget2 not edit text
    df_predict = pd.read_csv(predict_map_path,encoding='latin-1')
    df_predict_choose = pd.DataFrame(columns=df_predict.columns)
    for idx in range(len(df_predict)):
        widget1_class = df_predict.loc[idx,'widget1'].split("/")[-1].lower().split("\'")[1]
        widget2_class = df_predict.loc[idx,'widget2'].split("/")[-1].lower().split("\'")[1]
        if 'button' in widget1_class and 'edittext' in widget2_class:
            continue
        if 'edittext' in widget1_class and 'button' in widget2_class:
            continue
        df_line = df_predict.loc[idx].to_frame().T
        if len(df_predict_choose) == 0:
            df_predict_choose = df_line
        else:
            df_predict_choose = pd.concat([df_predict_choose,df_line],axis=0, ignore_index=True)
    predict_scores = df_predict_choose['predict_score']
    choose_predict_score = heapq.nlargest(back_count+1,predict_scores)[back_count]
    if choose_predict_score > threhold:
        df_choose = df_predict_choose.query("predict_score == @choose_predict_score")
        match_widget = [df_choose.iloc[0].at['tgt_x_start'],df_choose.iloc[0].at['tgt_x_end'],
                        df_choose.iloc[0].at['tgt_y_start'],df_choose.iloc[0].at['tgt_y_end']]
        print("choose predict score:",choose_predict_score)
        print("choose tgt_resource",df_choose.iloc[0].at['tgt_resource'])
        print("choose tgt_content", df_choose.iloc[0].at['tgt_content'])
        print("choose tgt_text", df_choose.iloc[0].at['tgt_text'])
        print("choose tgt_class",df_choose.iloc[0].at['widget2'].split("/")[-1].lower().split("\'")[1])
        if os.path.exists(match_csv_path)==False:
            df_choose.to_csv(match_csv_path)
        else:
            df = pd.read_csv(match_csv_path)
            df_new = pd.concat([df,df_choose],axis=0,ignore_index=True)
            df_new.to_csv(match_csv_path)
        return match_widget,df_choose
    else:
        return [],None



def generate_match_event(match_widget, src_csv_path,src_event_id):
    # output widget coordinate (center_x, center_y) and event type and related parameter (e.g., input)
    df_src = pd.read_csv(src_csv_path)
    # example event_id = 'b1-0'
    function = src_event_id.split("-")[0]
    e_id = int(src_event_id.split("-")[1])
    df_src_line = ''
    if function == 'b1':
        df_src_line = df_src[df_src['b1'] == float(e_id)]
    else:
        df_src_line = df_src[df_src['b2'] == float(e_id)]
    idx = df_src_line.index.tolist()[0]
    type = df_src_line.loc[idx,'type']
    input = df_src_line.loc[idx,'input']
    if input == input and isinstance(input,str):
        input = input.replace("\"","")
    action = df_src_line.loc[idx,'action']
    droidbot_type = np.nan
    droidbot_parameter = np.nan
    if action == 'click':
        droidbot_type = 'touch'
    if 'send' in action:
        if 'enter' in action:
            droidbot_type = 'set_text_and_enter'
        else:
            droidbot_type = 'set_text'
        droidbot_parameter = input
    if action == 'swipe_right':
        droidbot_type = 'scroll'
        droidbot_parameter = 'RIGHT'
    if action == 'long_press':
        droidbot_type = 'long_touch'
    if type == 'oracle':
        droidbot_type = 'oracle'
        if action == 'wait_until_text_invisible':
            droidbot_parameter = 'disappear'
    center_x = (match_widget[0] + match_widget[1]) / 2
    center_y = (match_widget[2] + match_widget[3]) / 2
    return [center_x,center_y,droidbot_type,droidbot_parameter]





