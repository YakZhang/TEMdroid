
import pandas as pd


def generate_false_groundtruth(temdroid_full_result_file,pair_type_dict):
    df= pd.read_csv(temdroid_full_result_file)
    df_new = pd.DataFrame(columns=['src_app','tgt_app','function','src_ori_id','src_ori_xpath','src_add_id','src_add_xpath',
                               'tgt_ori_id','tgt_ori_xpath','tgt_add_id','tgt_add_xpath','src_text','tgt_text',
                               'src_content','tgt_content','src_state','tgt_state','src_index','tgt_index','type','label','predict_label'])
    for idx in range(len(df)):
        if df.loc[idx,'label'] == 1:
            continue
        src_widget = df.loc[idx,'widget1']
        tgt_widget = df.loc[idx,'widget2']
        src_app = src_widget.split("/")[0]
        src_add_xpath = src_widget.split(":")[1]
        src_state_name = src_widget.split(":")[0]
        tgt_app = tgt_widget.split("/")[0]
        tgt_add_xpath = tgt_widget.split(":")[1]
        tgt_state_name = tgt_widget.split(":")[0]
        predict_label = df.loc[idx,'predict_label']
        predict_score = df.loc[idx,'predict_score']
        type = pair_type_dict[src_add_xpath]
        assert df.loc[idx,'label']==0
        df_line = pd.DataFrame(
            {
                'src_app':src_app,
                'tgt_app':tgt_app,
                'src_add_xpath':src_add_xpath,
                'tgt_add_xpath':tgt_add_xpath,
                'src_state':src_state_name,
                'tgt_state':tgt_state_name,
                'type':type,
                'label':0,
                'predict_label':predict_label,
                'predict_score':predict_score
            },index=[1]
        )
        df_new = df_new.append(df_line,ignore_index=True)
    return df_new


def get_pair_type(groundtruth_full_pair_file):
    df = pd.read_csv(groundtruth_full_pair_file)
    pair_type_dict = {}
    for idx in range(len(df)):
        src_add_xpath = df.loc[idx,'src_add_xpath']
        tgt_add_xpath = df.loc[idx,'tgt_add_xpath']
        if src_add_xpath!=src_add_xpath or tgt_add_xpath!=tgt_add_xpath:
            continue
        type = df.loc[idx,'type']
        pair_type_dict[src_add_xpath] = type
    return pair_type_dict







