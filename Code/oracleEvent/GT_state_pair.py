
import pandas as pd

a1 = {'a11','a12','a13','a14','a15'}
a2 = {'a21','a22','a23','a24','a25'}
a3 = {'a31','a32','a33','a34','a35'}
a4 = {'a41','a42','a43','a44','a45'}
a5 = {'a51','a52','a53','a54','a55'}


def get_groundtruth_pair(groundtruth_path,widget_path_prefix):
    transfer_index = 0


    df = pd.DataFrame(columns=['src_app','tgt_app','function','src_ori_id','src_ori_xpath','src_add_id','src_add_xpath',
                               'tgt_ori_id','tgt_ori_xpath','tgt_add_id','tgt_add_xpath','src_text','tgt_text',
                               'src_content','tgt_content','src_state','tgt_state','src_index','tgt_index','type'])
    df_full = pd.DataFrame(columns=['src_app','tgt_app','function','src_ori_id','src_ori_xpath','src_add_id','src_add_xpath',
                               'tgt_ori_id','tgt_ori_xpath','tgt_add_id','tgt_add_xpath','src_text','tgt_text',
                               'src_content','tgt_content','src_state','tgt_state','src_index','tgt_index','type'])
    groundtruth_file = pd.read_csv(groundtruth_path)
    for index in range(len(groundtruth_file)):
        if groundtruth_file.loc[index,'step_to'] == -1:
            continue
        src_app = groundtruth_file.loc[index,'aid_from']
        tgt_app = groundtruth_file.loc[index,'aid_to']
        function = groundtruth_file.loc[index,'function']
        src_event_index = groundtruth_file.loc[index,'step_from']
        tgt_event_index = groundtruth_file.loc[index,'step_to']
        src_path = get_app_path(src_app,widget_path_prefix)
        tgt_path = get_app_path(tgt_app,widget_path_prefix)
        df_src = pd.read_csv(src_path)
        df_tgt = pd.read_csv(tgt_path)
        src_information = get_event_information(function, df_src, src_event_index)
        tgt_information = get_event_information(function,df_tgt,tgt_event_index)
        if src_information == 1 and tgt_information == 1:
            df_line = pd.DataFrame(
                {
                    'src_app':src_app,
                    'tgt_app':tgt_app,
                    'type':"SYS_EVENT",
                    'src_index':src_event_index,
                    'tgt_index':tgt_event_index,
                    'function':function,
                },index=[0]
            )
            df = df.append(df_line,ignore_index=True)
            df_full = df_full.append(df_line, ignore_index=True)
        elif len(src_information) == 3 or len(tgt_information) == 3:
            transfer_index = transfer_index + 1
            df_line = pd.DataFrame(
                {
                    'src_app': src_app,
                    'tgt_app': tgt_app,
                    'function': function,
                    'src_ori_id': src_information[0],
                    'src_ori_xpath': src_information[1],
                    'tgt_ori_id': tgt_information[0],
                    'tgt_ori_xpath': tgt_information[1],
                    'src_index':src_event_index,
                    'tgt_index':tgt_event_index,
                    'type': src_information[-1]

                },index=[0]
            )
            df_full = df_full.append(df_line,ignore_index=True)
            continue
        else:
            df_line = pd.DataFrame(
                {
                    'src_app':src_app,
                    'tgt_app':tgt_app,
                    'function':function,
                    'src_ori_id':src_information[0],
                    'src_ori_xpath':src_information[1],
                    'src_add_id':src_information[2],
                    'src_add_xpath':src_information[3],
                    'src_text':src_information[4],
                    'src_content':src_information[5],
                    'src_state':src_information[6],
                    'tgt_ori_id': tgt_information[0],
                    'tgt_ori_xpath': tgt_information[1],
                    'tgt_add_id': tgt_information[2],
                    'tgt_add_xpath': tgt_information[3],
                    'tgt_text': tgt_information[4],
                    'tgt_content': tgt_information[5],
                    'tgt_state': tgt_information[6],
                    'src_index':src_event_index,
                    'tgt_index':tgt_event_index,
                    'type':src_information[7],
                },index=[0]
            )
            df = df.append(df_line,ignore_index=True)
            df_full = df_full.append(df_line, ignore_index=True)
        transfer_index = transfer_index + 1
    return df, df_full, transfer_index



def get_app_path(app_name,widget_path_prefix):
    app_path = ''
    if app_name in a1:
        app_path = widget_path_prefix + 'a1/'+ app_name +"_revise.csv"
    elif app_name in a2:
        app_path = widget_path_prefix + 'a2/' + app_name + "_revise.csv"
    elif app_name in a3:
        app_path = widget_path_prefix + 'a3/' + app_name + "_revise.csv"
    elif app_name in a4:
        app_path = widget_path_prefix + 'a4/' + app_name + "_revise.csv"
    elif app_name in a5:
        app_path = widget_path_prefix + 'a5/' + app_name + "_revise.csv"
    return app_path


def get_event_information(function,groundtruth_file,event_index):
    event_line = ''
    if function[2] == '1':
        event_line = groundtruth_file.query('@event_index==b1')
    else:
        event_line = groundtruth_file.query('@event_index==b2')
    ori_id = event_line.iloc[0].at['ori_id']
    ori_xpath = event_line.iloc[0].at['ori_appium_xpath']
    add_id = event_line.iloc[0].at['add_id']
    add_xpath = event_line.iloc[0].at['add_xpath']
    text = event_line.iloc[0].at['text']
    content = event_line.iloc[0].at['content_desc']
    state = event_line.iloc[0].at['state_name']
    type = event_line.iloc[0].at['type']
    if type == 'SYS_EVENT':
        return 1
    else:
        if add_id != add_id and add_xpath!= add_xpath: # cannot find widget
            return [ori_id,ori_xpath,type]
        else:
            return [ori_id, ori_xpath, add_id, add_xpath, text, content, state,type]

# remove system_event
def get_groundtruth_pair_without_sys(groundtruth_path,widget_path_prefix):
    transfer_index = 0
    df = pd.DataFrame(columns=['src_app','tgt_app','function','src_ori_id','src_ori_xpath','src_add_id','src_add_xpath',
                               'tgt_ori_id','tgt_ori_xpath','tgt_add_id','tgt_add_xpath','src_text','tgt_text',
                               'src_content','tgt_content','src_state','tgt_state','src_index','tgt_index','type'])
    groundtruth_file = pd.read_csv(groundtruth_path)
    for index in range(len(groundtruth_file)):
        if groundtruth_file.loc[index,'step_to'] == -1:
            continue
        src_app = groundtruth_file.loc[index,'aid_from']
        tgt_app = groundtruth_file.loc[index,'aid_to']
        function = groundtruth_file.loc[index,'function']
        src_event_index = groundtruth_file.loc[index,'step_from']
        tgt_event_index = groundtruth_file.loc[index,'step_to']
        src_path = get_app_path(src_app,widget_path_prefix)
        tgt_path = get_app_path(tgt_app,widget_path_prefix)
        df_src = pd.read_csv(src_path)
        df_tgt = pd.read_csv(tgt_path)
        src_information = get_event_information(function, df_src, src_event_index)
        tgt_information = get_event_information(function,df_tgt,tgt_event_index)
        if src_information == -1 or tgt_information == -1:
            transfer_index = transfer_index + 1
            continue
        elif len(src_information) == 3 or len(tgt_information) == 3:
            transfer_index = transfer_index + 1
            continue
        else:
            df_line = pd.DataFrame(
                {
                    'src_app':src_app,
                    'tgt_app':tgt_app,
                    'function':function,
                    'src_ori_id':src_information[0],
                    'src_ori_xpath':src_information[1],
                    'src_add_id':src_information[2],
                    'src_add_xpath':src_information[3],
                    'src_text':src_information[4],
                    'src_content':src_information[5],
                    'src_state':src_information[6],
                    'tgt_ori_id': tgt_information[0],
                    'tgt_ori_xpath': tgt_information[1],
                    'tgt_add_id': tgt_information[2],
                    'tgt_add_xpath': tgt_information[3],
                    'tgt_text': tgt_information[4],
                    'tgt_content': tgt_information[5],
                    'tgt_state': tgt_information[6],
                    'src_index':src_event_index,
                    'tgt_index':tgt_event_index,
                    'type':src_information[7],
                },index=[0]
            )
            df = df.append(df_line,ignore_index=True)
        transfer_index = transfer_index + 1
    return df, transfer_index

# # postproces remove multiple ori_id xpath in the groundtruth_pair and groundtruth_pair_without_sys
def postprocess_groundtruth_pair(groundtruth_pair_path):
    df = pd.read_csv(groundtruth_pair_path)
    df_new = pd.DataFrame(columns=['src_app','tgt_app','src_ori_id','src_ori_xpath','src_add_id','src_add_xpath','tgt_ori_id',
                                   'tgt_ori_xpath','tgt_add_id','tgt_add_xpath','src_text','tgt_text','src_content','tgt_content',
                                   'src_state','tgt_state','match','type'],index=[])
    df_new_without_sys = pd.DataFrame(columns=df_new.columns.values.tolist(),index=[])
    df_new_index = 0
    df_new_without_sys_index = 0
    pair_signature_dict = dict()
    pair_signature_without_sys_dict = dict()
    for idx in range(len(df)):
        type = df.loc[idx,'type']
        if type == 'SYS_EVENT':
            src_index = df.loc[idx,'src_index']
            tgt_index = df.loc[idx,'tgt_index']
            function = df.loc[idx,'function']
            groundtruth_pair = function+":"+str(int(src_index))+"-"+str(int(tgt_index))
            df_line = pd.DataFrame(
                {
                    'src_app':df.loc[idx,'src_app'],
                    'tgt_app':df.loc[idx,'tgt_app'],
                    'type':df.loc[idx,'type'],
                    'match':groundtruth_pair,
                },index= [1]
            )
            df_new = df_new.append(df_line,ignore_index=True)
            df_new_index = df_new_index + 1
        else:
            src_add_xpath = df.loc[idx,'src_add_xpath']
            tgt_add_xpath = df.loc[idx,'tgt_add_xpath']
            src_app = df.loc[idx,'src_app']
            tgt_app = df.loc[idx,'tgt_app']
            src_index = df.loc[idx, 'src_index']
            tgt_index = df.loc[idx, 'tgt_index']
            function = df.loc[idx, 'function']
            if src_add_xpath!=src_add_xpath or tgt_add_xpath!=tgt_add_xpath:
                groundtruth_pair= function+":"+str(int(src_index))+"-"+str(int(tgt_index))
                df_line = pd.DataFrame(
                    {
                        'src_app': src_app,
                        'tgt_app': tgt_app,
                        'src_ori_id': df.loc[idx, 'src_ori_id'],
                        'src_ori_xpath': df.loc[idx, 'src_ori_xpath'],
                        'tgt_ori_id': df.loc[idx, 'tgt_ori_id'],
                        'tgt_ori_xpath': df.loc[idx, 'tgt_ori_xpath'],
                        'match':groundtruth_pair,
                        'type': df.loc[idx,'type'],
                    },index=[1]
                )
                df_new = df_new.append(df_line,ignore_index=True)
                df_new_index = df_new_index + 1
            else:
                pair_signature = src_app + "-" + src_add_xpath + "-" + tgt_app + "-" + tgt_add_xpath
                groundtruth_pair = function + ":" + str(int(src_index)) + "-" + str(int(tgt_index))
                if pair_signature not in pair_signature_dict:
                    df_line = pd.DataFrame(
                        {
                            'src_app': df.loc[idx, 'src_app'],
                            'tgt_app': df.loc[idx, 'tgt_app'],
                            'src_ori_id': df.loc[idx, 'src_ori_id'],
                            'src_ori_xpath': df.loc[idx, 'src_ori_xpath'],
                            'src_add_id': df.loc[idx, 'src_add_id'],
                            'src_add_xpath': df.loc[idx, 'src_add_xpath'],
                            'tgt_ori_id': df.loc[idx, 'tgt_ori_id'],
                            'tgt_ori_xpath': df.loc[idx, 'tgt_ori_xpath'],
                            'tgt_add_id': df.loc[idx, 'tgt_add_id'],
                            'tgt_add_xpath':df.loc[idx,'tgt_add_xpath'],
                            'src_text':df.loc[idx,'src_text'],
                            'tgt_text':df.loc[idx,'tgt_text'],
                            'src_content':df.loc[idx,'src_content'],
                            'tgt_content':df.loc[idx,'tgt_content'],
                            'src_state':df.loc[idx,'src_state'],
                            'tgt_state':df.loc[idx,'tgt_state'],
                            'match':groundtruth_pair,
                            'type':df.loc[idx,'type'],

                        },index=[1]
                    )
                    df_new = df_new.append(df_line,ignore_index=True)
                    pair_signature_dict[pair_signature] = df_new_index
                    df_new_index = df_new_index + 1

                    df_new_without_sys = df_new_without_sys.append(df_line,ignore_index=True)
                    pair_signature_without_sys_dict[pair_signature] = df_new_without_sys_index
                    df_new_without_sys_index = df_new_without_sys_index + 1

                else:
                    pair_index = pair_signature_dict[pair_signature]
                    df_new.loc[pair_index,'match'] = df_new.loc[pair_index,'match'] + "," + groundtruth_pair

                    pair_without_index = pair_signature_without_sys_dict[pair_signature]
                    df_new_without_sys.loc[pair_without_index,'match'] = df_new_without_sys.loc[pair_without_index,'match']+ "," + groundtruth_pair
    return df_new,df_new_without_sys





if __name__ == "__main__":

    csv_name = 'true_groundtruth_pair.csv'
    groundtruth_file_path = ''
    test_case_path = ''
    df, df_full,transfer_index = get_groundtruth_pair(groundtruth_file_path,test_case_path)
    write_csv(df,csv_name)
    print("df_len",len(df))
    print("transfer_index",transfer_index)







