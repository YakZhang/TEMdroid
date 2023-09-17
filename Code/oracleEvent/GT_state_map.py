
import pandas as pd
from stage2.get_same_state import get_all_file, get_image_file
from lxml import etree
from stage2.GT_state_map_revise import get_xpath_from_node_lxml, revise_xpath,get_nodelist_by_id,write_csv
import os



def load_csv(csv_path):
    df = pd.read_csv(csv_path)
    id_list = df['id'].tolist()  # str and float(nan)
    xpath_list = df['oracle_xpath'].tolist()
    type_list =df['type'].tolist()
    action_list = df['action'].tolist()
    activity_list = df['current Activity']
    b1_list = df['b1']
    b2_list = df['b2']
    state_list = []
    if 'state_name' in df.columns:
        state_list = df['state_name']

    return id_list, xpath_list,type_list, action_list, activity_list,b1_list,b2_list, state_list


def map_event_state(state_path, id_list, xpath_list,type_list,action_list,activity_list,b1_list,b2_list,current_state_list,state_path_prefix):


    row_num = len(id_list)
    # df_map = pd.DataFrame(np.zeros((row_num, 9)), columns=['add_id', 'add_xpath', 'state_name', 'content_desc', 'text','canonical','ori_id','ori_appium_xpath','ori_uiautomator_xpath'])
    df_map = pd.DataFrame(columns=['add_id', 'add_xpath', 'state_name', 'content_desc', 'text','type','ori_id','ori_appium_xpath','action','current_activity','b1','b2'],index=list(range(0,row_num)))
    # # get an empty dataframe
    # df_map = pd.DataFrame(columns=['add_id', 'add_xpath', 'state_name', 'content_desc', 'text', 'canonical','ori_appium_id','ori_appium_xpath','ori_id','ori_appium_xpath','ori_uiautomator_xpath'], index=[])


    file_List = []
    get_all_file(state_path, file_List)
    state_list_xml = get_image_file(file_List, suffix='.xml')
    state_list = state_list_xml
    prefix = state_path_prefix

    if len(current_state_list) == 0:
        # given the id, find the xpath and others
        for index in range(len(id_list)):
            id = id_list[index]
            xpath = xpath_list[index]
            type = type_list[index]
            action = action_list[index]
            activity = activity_list[index]
            b1 = b1_list[index]
            b2 = b2_list[index]
            if isinstance(xpath, str):
                get_information_from_xpath(state_list,prefix, df_map,xpath,index,id,type,action,activity,b1,b2)
            elif isinstance(id,str):
                get_information_from_id(state_list, prefix, df_map, id, index, xpath,type,action,activity,b1,b2)
            else: # System event
                get_information_system(df_map,type,index,b1,b2)
    else:
        # given the id, state_name, find the xpath and others in the current state_name
        for index in range(len(id_list)):
            id = id_list[index]
            xpath = xpath_list[index]
            type = type_list[index]
            action = action_list[index]
            activity = activity_list[index]
            b1 = b1_list[index]
            b2 = b2_list[index]
            state = [current_state_list[index]]
            if isinstance(xpath, str):
                get_information_from_xpath(state,prefix, df_map,xpath,index,id,type,action,activity,b1,b2)
            elif isinstance(id, str):
                get_information_from_id(state, prefix, df_map, id, index, xpath, type, action, activity, b1, b2)
            else:
                get_information_system(df_map, type, index, b1, b2)


    return df_map

def get_information_system(df_map,type,index,b1,b2):
    df_map.loc[index,'type'] = type
    df_map.loc[index,'b1'] = b1
    df_map.loc[index,'b2'] = b2

def get_information_from_id(state_list, prefix,df_map,id,index,xpath,type,action,activity,b1,b2):
    df_map.loc[index, 'ori_id'] = id
    df_map.loc[index, 'ori_appium_xpath'] = xpath
    df_map.loc[index,'type'] = type
    df_map.loc[index,'action']=action
    df_map.loc[index,'activity']=activity
    df_map.loc[index,'b1'] = b1
    df_map.loc[index,'b2'] = b2
    # print(index)
    if len(state_list) > 1:
        for state_path in state_list:
            file_name = state_path.replace(prefix, "")
            # print(state_path)
            tree = etree.parse(state_path)
            root = tree.getroot()
            nodelist = []
            get_nodelist_by_id(root, id,nodelist)
            if len(nodelist) >0:
                chose_node = ''
                if len(nodelist) >1:
                    # for test
                    print("more than one node has the same xpath in one .uix")
                    print("state_path",state_path)
                    print("resource_id",id)

                chose_node = nodelist[0]
                df_map.loc[index,'add_id'] = id
                if chose_node.get("text") != "":
                    df_map.loc[index,'text'] = chose_node.get("text")
                if chose_node.get("content-desc") != "":
                    df_map.loc[index,'content_desc'] = chose_node.get("content-desc")
                if (isinstance(df_map.loc[index]['state_name'], float)):  # float{nan}
                    df_map.loc[index,'state_name'] = file_name  # initial the string not the nan
                else:
                    df_map.loc[index,'state_name'] = df_map.loc[index,'state_name'] + "\t" + file_name
                df_map.loc[index,'add_xpath'] = get_xpath_from_node_lxml(chose_node)
    else:
        state_name = state_list[0]
        if isinstance(state_name,str): # judge whether the state_name exists
            state_path = prefix + state_name
            tree = etree.parse(state_path)
            root = tree.getroot()
            nodelist = []
            get_nodelist_by_id(root, id, nodelist)
            if len(nodelist) > 0:
                chose_node = ''
                if len(nodelist) > 1:
                    # for test
                    print("more than one node has the same xpath in one .uix")
                    print("state_path", state_path)
                    print("resource_id", id)

                chose_node = nodelist[0]
                df_map.loc[index, 'add_id'] = id
                if chose_node.get("text") != "":
                    df_map.loc[index, 'text'] = chose_node.get("text")
                if chose_node.get("content-desc") != "":
                    df_map.loc[index, 'content_desc'] = chose_node.get("content-desc")
                df_map.loc[index, 'state_name'] = state_name  # initial the string not the nan
                df_map.loc[index, 'add_xpath'] = get_xpath_from_node_lxml(chose_node)





def get_information_from_xpath(state_list,prefix,df_map,xpath,index,id,type,action,activity,b1,b2):
    # get node from xpath
    # print(index)
    # print(xpath)
    xpath_fit = revise_xpath(xpath)
    df_map.loc[index, 'ori_id'] = id
    df_map.loc[index, 'ori_appium_xpath'] = xpath
    df_map.loc[index,'type']=type
    df_map.loc[index,'action']=action
    df_map.loc[index,'activity']=activity
    df_map.loc[index,'b1'] = b1
    df_map.loc[index,'b2'] = b2
    if len(state_list) > 1:
        for state_path in state_list:
            file_name = state_path.replace(prefix, "")
            tree = etree.parse(state_path)
            root = tree.getroot()
            nodelist = []
            if '.uix' in state_path:
                findall = etree.XPath(xpath_fit)
                nodelist = findall(root)  # get the work
            elif '.xml' in state_path:
                findall = etree.XPath(xpath)
                nodelist = findall(root)
            if len(nodelist) != 0:
                # identify whether the there are more than one node has the same path
                chose_node = ''
                if len(nodelist) > 1:
                    # for test
                    print("more than one node has the same xpath in one .uix")
                    print("state_path",state_path)
                    # for node in nodelist:
                    #     if id!=id and node.get("resource-id")=="" or id==id and node.get("resource-id")==id :
                    #         chose_node = node
                    #         break

                chose_node = nodelist[0]
                if chose_node.get("resource-id") != "":
                    df_map.loc[index,'add_id'] = chose_node.get("resource-id")
                if chose_node.get("text") != "":
                    df_map.loc[index,'text']= chose_node.get("text")
                if chose_node.get("content-desc") != "":
                    df_map.loc[index,'content_desc'] = chose_node.get("content-desc")
                if (isinstance(df_map.loc[index,'state_name'], float)):  # float{nan}
                    df_map.loc[index,'state_name'] = file_name  # initial the string not the nan
                else:
                    df_map.loc[index,'state_name'] = df_map.loc[index,'state_name'] + "\t" + file_name
                df_map.loc[index, 'add_xpath'] = get_xpath_from_node_lxml(chose_node)
    else:
        state_name = state_list[0]
        if isinstance(state_name,str):
            state_path = prefix + state_name
            tree = etree.parse(state_path)
            root = tree.getroot()
            nodelist = []
            if '.uix' in state_path:
                findall = etree.XPath(xpath_fit)
                nodelist = findall(root)  # get the work
            elif '.xml' in state_path:
                findall = etree.XPath(xpath)
                nodelist = findall(root)
            if len(nodelist) != 0:
                # identify whether the there are more than one node has the same path
                chose_node = ''
                if len(nodelist) > 1:
                    # for test
                    print("more than one node has the same xpath in one .uix")
                    print("state_path", state_path)


                chose_node = nodelist[0]
                if chose_node.get("resource-id") != "":
                    df_map.loc[index, 'add_id'] = chose_node.get("resource-id")
                if chose_node.get("text") != "":
                    df_map.loc[index, 'text'] = chose_node.get("text")
                if chose_node.get("content-desc") != "":
                    df_map.loc[index, 'content_desc'] = chose_node.get("content-desc")
                df_map.loc[index, 'state_name'] = state_name  # initial the string not the nan
                df_map.loc[index, 'add_xpath'] = get_xpath_from_node_lxml(chose_node)






# save only one ori_id in one file
def remove_dumplicate_widget(csv_path):
    df = pd.read_csv(csv_path)
    df_column_name = df.columns.values.tolist()
    df_new = pd.DataFrame(columns=df_column_name,index=[])
    df_new_index = 0
    widget_signature_dict = dict() # key: id-xpath value:new_index
    for idx in range(len(df)):
        add_id = df.loc[idx,'add_id']
        if add_id != add_id: # np.nan
            add_id = ""
        add_xpath = df.loc[idx,'add_xpath']
        if add_xpath != add_xpath:
            add_xpath = ""
        widget_signature = add_id + "-" + add_xpath
        b1 = df.loc[idx,'b1']
        if b1 == b1:
            b1 = str(int(b1))
        else:
            b1 = b1
        b2 = df.loc[idx,'b2']
        if b2 == b2:
            b2 = str(int(b2))
        else:
            b2 = b2
        action = df.loc[idx,'action']
        if widget_signature not in widget_signature_dict:
            df_line = pd.DataFrame(
                {
                    'add_id':add_id,
                    'add_xpath':add_xpath,
                    'state_name':df.loc[idx,'state_name'],
                    'content_desc':df.loc[idx,'content_desc'],
                    'text':df.loc[idx,'text'],
                    'ori_id':df.loc[idx,'ori_id'],
                    'ori_appium_xpath':df.loc[idx,'ori_appium_xpath'],
                    'action':action,
                    'activity':df.loc[idx,'activity'],
                    'b1':b1,
                    'b2':b2,
                    'type':df.loc[idx,'type']
                },index=[0]
            )
            df_new = df_new.append(df_line,ignore_index=True)
            widget_signature_dict[widget_signature] = df_new_index
            df_new_index = df_new_index + 1
        else:
            index_exist = widget_signature_dict[widget_signature]
            if b1==b1: #not np.nan
                b1_ori = df_new.loc[index_exist,'b1']
                if b1_ori == b1_ori:
                    df_new.loc[index_exist,'b1'] = df_new.loc[index_exist,'b1'] + "\t" + str(b1)
                else:
                    df_new.loc[index_exist, 'b1'] = str(int(b1))
            if b2==b2:
                b2_ori = df_new.loc[index_exist,'b2']
                if b2_ori == b2_ori:
                    df_new.loc[index_exist, 'b2'] = df_new.loc[index_exist, 'b2'] + "\t" + str(b2)
                else:
                    df_new.loc[index_exist,'b2']=str(int(b2))
    return df_new




def main(csv_paths, state_path_prefix, save_path_prefix,dumplicate_save_path_prefix):
    # get GT_revise
    file_List = []
    get_all_file(csv_paths, file_List)
    for csv_path in file_List:
        if '.DS_Store' in csv_path:
            continue
        id_list, xpath_list,type_list,action_list,activity_list,b1_list,b2_list,state_list = load_csv(csv_path)
        prefix = csv_path.replace(csv_paths, "").replace(".csv", "")
        state_path = state_path_prefix + prefix.replace("GT_", "")
        df_map = map_event_state(state_path, id_list, xpath_list,type_list,action_list,activity_list,b1_list, b2_list,state_list, state_path_prefix)
        revise_csv_name = prefix + "_revise.csv"
        if not os.path.exists(save_path_prefix):
            os.makedirs(save_path_prefix)
        write_csv(df_map, save_path_prefix + revise_csv_name)
    print("finish writing")

    # remove multiple same id/xpath
    if not os.path.exists(save_path_prefix):
        os.makedirs(save_path_prefix)
    if not os.path.exists(dumplicate_save_path_prefix):
        os.makedirs(dumplicate_save_path_prefix)
    file_List = []
    get_all_file(save_path_prefix,file_List)
    for csv_path in file_List:
        if '.DS_Store' in csv_path:
            continue
        df_new = remove_dumplicate_widget(csv_path)
        csv_name = csv_path.replace(save_path_prefix,"")
        write_csv(df_new,dumplicate_save_path_prefix+csv_name)


