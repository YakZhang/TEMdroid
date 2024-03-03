# coding=utf-8

"""
common utils

"""

from lxml import etree
import cv2
import pytesseract as pt
import cmath
import os
import copy
import pandas as pd
import numpy as np
from stage2.GT_state_map_revise import get_xpath_from_node_lxml
from stage2.get_same_state import get_all_file, get_image_file
from oracleEvent.groundtruth_true_false_generation import get_row_index_from_df

XML_PATH = "/Users/Fruiter/TestBenchmark-Java-client-master/screenshots/shopping/home/state_spinner.uix"



def check_unuse_class(class_name):
    if 'Button' not in class_name and 'View' not in class_name and 'EditText' not in class_name and 'LinearLayoutCompat' not in class_name:
        return False
    else:
        return True

def check_use_class(class_name):
    if 'android.widget.Button' in class_name or 'android.widget.EditText' in class_name or 'LinearLayoutCompat' in class_name or 'android.widget.TextView' in class_name or 'MultiAutoCompleteTextView' in class_name\
            or 'android.widget.ImageButton' in class_name or 'android.view.View' in class_name:
        return True
    else:
        return False
    
def get_nodelist(root):
    nodelist = []

    def walk_root(node):
        nodelist.append(node)
        for child in node:
            walk_root(child)
    walk_root(root)
    return nodelist[1:]  # Delete root

def load_jsonl(jsonline_path):
    data = []
    with open(jsonline_path) as jsonl:
        data = jsonl.readlines()
    return data


def Blacklist():
    blacklist = {}

    return blacklist

def extract_ori_feature_revise(XML_PATH, feature_list_by_widget,id):
    widget_class_dict = dict()
    tree = etree.parse(XML_PATH)
    root = tree.getroot()
    search_widgets = get_nodelist(root)
    for widget_list in search_widgets:
        feature_list = []
        ori_class = widget_list.get("class")
        # class_checker = check_unuse_class(ori_class)
        class_checker = check_use_class(ori_class)
        if class_checker == False:
            continue
        if ori_class:
            widget_class = ori_class.replace("android.widget.", "")
        else:
            widget_class = ""
        widget_text = widget_list.get("text")
        widget_bounded = widget_list.get("bounds")
        former_coordination = widget_bounded.split("][")[0]
        latter_coordination = widget_bounded.split("][")[1]
        former_x_bound = int(former_coordination.split(",")[0][1:])
        former_y_bound = int(former_coordination.split(",")[1])
        latter_x_bound = int(latter_coordination.split(",")[0])
        latter_y_bound = int(latter_coordination.split(",")[1][:-1])
        if former_x_bound < 0:
            former_x_bound = 0
        if latter_x_bound < 0:
            latter_x_bound = 0
        widget_bounded = [former_x_bound, former_y_bound,
                          latter_x_bound, latter_y_bound]
        widget_square = (latter_x_bound - former_x_bound) * \
            (latter_y_bound - former_y_bound)
        widget_content = widget_list.get("content-desc")
        widget_resource = widget_list.get("resource-id")
        if widget_resource != "":
            if ":id/" in widget_resource:
                widget_resource = widget_resource.split(":id/")[1]
            # if widget_resource =="com.mobilesrepublic.appy:id/search_bar_layout":
            #     print("here")
        widget_text_size = 0   # initial
        if widget_text != "":
            text_num = len(widget_text.split("\n"))
            # need to reduce the widget_text_size
            widget_text_size = cmath.sqrt(widget_square) / text_num
        widget_clickable = 0
        widget_checkable = 0
        widget_enable = 0
        widget_focusable = 0
        widget_scrollable = 0
        widget_long_clickable = 0
        widget_password = 0
        widget_selected = 0
        if widget_list.get("clickable") == 'true':
            widget_clickable = 1
        if widget_list.get("checkable") == 'true':
            widget_checkable = 1
        if widget_list.get("enable") == 'true':
            widget_enable = 1
        if widget_list.get("focusable") == 'true':
            widget_focusable = 1
        if widget_list.get("scrollable") == 'true':
            widget_scrollable = 1
        if widget_list.get("long-clickable") == 'true':
            widget_long_clickable = 1
        if widget_list.get("password") == 'true':
            widget_password = 1
        if widget_list.get("selected") == 'true':
            widget_selected = 1
        # identify sibling
        sibling_num = 0
        if len(widget_list.getparent()) != 0:
            sibling_num = len(widget_list.getparent())

        widget_all_text = widget_text +" "+ widget_content +" "+ widget_resource

        # get the xpath from the node
        xpath = get_xpath_from_node_lxml(widget_list)

        # for widget_class
        if widget_class in widget_class_dict:
            widget_class_index =widget_class_dict[widget_class]
        else:
            widget_class_index = len(widget_class_dict) 
            widget_class_dict[widget_class] = widget_class_index

        # for relative postion
        x_start = widget_bounded[0]
        y_start = widget_bounded[1]
        x_end = widget_bounded[2]
        y_end = widget_bounded[3]
        widget_center_x = (x_start + x_end) / 2
        widget_center_y = (y_start + y_end) / 2
        image_center_x = (0 + 1440) / 2
        image_center_y = (0 + 2960) / 2
        left_relative_position = 0
        top_relative_position = 0
        if widget_center_x < image_center_x:
            left_relative_position = 1
        if widget_center_y < image_center_y:
            top_relative_position = 1


        feature_list.append(xpath)
        feature_list.append(widget_all_text) # for without text
        feature_list.append(widget_resource)
        feature_list.append(widget_content)
        feature_list.append(widget_text)
        feature_list.append(x_start)
        feature_list.append(x_end)
        feature_list.append(y_start)
        feature_list.append(y_end)



        feature_list_by_widget.append(feature_list)

def Blacklist():
    blacklist = {}
    blacklist[key] = 1

    return blacklist

def get_all_text_path(prefix,screen_file):
    file_List = []
    get_all_file(prefix+screen_file,file_List)
    text_path_list_uix = get_image_file(file_List,".uix")
    text_path_list_xml = get_image_file(file_List,".xml")
    return text_path_list_uix + text_path_list_xml

def get_all_features(ui_path_list, blacklist, prefix,screen_file,id):
    # extract_ori_feature_revise / extract_ori_feature_Semi
    feature_map = {}
    key_list = []
    index = 0
    for XML_PATH in ui_path_list:
        if XML_PATH in blacklist:
            continue
        index +=1
        feature_list_by_widget = []
        # for Migration
        extract_ori_feature_revise(XML_PATH, feature_list_by_widget,id)
        # for SemiFinder
        # extract_ori_feature_Semi(XML_PATH,feature_list_by_widget)
        root = os.path.join(prefix, screen_file)
        for widget_index in range(len(feature_list_by_widget)):
            widget_information = feature_list_by_widget[widget_index]
            widget_xpath = widget_information[0]
            text_feature = widget_information[1]
            other_feature = widget_information[2:]
            prefix_shopping = os.path.join(root, "shopping/")
            prefix_news = os.path.join(root, "news/")
            key = ''
            if prefix_shopping in XML_PATH:
                key = XML_PATH.replace(prefix_shopping, "")+":"+widget_xpath
            if prefix_news in XML_PATH:
                key = XML_PATH.replace(prefix_news, "")+":"+widget_xpath
            # for Craftdroid
            prefix_a1 = os.path.join(root,"a1/")
            prefix_a2 = os.path.join(root,"a2/")
            prefix_a3 = os.path.join(root,"a3/")
            prefix_a4 = os.path.join(root,"a4/")
            prefix_a5 = os.path.join(root,"a5/")
            if prefix_a1 in XML_PATH:
                key = XML_PATH.replace(prefix_a1,"")+":"+widget_xpath
            if prefix_a2 in XML_PATH:
                key = XML_PATH.replace(prefix_a2,"")+":"+widget_xpath
            if prefix_a3 in XML_PATH:
                key = XML_PATH.replace(prefix_a3,"")+":"+widget_xpath
            if prefix_a4 in XML_PATH:
                key = XML_PATH.replace(prefix_a4,"")+":"+widget_xpath
            if prefix_a5 in XML_PATH:
                key = XML_PATH.replace(prefix_a5,"")+":"+widget_xpath
            # for Craftdroid (one source one tgt)
            if 'a1/' in XML_PATH:
                key = XML_PATH.split("a1/")[1]+":"+widget_xpath
            if 'a2/' in XML_PATH:
                key = XML_PATH.split("a2/")[1] + ":" + widget_xpath
            if 'a3/' in XML_PATH:
                key = XML_PATH.split("a3/")[1]+":"+widget_xpath
            if 'a4/' in XML_PATH:
                key = XML_PATH.split("a4/")[1] + ":" + widget_xpath
            if 'a5/' in XML_PATH:
                key = XML_PATH.split("a5/")[1] + ":" + widget_xpath
            key = key.strip()   # delete /n /t
            value = [text_feature, other_feature]
            feature_map[key] = value
            key_list.append(key)
    return feature_map


def mask_text_feature(id,widget_text,widget_content,widget_resource,ocr_text):
    widget_text_output = ""
    if id == -1:
        # for all
        if widget_text == "" and widget_content == "":
            widget_text_output = widget_resource+"\n"+ocr_text # (add_ocr_text)
        else:
            widget_text_output = widget_text +"\n"+ widget_content +"\n"+ widget_resource
    if id == 0:
        # w/o widget_text
        widget_text_output = widget_content +"\n"+ widget_resource+"\n"+ocr_text
    if id == 1:
        # w/o ocr
        widget_text_output = widget_text +"\n"+ widget_content +"\n"+ widget_resource
    if id == 2:
        # w/o content
        if widget_text == "":
            widget_text_output = widget_resource+"\n"+ocr_text
        else:
            widget_text_output = widget_text +"\n"+ widget_resource
    if id == 3:
        # w/o id
        if widget_text == "":
            widget_text_output = widget_content +"\n"+ocr_text
        else:
            widget_text_output = widget_text +"\n"+ widget_content
    return widget_text_output

def CropImage4File(XML_PATH,widget_bounded):
    """
    cropped image
    input: originial image + position information
    output: cropped image + related OCR
    :return:
    """
    if '.uix' in XML_PATH:
        image_path = XML_PATH.replace(".uix",".png")
    else:
        image_path = XML_PATH.replace(".xml",".png")
    # for droidbot dump
    if 'xml_' in image_path:
        image_path = image_path.replace("xml_",'screen_')
    suffix = image_path.split("/")[-1]
    image_root = image_path.replace(suffix,"")
    if os.path.exists(image_path)==False: # cannot find a png in the fruiter benchmark
        text = ""
        return text
    image = cv2.imread(image_path)

    # plt.imshow(image)
    # plt.show()

    x_start = widget_bounded[0]
    y_start = widget_bounded[1]
    x_end = widget_bounded[2]
    y_end = widget_bounded[3]

    # prevend x_end > image_x; prevent y_end > image_y
    if x_end > image.shape[1]:
        x_end = image.shape[1]
    if y_end > image.shape[0]:
        y_end = image.shape[0]

    if x_end> x_start and y_end > y_start:
        cropImg = image[y_start:y_end,x_start:x_end,:]  # y x in image are reversed from x y in bounded
        # test cropImg
        # plt.imshow(cropImg)
        # plt.show()

        text = pt.image_to_string(cropImg)
        # delete unused /n
        text.replace("\n\n","\n").replace("\f","")
    else:
        text = ""

    return text

# text_based features and other_features are all settled into the texts
def extract_ori_feature_save_text(XML_PATH, feature_list_by_widget,id):
    widget_class_dict = dict()
    tree = etree.parse(XML_PATH)
    root = tree.getroot()
    search_widgets = get_nodelist(root)
    for widget_list in search_widgets:
        feature_list = []
        ori_class = widget_list.get("class")
        if ori_class:
            widget_class = ori_class.replace("android.widget.", "")
        else:
            widget_class = ""
        widget_text = widget_list.get("text")
        widget_bounded = widget_list.get("bounds")
        former_coordination = widget_bounded.split("][")[0]
        latter_coordination = widget_bounded.split("][")[1]
        former_x_bound = int(former_coordination.split(",")[0][1:])
        former_y_bound = int(former_coordination.split(",")[1])
        latter_x_bound = int(latter_coordination.split(",")[0])
        latter_y_bound = int(latter_coordination.split(",")[1][:-1])
        if former_x_bound < 0:
            former_x_bound = 0
        if latter_x_bound < 0:
            latter_x_bound = 0
        widget_bounded = [former_x_bound, former_y_bound,
                          latter_x_bound, latter_y_bound]
        widget_square = (latter_x_bound - former_x_bound) * \
            (latter_y_bound - former_y_bound)
        widget_content = widget_list.get("content-desc")
        widget_resource = widget_list.get("resource-id")
        if widget_resource != "":
            if ":id/" in widget_resource:
                widget_resource = widget_resource.split(":id/")[1]
            # if widget_resource =="com.mobilesrepublic.appy:id/search_bar_layout":
            #     print("here")
        widget_text_size = 0   # initial
        if widget_text != "":
            text_num = len(widget_text.split("\n"))
            # need to reduce the widget_text_size
            widget_text_size = cmath.sqrt(widget_square) / text_num
        widget_clickable = "non-clickable"
        widget_checkable = "non-checkable"
        widget_enable = "non-enable"
        widget_focusable = "non-focusable"
        widget_scrollable = "non-scrollable"
        widget_long_clickable = "non-long_clickable"
        widget_password = "non-password"
        widget_selected = "non-selected"
        if widget_list.get("clickable") == 'true':
            widget_clickable = "clickable"
        if widget_list.get("checkable") == 'true':
            widget_checkable = "checkable"
        if widget_list.get("enable") == 'true':
            widget_enable = "enable"
        if widget_list.get("focusable") == 'true':
            widget_focusable = "focusable"
        if widget_list.get("scrollable") == 'true':
            widget_scrollable = "scrollable"
        if widget_list.get("long-clickable") == 'true':
            widget_long_clickable = "clickable"
        if widget_list.get("password") == 'true':
            widget_password = "password"
        if widget_list.get("selected") == 'true':
            widget_selected = "selected"
        # identify sibling
        sibling_text = ""
        if len(widget_list.getparent()) != 1:
            for idx in range(len(widget_list.getparent())-1):
                sibling = widget_list.getparent()[idx]
                widget_candidate = widget_list.getparent()[idx+1]
                if widget_candidate == widget_list:
                    sibling_text = sibling.get("text")
                    break

                # for relative postion
        x_start = widget_bounded[0]
        y_start = widget_bounded[1]
        x_end = widget_bounded[2]
        y_end = widget_bounded[3]
        widget_center_x = (x_start + x_end) / 2
        widget_center_y = (y_start + y_end) / 2
        image_center_x = (0 + 1440) / 2
        image_center_y = (0 + 2960) / 2
        left_relative_position = "non-left"
        top_relative_position = "non-top"
        if widget_center_x < image_center_x:
            left_relative_position = "left"
        if widget_center_y < image_center_y:
            top_relative_position = "top"


        widget_all_text = widget_text +"\n"+ widget_content +"\n"+ widget_resource
        # identify OCR
        # only for server
        ocr_text = CropImage4File(XML_PATH, widget_bounded)
        widget_all_text_add = ""
        if widget_text == "":
            widget_all_text_add = widget_content +"\n"+ widget_resource+"\n"+ocr_text # (add_ocr_text)
        else:
            widget_all_text_add = widget_all_text
        widget_all_text_add = mask_text_feature(id,widget_text,widget_content,widget_resource,ocr_text)
        widget_all_text_add = widget_all_text_add.lower().replace("_"," ").replace("-"," ") # for resource_id

        layout_text = sibling_text + "\n" + left_relative_position + "\n" + top_relative_position
        type_text = widget_class + "\n" + widget_checkable + "\n" + widget_long_clickable + "\n" + widget_clickable +"\n" +widget_focusable + "\n"+ widget_password 

        widget_other_feature_text = layout_text + "\n" + type_text
        widget_other_feature_text = widget_other_feature_text.lower().replace("_"," ")

        # get the xpath from the node
        xpath = get_xpath_from_node_lxml(widget_list)

        feature_list.append(xpath)
        # feature_list.append(widget_all_text) # for without text
        feature_list.append(widget_all_text_add) # feature-0
        feature_list.append(widget_other_feature_text)


        feature_list_by_widget.append(feature_list)


# save_other_feature as text feature
def get_all_features_save_text(ui_path_list, blacklist, prefix,screen_file,id):
    # extract_ori_feature_revise / extract_ori_feature_Semi
    feature_map = {}
    key_list = []
    index = 0
    for XML_PATH in ui_path_list:
        if XML_PATH in blacklist:
            continue
        print(index)
        index +=1
        feature_list_by_widget = []
        # for Migration (other_feature->text)
        extract_ori_feature_save_text(XML_PATH, feature_list_by_widget,id)
        root = os.path.join(prefix, screen_file)
        for widget_index in range(len(feature_list_by_widget)):
            widget_information = feature_list_by_widget[widget_index]
            widget_xpath = widget_information[0]
            text_feature = widget_information[1]
            other_feature = widget_information[2]
            prefix_shopping = os.path.join(root, "shopping/")
            prefix_news = os.path.join(root, "news/")
            if prefix_shopping in XML_PATH:
                key = XML_PATH.replace(prefix_shopping, "")+":"+widget_xpath
            if prefix_news in XML_PATH:
                key = XML_PATH.replace(prefix_news, "")+":"+widget_xpath
            # for Craftdroid
            prefix_a1 = os.path.join(root,"a1/")
            prefix_a2 = os.path.join(root,"a2/")
            prefix_a3 = os.path.join(root,"a3/")
            prefix_a4 = os.path.join(root,"a4/")
            prefix_a5 = os.path.join(root,"a5/")
            if prefix_a1 in XML_PATH:
                key = XML_PATH.replace(prefix_a1,"")+":"+widget_xpath
            if prefix_a2 in XML_PATH:
                key = XML_PATH.replace(prefix_a2,"")+":"+widget_xpath
            if prefix_a3 in XML_PATH:
                key = XML_PATH.replace(prefix_a3,"")+":"+widget_xpath
            if prefix_a4 in XML_PATH:
                key = XML_PATH.replace(prefix_a4,"")+":"+widget_xpath
            if prefix_a5 in XML_PATH:
                key = XML_PATH.replace(prefix_a5,"")+":"+widget_xpath
            key = key.strip()   # delete /n /t
            value = [text_feature, other_feature]
            feature_map[key] = value
            key_list.append(key)
    return feature_map

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

def get_pre_re_f1_group(df_predict,predict_map_path):
    tp_num = 0
    fp_num = 0
    fn_num = 0
    groups = df_predict.groupby(['widget1'])
    df_predict['evaluate'] = np.nan
    for group in groups:
        df_group = group[1]
        df_groundtruth = df_group.query("label==1")
        if len(df_groundtruth) == 0:
            df_predict_true = df_group.query("predict_label==1")
            if len(df_predict_true) == 1:
                row_index = get_row_index_from_df(df_predict_true)
                df_predict.loc[row_index[0],'evaluate'] = 'fp'
                fp_num += 1
        else:
            gt_predict_result = df_groundtruth.iloc[0].at['predict_label']
            row_index = get_row_index_from_df(df_groundtruth)
            if gt_predict_result == 1:
                tp_num += 1
                df_predict.loc[row_index[0], 'evaluate'] = 'tp'
            else:
                df_predict_true =df_group.query("predict_label==1")
                if len(df_predict_true) == 0:
                    fn_num += 1
                    row_index = get_row_index_from_df(df_groundtruth)
                    df_predict.loc[row_index[0], 'evaluate'] = 'fn'
                else:
                    fp_num += 1
                    row_index = get_row_index_from_df(df_predict_true)
                    df_predict.loc[row_index[0], 'evaluate'] = 'fp'
    precision = tp_num / (tp_num+fp_num)
    recall = tp_num / (tp_num+fn_num)
    f1 = 2*precision*recall/(precision+recall)
    print(predict_map_path)
    print("precision group",precision)
    print("recall group",recall)
    print("f1 group",f1)
    df_predict.to_csv(predict_map_path)

    return tp_num, fp_num, fn_num
