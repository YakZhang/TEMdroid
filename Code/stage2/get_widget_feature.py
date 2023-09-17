

import torch
from transformers import BertModel, BertTokenizer, BertConfig
from lxml import etree
import cv2
import pytesseract as pt
from PIL import Image
import matplotlib.pyplot as plt
from stage2.common import get_all_text_path
import pandas as pd
import json
import random
import cmath
import csv
import os
import json








def check_use_class(class_name):
    if 'android.widget.Button' in class_name or 'android.widget.EditText' in class_name or 'LinearLayoutCompat' in class_name or 'android.widget.TextView' in class_name or 'MultiAutoCompleteTextView' in class_name\
            or 'android.widget.ImageButton' in class_name or 'android.view.View' in class_name:
        return True
    else:
        return False

def extract_ori_feature_revise(XML_PATH, feature_list_by_widget,id):
    widget_class_dict = dict()
    tree = etree.parse(XML_PATH)
    root = tree.getroot()
    search_widgets = get_nodelist(root)
    for widget_list in search_widgets:
        feature_list = []
        ori_class = widget_list.get("class")
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
        widget_text_size = 0   
        if widget_text != "":
            text_num = len(widget_text.split("\n"))
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
        sibling_num = 0
        if len(widget_list.getparent()) != 0:
            sibling_num = len(widget_list.getparent())








        widget_all_text = widget_text +" "+ widget_content +" "+ widget_resource
        # widget_all_text = widget_all_text.lower().replace("_"," ").replace("-"," ")
        # #identify OCR
        # # only for server
        # ocr_text = CropImage4File(XML_PATH,widget_bounded)
        # widget_all_text_add = ""
        # if widget_text == "" or widget_content == "":
        #     widget_all_text_add =  widget_resource+"\n"+ocr_text # (add_ocr_text)
        # else:
        #     widget_all_text_add = widget_all_text
        # widget_all_text_add = mask_text_feature(id,widget_text,widget_content,widget_resource,ocr_text)
        # widget_all_text_add = widget_all_text_add.lower().replace("_"," ").replace("-"," ") # for resource_id
        # widget_all_text_new = ""
        # if widget_text == "":
        #     widget_all_text_new = widget_content+"\n"+widget_resource+"\n"+ocr_text # (add_ocr_text)
        # else:
        #     widget_all_text_new = widget_text +"\n"+ widget_content +"\n"+ widget_resource



        # get the xpath from the node
        xpath = get_xpath_from_node_lxml(widget_list)


        test_xpath = '/hierarchy/node[@class=\'android.widget.FrameLayout\'][1]/node[@class=\'android.widget.LinearLayout\'][1]/node[@class=\'android.widget.FrameLayout\'][1]/node[@class=\'android.widget.LinearLayout\'][1]/node[@class=\'android.widget.FrameLayout\'][1]/node[@class=\'android.view.ViewGroup\'][1]/node[@class=\'android.support.v4.widget.DrawerLayout\'][1]/node[@class=\'android.widget.LinearLayout\'][1]/node[@class=\'android.widget.FrameLayout\'][1]/node[@class=\'android.webkit.WebView\'][1]/node[@class=\'android.webkit.WebView\'][1]/node[@class=\'android.view.View\'][2]/node[@class=\'android.view.View\'][1]'
        if xpath == test_xpath:
            print("here")


        if widget_class in widget_class_dict:
            widget_class_index =widget_class_dict[widget_class]
        else:
            widget_class_index = len(widget_class_dict) 
            widget_class_dict[widget_class] = widget_class_index


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
        feature_list.append(widget_all_text) # 
        feature_list.append(widget_resource)
        feature_list.append(widget_content)
        feature_list.append(widget_text)



        feature_list_by_widget.append(feature_list)


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
        widget_text_size = 0   # initial
        if widget_text != "":
            text_num = len(widget_text.split("\n"))
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

        sibling_text = ""
        if len(widget_list.getparent()) != 1:
            for idx in range(len(widget_list.getparent())-1):
                sibling = widget_list.getparent()[idx]
                widget_candidate = widget_list.getparent()[idx+1]
                if widget_candidate == widget_list:
                    sibling_text = sibling.get("text")
                    break

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
        ocr_text = CropImage4File(XML_PATH,widget_bounded)
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


        xpath = get_xpath_from_node_lxml(widget_list)

        feature_list.append(xpath)              # 1207 add feature-0 xpath
        # feature_list.append(widget_all_text) # for without text
        feature_list.append(widget_all_text_add) # feature-0
        feature_list.append(widget_other_feature_text)


        feature_list_by_widget.append(feature_list)













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


        


# only for server
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
    if os.path.exists(image_path)==False:# cannot find a png in the fruiter benchmark
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

def extract_ori_feature_Semi(XML_PATH, feature_list_by_widget):
    tree = etree.parse(XML_PATH)
    root = tree.getroot()
    search_widgets = get_nodelist(root)
    for widget_list in search_widgets:
        feature_list = []
        widget_class = widget_list.get("class")
        widget_text = widget_list.get("text")
        widget_bounded = widget_list.get("bounds")
        former_coordination = widget_bounded.split("][")[0]
        latter_coordination = widget_bounded.split("][")[1]
        former_x_bound = int(former_coordination.split(",")[0][1:])
        former_y_bound = int(former_coordination.split(",")[1])
        latter_x_bound = int(latter_coordination.split(",")[0])
        latter_y_bound = int(latter_coordination.split(",")[1][:-1])
        widget_bounded = [former_x_bound, former_y_bound,
                          latter_x_bound, latter_y_bound]
        widget_square = (latter_x_bound - former_x_bound) * \
            (latter_y_bound - former_y_bound)
        widget_content = widget_list.get("content-desc")
        widget_resource = widget_list.get("resource-id")
        if widget_resource != "":
            if ":id/" in widget_resource:
                widget_resource = widget_resource.split(":id/")[1]

        widget_text_size = 0   # initial
        if widget_text != "":
            text_num = len(widget_text.split("\n"))
            # need to reduce the widget_text_size
            widget_text_size = cmath.sqrt(widget_square) / text_num
        widget_clickable = 0
        widget_fillable = 0
        if widget_list.get("clickable") == 'true' or widget_list.get("long-clickable") == 'true':
            widget_clickable = 1
        if widget_list.get("enable") == 'true' or widget_list.get("password") == 'true' :
            widget_fillable = 1
        # identify sibling
        sibling_num = 0
        if len(widget_list.getparent()) != 0:
            sibling_num = len(widget_list.getparent())


        # widget_all_text = widget_text +"\n"+ widget_content +"\n"+ widget_resource+"\n"+ocr_text
        widget_all_text = widget_text + "\n" + widget_content + "\n" + widget_resource

        # get class
        widget_EditText = 0
        widget_View = 0
        widget_ImageButton = 0
        widget_Button = 0
        widget_TextView = 0
        if widget_class == 'android.widget.EditText':
            widget_EditText = 1
        if widget_class == 'android.view.View':
            widget_View = 1
        if widget_class == 'android.widget.ImageButton':
            widget_ImageButton = 1
        if widget_class == 'android.widget.Button':
            widget_Button = 1
        if widget_class == 'android.widget.TextView':
            widget_TextView = 1

        # get the xpath from the node
        xpath = get_xpath_from_node_lxml(widget_list)
        # print(xpath)
        # for test
        # if xpath == '/hierarchy/node[@class=\'android.widget.FrameLayout\']/node[@class=\'android.widget.LinearLayout\']/node[@class=\'android.widget.FrameLayout\']/node[@class=\'android.widget.LinearLayout\']/node[@class=\'android.widget.LinearLayout\']/node[@class=\'androidx.recyclerview.widget.RecyclerView\']/node[@class=\'androidx.recyclerview.widget.RecyclerView\'][2]/node[@class=\'android.view.ViewGroup\'][1]'
        #     print("true")

        feature_list.append(xpath)

        feature_list.append(widget_all_text)    # feature-0



        feature_list_by_widget.append(feature_list)


def get_nodelist(root):
    nodelist = []

    def walk_root(node):
        nodelist.append(node)
        for child in node:
            walk_root(child)
    walk_root(root)
    return nodelist[1:]  # Delete root





def get_all_features(ui_path_list, blacklist, prefix,screen_file,id):
    # extract_ori_feature_revise / extract_ori_feature_Semi
    feature_map = {}
    key_list = []
    index = 0
    for XML_PATH in ui_path_list:
        if XML_PATH in blacklist:
            continue
        if index == 110:
            print("here")
        # print(index)
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
            # for Fruiter
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
            #key = os.path.join("/",key)
            key = key.strip()   # delete /n /t
            # todo: value can be added in widget_xpath(partly)
            value = [text_feature, other_feature]
            feature_map[key] = value
            key_list.append(key)
    # for test feature_map_key
    # df = pd.DataFrame(key_list)
    # df.to_csv("feature_map_key_1212.csv")
    return feature_map

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
            # for Fruiter
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
            #key = os.path.join("/",key)
            key = key.strip()   # delete /n /t
            # todo: value can be added in widget_xpath(partly)
            value = [text_feature, other_feature]
            feature_map[key] = value
            key_list.append(key)
    # for test feature_map_key
    # df = pd.DataFrame(key_list)
    # df.to_csv("feature_map_key_1212.csv")
    return feature_map






def get_feature_embedding_test(prefix):
    # replace the get_feature_embeddding
    # feature extraction
    ui_path_list = get_all_text_path(prefix)
    blacklist = Blacklist()
    feature_map = get_all_features(ui_path_list, blacklist, prefix)
    feature_vector_map = {}
    feature_vector_key = []
    """
    initial bert
    """
    device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")
    model_name = 'bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(model_name)
    # note: Use BertModel (have pretrained) not BertforPreTraining
    model = BertModel.from_pretrained(model_name)
    model = model.to(device)
    for key in feature_map:
        feature = feature_map[key]
        text_feature = feature[0].replace("\n", "").replace("_", " ")

        has_text_feature = 1
        if text_feature == "":
            has_text_feature = 0
            text_feature = "[UNK]"

        text_feature_vector = get_output_from_BertModel(
            text_feature, tokenizer=tokenizer, model=model, device=device
        )   # tensor size = 768

        has_sibling_feature = feature[1][-1]
        has_sibling_feature = int(has_sibling_feature > 1)
        has_other_feature = int(1 in feature[1][:-1])
        other_feature = feature[1][:-1]

        other_feature.append(has_sibling_feature)
        other_feature.append(has_text_feature)
        other_feature.append(has_other_feature)  # other feature number = 11
        other_feature_vector = torch.tensor(other_feature).float().to(device)
        feature_vector = concate_feature(
            text_feature_vector, other_feature_vector).detach().cpu().numpy()    # numpy size = 768+16*10
        #new_key = key.replace("shopping/", "").replace("news/", "")
        feature_vector_map[key] = feature_vector
        feature_vector_key.append(key)
        

    return feature_vector_map, feature_vector_key


def get_true_label_map(ground_truth_map_path):

    true_label_map = pd.DataFrame(columns=['widget1', 'widget2'], index=[])
    key_list = []
    # only for server
    ground_truth_map_path = ground_truth_map_path.replace(
        "../../", "/raid/Migration/")
    df = pd.read_csv(ground_truth_map_path)
    if 'add-xpath_x' in df: 
        # for Fruiter
        xpath_x = df['add-xpath_x'].tolist()
        xpath_y = df['add-xpath_y'].tolist()
        screen_x_files = df['state-name_x'].tolist()
        screen_y_files = df['state-name_y'].tolist()
    elif 'add_xpath_x' in df:
        # for Fruiter (groundtruth false)
        xpath_x = df['add_xpath_x'].tolist()
        xpath_y = df['add_xpath_y'].tolist()
        screen_x_files = df['state_name_x'].tolist()
        screen_y_files = df['state_name_y'].tolist()
    else:
        # for Craftdroid
        xpath_x = df['src_add_xpath'].tolist()
        xpath_y = df['tgt_add_xpath'].tolist()
        screen_x_files = df['src_state'].tolist()
        screen_y_files = df['tgt_state'].tolist()
    for index in range(len(df)):
        # for one x and y in one row
        screen_x = screen_x_files[index].strip().split("\t")[-1] # the last screen can be match with xpath
        screen_y = screen_y_files[index].strip().split("\t")[-1]
        if ' ' in screen_x: # space 
            screen_x = screen_x.split(" ")[-1] 
        if ' ' in screen_y:
            screen_y = screen_y.split(" ")[-1]
        widget1 = screen_x+":"+xpath_x[index]  # can change to entity1
        widget2 = screen_y+":"+xpath_y[index]    # can change to entity2
        key_list.append(widget1)
        # need to use hd5 to represent
        df_line = pd.DataFrame({
            'widget1': widget1,
            'widget2': widget2,
        }, index=[1])
        true_label_map = true_label_map.append(df_line, ignore_index=True)
    # for test
    # df = pd.DataFrame(key_list)
    # df.to_csv("true_feature_map_1212.csv")

    return true_label_map




def dump_feature_labels(feature_save_path, feature_vector_map, feature_vector_key, true_label_map):
    """
    input: two features + related labels
    """
    # get true_feature_label
    true_data_list = []
    # the keys in true_label_map is the subset of the keys in feature_vector_map
    for index, row in true_label_map.iterrows():
        widget1 = row['widget1']
        widget2 = row['widget2']
        feature1 = feature_vector_map[widget1].tolist()
        feature2 = feature_vector_map[widget2].tolist()
        file1 = widget1.split(":")[0]
        file2 = widget2.split(":")[0]
        category = 0
        if file1 == file2:
            category = 1
        data = {
            "widget1": widget1,
            "widget2": widget2,
            "feature1": feature1,
            "feature2": feature2,
            "label": 1,
            "category": category
        }
        true_data_list.append(json.dumps(data) + '\n')

    true_data_path = feature_save_path + \
        "true_feature_label_"+".jsonl"
    with open(true_data_path, 'w') as jsonl:
        jsonl.writelines(true_data_list)
        print("number of true_data_list", len(true_data_list))
        print("finish dump true_data_list")


    index = 0
    false_data_list = []
    # key_list = list(key_set)    # set to list
    while(index < len(true_data_list)):
        random_index1 = random.randint(0, len(feature_vector_key)-1)
        widget1 = feature_vector_key[random_index1]
        random_index2 = random.randint(0, len(feature_vector_key)-1)
        widget2 = feature_vector_key[random_index2]
        # find whether widget1 and widget2 belong to the same line in the true_label_map
        test_row_index = true_label_map[(true_label_map['widget1'].isin(
            [widget1])) & true_label_map['widget2'].isin([widget2])].index.tolist()
        if len(test_row_index) == 0:  # not row in the true_label_map
            feature1 = feature_vector_map[widget1].tolist()
            feature2 = feature_vector_map[widget2].tolist()
            index = index + 1
            file1 = widget1.split(":")[0]
            file2 = widget2.split(":")[0]
            category = 0
            if file1 == file2:
                category = 1
            data = {
                "widget1": widget1,
                "widget2": widget2,
                "feature1": feature1,
                "feature2": feature2,
                "label": 0,
                "category": category
            }
            false_data_list.append(json.dumps(data)+"\n")

    false_data_path = feature_save_path + \
        "false_feature_label_"+".jsonl"
    with open(false_data_path, 'w') as jsonl:
        jsonl.writelines(false_data_list)
        print("number of false_data_list", len(false_data_list))
        print("finish dump false_data_list")
    print("true_data_path", true_data_path)
    print("false_data_path", false_data_path)
    return true_data_path, false_data_path

def main():
    ground_truth_true_path = ''
    true_label_map = get_true_label_map(ground_truth_true_path)
    test_case_path = ''  
    feature_vector_map, feature_vector_list = get_feature_embedding_test(test_case_path)
    feature_save_path =''
    dump_feature_labels(feature_save_path,feature_vector_map,feature_vector_list, true_label_map)


if __name__ == "__main__":
    main()
