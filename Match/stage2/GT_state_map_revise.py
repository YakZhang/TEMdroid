
import pandas as pd
import numpy as np
from stage2.get_same_state import get_all_file, get_image_file
from lxml import etree


def load_csv(csv_path):
    df = pd.read_csv(csv_path)
    id_list = df['id'].tolist()  # str and float(nan)
    xpath_list = df['xpath'].tolist()
    canonical_list = df['canonical'].tolist()
    return id_list, xpath_list,canonical_list


def map_event_state(state_path, id_list, xpath_list,canonical_list):
    df_map = pd.DataFrame(columns=['add-id', 'add-xpath', 'state-name', 'content-desc', 'text', 'canonical'], index=[])

    file_List = []
    get_all_file(state_path, file_List)
    state_list = get_image_file(file_List, suffix='.uix')  # a list of uix files
    prefix = 'a.b.path'

    # given the id, find the xpath and others
    for index in range(len(id_list)):
        id = id_list[index]
        canonical = canonical_list[index]
        #state_index = 0  # the number of state-name
        if isinstance(id, float):  # {float} nan
            xpath = xpath_list[index]
            df_map = get_information_from_xpath(canonical, state_list, prefix, df_map,xpath)
        else:
            df_map = get_information_from_id(state_list, prefix, canonical, df_map,id)

    return df_map

def get_information_from_id(state_list,prefix,canonical,df_map,id):
    for state_path in state_list:
        file_name = state_path.replace(prefix, "")
        tree = etree.parse(state_path)
        root = tree.getroot()
        node = get_node_by_id(root, id)
        if node is not None:
            # state_index = state_index + 1
            # more than one state names means we need to add more rows
            text = np.nan
            if node.get("text") != "":
                text = node.get("text")
            content_desc = np.nan
            if node.get("content-desc") != "":
                content_desc = node.get("content-desc")
            df_line = pd.DataFrame({
                'add-id': id,
                'add-xpath': get_xpath_from_node_lxml(node),
                'state-name': file_name,
                'content-desc': content_desc,
                'text': text,
                'canonical': canonical
            }, index=[1]
            )
            df_map = df_map.append(df_line, ignore_index=True)
    return df_map

def get_information_from_xpath(canonical,state_list,prefix,df_map,xpath):

    xpath_fit = revise_xpath(xpath)
    for state_path in state_list:
        file_name = state_path.replace(prefix, "")
        tree = etree.parse(state_path)
        root = tree.getroot()
        findall = etree.XPath(xpath_fit)
        nodelist = findall(root)  # get the work
        if len(nodelist) != 0:
            # identify whether the there are more than one node has the same path
            if len(nodelist) > 1:
                print("more than one node has the same xpath in one .uix")
            add_id = np.nan
            if nodelist[0].get("resource-id") != "":
                add_id = nodelist[0].get("resource-id")
            text = np.nan
            if nodelist[0].get("text") != "":
                text = nodelist[0].get("text")
            content_desc = np.nan
            if nodelist[0].get("content-desc") != "":
                content_desc = nodelist[0].get("content-desc")
            df_line = pd.DataFrame({
                'add-id': add_id,
                'add-xpath': get_xpath_from_node_lxml(nodelist[0]),
                'state-name': file_name,
                'content-desc': content_desc,
                'text': text,
                'canonical': canonical
            }, index=[1]
            )
            df_map = df_map.append(df_line, ignore_index=True)
    return df_map


def get_node_by_id(root,id):
    if root.get("resource-id") == id:
        return root
    for child in root:
        node = get_node_by_id(child,id)
        if node is not None:
            return node
    return None

def get_nodelist_by_id(root,id,nodelist):
    if root.get("resource-id") == id:
        nodelist.append(root)
    for child in root:
        node = get_nodelist_by_id(child,id,nodelist)
        if node is not None:
            nodelist.append(node)

def get_xpath_from_node_lxml(node):
    xpath_elements = []
    xpath_elements.append(node.get("class"))
    xpath_elements_index = []
    while (node.tag != "hierarchy"):
        parentNode = node.getparent()
        xpath_elements.append(parentNode.get("class"))
        node_index = 1
        if (len(parentNode) > 1):  # has children
            xpath_name = node.get("class")
            for index in range(len(parentNode)):
                test_node = parentNode[index]  # child node
                if test_node.get("class")== xpath_name and test_node != node:
                    node_index = node_index + 1
                elif test_node == node:
                    break
                # if test_node == node:
                #     node_index = index + 1
                #     break
        xpath_elements_index.append(node_index)
        node = parentNode
    xpath_elements = xpath_elements[:-1] # the class of hierarchy is None
    xpath = "/hierarchy"
    xpath_elements.reverse()
    xpath_elements_index.reverse()
    for index in range(len(xpath_elements)):
        element_index = xpath_elements_index[index]
        element = xpath_elements[index]
        if element_index == 0:
            xpath_element = "/node[@class='" + element + "']"
        else:
            xpath_element = "/node[@class='" + element + "']" + "[" + str(element_index) + "]"
        xpath = xpath + xpath_element
    return xpath

def revise_xpath(ori_xpath):
    revised_xpath = ""
    """
    # test case
    ori_xpath = '//android.widget.ImageButton[@content-desc="Drawer Opened"]'
    ori_xpath = '/hierarchy/android.widget.FrameLayout/android.widget.LinearLayout/android.widget.FrameLayout/android.widget.LinearLayout/android.widget.FrameLayout/androidx.drawerlayout.widget.DrawerLayout/android.widget.LinearLayout[2]/android.widget.ListView/android.widget.RelativeLayout'
    """
    if '/hierarchy' in ori_xpath:
        element_list = ori_xpath.split("/")
        revised_xpath = '/hierarchy'
        for index in range(2,len(element_list)):
            element = element_list[index]
            if "[" in element:
                xpath_element = "/node[@class='"+element.split("[")[0]+"']"+"["+element.split("[")[1]
            else:
                xpath_element = "/node[@class='"+element+"']"
            revised_xpath = revised_xpath + xpath_element
    else:
        ori_xpath = ori_xpath.replace("//","")
        element_list = ori_xpath.split("[")
        revised_xpath = "//node[@class='"+element_list[0]+"']["+element_list[1]
    return revised_xpath


def write_csv(df_map, save_path):
    df_map.to_csv(save_path)


def get_nodelist(root):
    nodelist = []
    def walk_root(node):
        nodelist.append(node)
        for child in node:
            walk_root(child)
    walk_root(root)
    return nodelist[1:]


