# coding=utf-8



import cv2
import os



def calculate(image1, image2):
    # gray images
    # calculate the hisgram similarity
    hist1 = cv2.calcHist([image1], [0], None, [256], [0.0, 255.0])
    hist2 = cv2.calcHist([image2], [0], None, [256], [0.0, 255.0])
    degree = 0
    for i in range(len(hist1)):
        if hist1[i] != hist2[i]:
            degree = degree + (1 - abs(hist1[i] - hist2[i]) / max(hist1[i], hist2[i]))
        else:
            degree = degree + 1
    degree = degree / len(hist1)
    return degree


def classify_hist_with_split(image1, image2, size,threadhold):
    # RGB the similarity of each channel
    # resize figures in the RGB three channels and calculate the similarity
    image1 = cv2.resize(image1, size)
    image2 = cv2.resize(image2, size)
    sub_image1 = cv2.split(image1)
    sub_image2 = cv2.split(image2)
    sub_data = 0
    for im1, im2 in zip(sub_image1, sub_image2):
        sub_data += calculate(im1, im2)
    sub_data = sub_data / 3
    if sub_data >= threadhold:
        return 1
    else:
        return 0



def sort_same_state_group(same_state_group):
    same_state_dict = {}
    for index in range(len(same_state_group)):
        same_states = same_state_group[index]
        for state_id in same_states:
            same_state_dict[state_id] = same_states
    return same_state_dict




def get_all_file(path,file_List):
    """
    get all path from current root
    """

    dir_List = [] # save all dir path
    for file in os.listdir(path):
        whole_path = os.path.join(path, file)
        if os.path.isdir(whole_path):
            dir_List.append(whole_path)
        if os.path.isfile(whole_path) and '.DS_Store' not in whole_path:
            file_List.append(whole_path)

    for dir in dir_List:
        get_all_file(dir, file_List)

def get_image_file(file_List,suffix):
    """
    get ".png" from all path to get the image path
    :param file_List:
    :param suffix:
    :return:
    """

    image_list = []
    for file in file_List:
        if  file.endswith(suffix):
            image_list.append(file)

    return image_list





