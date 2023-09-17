
import random
import numpy as np
import jsonlines
import json
import pandas as pd

fold_1 = {'a11','a21','a31','a41','a51'}
fold_2 = {'a12','a22','a32','a42','a52'}
fold_3 = {'a13','a23','a33','a43','a53'}
fold_4 = {'a14','a24','a34','a44','a54'}
fold_5 = {'a15','a25','a35','a45','a55'}

fold_1_fruiter = {'abc','bbcnews','5miles','6pm'}
fold_2_fruiter = {'buzzfeed','cnn','aliexpress','ebay'}
fold_3_fruiter = {'fox','newsrepublic','etsy','geek'}
fold_4_fruiter = {'reuters','smartnews','googleshopping','groupon'}
fold_5_fruiter = {'theguardian','usatoday','home','wish'}

fold_1_atm = {'Expense1', 'Note1', 'Shop1'}
fold_2_atm = {'Expense2', 'Note2', 'Shop2'}
fold_3_atm = {'Expense3', 'Note3', 'Shop3'}
fold_4_atm = {'Expense4', 'Note4', 'Shop4'}

def split_fold_data(true_feature_label_path,false_feature_label_path):
    fold_1_data = []
    fold_2_data = []
    fold_3_data = []
    fold_4_data = []
    fold_5_data = []
    with open(true_feature_label_path, 'r+', encoding='utf8') as f:
        for item in jsonlines.Reader(f):
            target_widget_name = item['widget2'].split(":")[0].split("/")[0]
            # for craftdroid and Fruiter
            if target_widget_name in fold_1 or target_widget_name in fold_1_fruiter or target_widget_name in fold_1_atm :
                fold_1_data.append(json.dumps(item)+"\n")
            elif target_widget_name in fold_2 or target_widget_name in fold_2_fruiter or target_widget_name in fold_2_atm :
                fold_2_data.append(json.dumps(item)+"\n")
            elif target_widget_name in fold_3 or target_widget_name in fold_3_fruiter or target_widget_name in fold_3_atm :
                fold_3_data.append(json.dumps(item)+"\n")
            elif target_widget_name in fold_4 or target_widget_name in fold_4_fruiter or target_widget_name in fold_4_atm :
                fold_4_data.append(json.dumps(item)+"\n")
            elif target_widget_name in fold_5 or target_widget_name in fold_5_fruiter:
                fold_5_data.append(json.dumps(item)+"\n")
    with open(false_feature_label_path, 'r+', encoding='utf8') as f:
        for item in jsonlines.Reader(f):
            target_widget_name = item['widget2'].split(":")[0].split("/")[0]
            if target_widget_name in fold_1 or target_widget_name in fold_1_fruiter or target_widget_name in fold_1_atm:
                fold_1_data.append(json.dumps(item)+"\n")
            elif target_widget_name in fold_2 or target_widget_name in fold_2_fruiter or target_widget_name in fold_2_atm:
                fold_2_data.append(json.dumps(item)+"\n")
            elif target_widget_name in fold_3 or target_widget_name in fold_3_fruiter or target_widget_name in fold_3_atm:
                fold_3_data.append(json.dumps(item)+"\n")
            elif target_widget_name in fold_4 or target_widget_name in fold_4_fruiter or target_widget_name in fold_4_atm:
                fold_4_data.append(json.dumps(item)+"\n")
            elif target_widget_name in fold_5 or target_widget_name in fold_5_fruiter:
                fold_5_data.append(json.dumps(item)+"\n")

    return fold_1_data,fold_2_data,fold_3_data,fold_4_data,fold_5_data

def get_train_valid_test(fold_1_data,fold_2_data,fold_3_data,fold_4_data,fold_5_data,csv_save_path):
    train_data = fold_1_data + fold_2_data + fold_4_data
    valid_data = fold_3_data
    test_data =  fold_5_data
    train_data_path = csv_save_path+ \
        "train_data.jsonl"
    valid_data_path = csv_save_path+ \
        "valid_data.jsonl"
    test_data_path = csv_save_path+ \
        "test_data.jsonl"
    with open(train_data_path , 'w') as jsonl:
        jsonl.writelines(train_data)
        print("number of train_data", len(train_data))
        print("train_data_path:", train_data_path)
        print("finish dump train_data")
    with open(valid_data_path , 'w') as jsonl:
        jsonl.writelines(valid_data)
        print("number of valid_data", len(valid_data))
        print("valid_data_path:", valid_data_path)
        print("finish dump valid_data")
    with open(test_data_path , 'w') as jsonl:
        jsonl.writelines(test_data)
        print("number of test_data", len(test_data))
        print("test_data_path:", test_data_path)
        print("finish dump test_data")
    return train_data_path,valid_data_path,test_data_path

def get_cross_validation_data(fold_1_data,fold_2_data,fold_3_data,fold_4_data,fold_5_data,csv_save_path):
    train_1_data = fold_2_data+fold_3_data+fold_4_data+fold_5_data
    test_1_data = fold_1_data
    train_2_data = fold_1_data+fold_3_data+fold_4_data+fold_5_data
    test_2_data = fold_2_data
    train_3_data = fold_2_data+fold_1_data+fold_4_data+fold_5_data
    test_3_data = fold_3_data
    train_4_data = fold_1_data+fold_3_data+fold_2_data+fold_5_data
    test_4_data = fold_4_data
    train_5_data = fold_1_data+fold_3_data+fold_2_data+fold_4_data
    test_5_data = fold_5_data
    fold_train_data = [train_1_data,train_2_data,train_3_data,train_4_data,train_5_data]
    fold_test_data = [test_1_data, test_2_data, test_3_data,test_4_data, test_5_data]
    for index in range(len(fold_train_data)):
        train_i_data = fold_train_data[index]
        idx = str(index + 1)
        fold_i_data_path = csv_save_path + "train_" + idx + "_data_" + ".jsonl"
        with open(fold_i_data_path, 'w') as jsonl:
            jsonl.writelines(train_i_data)
            print("number of train_i_data", len(train_i_data))
            print("train_i_data_path:", fold_i_data_path)
            print("finish dump train_i_data")

    for index in range(len(fold_test_data)):
        idx = str(index + 1)
        test_i_data = fold_test_data[index]
        fold_i_data_path = csv_save_path + "test_" + idx + "_data_" +  ".jsonl"
        with open(fold_i_data_path, 'w') as jsonl:
            jsonl.writelines(test_i_data)
            print("number of test_i_data", len(test_i_data))
            print("test_i_data_path:", fold_i_data_path)
            print("finish dump test_i_data")
    train_1_data_path =csv_save_path + "train_1_data_" +  ".jsonl"

    fold_i_test_data_path_list = []
    for index in range(len(fold_test_data)):
        idx = str(index + 1)
        fold_i_test_data_path = csv_save_path + "cross_valid_test_" + idx + "_data_" +  ".jsonl"
        fold_i_test_data_path_list.append(fold_i_test_data_path)
    return fold_i_test_data_path_list

def get_cross_validation_data_train_valid(train_1_data, train_2_data, train_3_data, train_4_data, train_5_data, csv_save_path):
    random.seed(42)
    fold_i_train_data_path_list = []
    fold_i_valid_data_path_list = []
    all_data = [train_1_data, train_2_data, train_3_data, train_4_data, train_5_data]
    for index in range(len(all_data)):
        ori_train_i_data = all_data[index]
        indeces = list(range(len(ori_train_i_data)))
        random.shuffle(indeces)
        data_random = np.array(ori_train_i_data)[indeces]
        train_max_index = int(len(ori_train_i_data) * 0.75)
        train_i_data = data_random[:train_max_index]
        valid_i_data = data_random[train_max_index:]
        idx = str(index + 1)
        fold_i_train_data_path = csv_save_path + "cross_valid_train_" + idx + "_data_"  + ".jsonl"
        fold_i_train_data_path_list.append(fold_i_train_data_path)
        with open(fold_i_train_data_path, 'w') as jsonl:
            jsonl.writelines(train_i_data)
            print("number of train_i_data", len(train_i_data))
            print("train_i_data_path:", fold_i_train_data_path)
            print("finish dump train_i_data")
        fold_i_valid_data_path = csv_save_path + "cross_valid_valid_" + idx + "_data_" +  ".jsonl"
        fold_i_valid_data_path_list.append(fold_i_valid_data_path)
        with open(fold_i_valid_data_path, 'w') as jsonl:
            jsonl.writelines(valid_i_data)
            print("number of valid_i_data", len(valid_i_data))
            print("valid_i_data_path:", fold_i_valid_data_path)
            print("finish dump valid_i_data")
    return fold_i_train_data_path_list, fold_i_valid_data_path_list



def postprocess_train_valid_test(train_data_path,valid_data_path,test_data_path,csv_save_path):
    # for train_valid_test
    train_new_data = postprocess_data(train_data_path, fold_5,fold_5_fruiter)
    valid_new_data = postprocess_data(valid_data_path,fold_5,fold_5_fruiter)
    test_new_data = []
    train_data_name = train_data_path.split("/")[-1]
    ori_Month_Day_Hour_Minute = train_data_name.replace("train_data_","").replace(".jsonl","")
    with open(test_data_path,'r+',encoding='utf8') as f:
        for item in jsonlines.Reader(f):
            test_new_data.append(json.dumps(item)+"\n")
    Month_Day_Hour_Minute = ori_Month_Day_Hour_Minute
    train_new_data_path = csv_save_path+ \
        "train_data_"+Month_Day_Hour_Minute+".jsonl"
    valid_new_data_path = csv_save_path+ \
        "valid_data_"+Month_Day_Hour_Minute+".jsonl"
    test_new_data_path = csv_save_path+ \
        "test_data_"+Month_Day_Hour_Minute+".jsonl"
    with open(train_new_data_path , 'w') as jsonl:
        jsonl.writelines(train_new_data)
        print("number of train_data", len(train_new_data))
        print("train_data_path:", train_new_data_path)
        print("finish dump train_data")
    with open(valid_new_data_path , 'w') as jsonl:
        jsonl.writelines(valid_new_data)
        print("number of valid_data", len(valid_new_data))
        print("valid_data_path:", valid_new_data_path)
        print("finish dump valid_data")
    with open(test_new_data_path , 'w') as jsonl:
        jsonl.writelines(test_new_data)
        print("number of test_data", len(test_new_data))
        print("test_data_path:", test_new_data_path)
        print("finish dump test_data")


def postprocess_cross_validation(train_1_data_path,csv_save_path):
    train_fold_data_path = []
    test_fold_data_path = []
    train_1_data_name = train_1_data_path.split("/")[-1]
    ori_csv_save_path = train_1_data_path.replace(train_1_data_name,"")
    ori_Month_Day_Hour_Minute = train_1_data_name.replace("train_1_data_","").replace(".jsonl","")
    for index in range(5):
        idx = str(index + 1)
        train_fold_i_data_path = ori_csv_save_path + "train_" + idx + "_data_" + ori_Month_Day_Hour_Minute + ".jsonl"
        test_fold_i_data_path =  ori_csv_save_path + "test_" + idx + "_data_" + ori_Month_Day_Hour_Minute + ".jsonl"
        train_fold_data_path.append(train_fold_i_data_path)
        test_fold_data_path.append(test_fold_i_data_path)

    compare_fold_all_craftdroid = [fold_1,fold_2,fold_3,fold_4,fold_5]
    compare_fold_all_fruiter = [fold_1_fruiter,fold_2_fruiter,fold_3_fruiter,fold_4_fruiter,fold_5_fruiter]
    compare_fold_all_atm = [fold_1_atm, fold_2_atm,fold_3_atm,fold_4_atm]
    for index in range(len(train_fold_data_path)):
        train_data_path = train_fold_data_path[index]
        test_data_path = test_fold_data_path[index]
        compare_fold_craftdroid = compare_fold_all_craftdroid[index]
        compare_fold_fruiter = compare_fold_all_fruiter[index]
        if index !=4:
            compare_fold_atm = compare_fold_all_fruiter[index]
        else:
            compare_fold_atm = []
        train_i_data = postprocess_data(train_data_path, compare_fold_craftdroid,compare_fold_fruiter,compare_fold_atm)
        test_i_data = []
        with open(test_data_path, 'r+', encoding='utf8') as f:
            for item in jsonlines.Reader(f):
                test_i_data.append(json.dumps(item)+"\n")
        idx = str(index + 1)
        Month_Day_Hour_Minute = ori_Month_Day_Hour_Minute
        train_fold_i_data_path = csv_save_path + "train_" + idx + "_data_" + Month_Day_Hour_Minute + ".jsonl"
        test_fold_i_data_path =  csv_save_path + "test_" + idx + "_data_" + Month_Day_Hour_Minute + ".jsonl"
        with open(train_fold_i_data_path, 'w') as jsonl:
            jsonl.writelines(train_i_data)
            print("number of train_i_data", len(train_i_data))
            print("train_i_data_path:", train_fold_i_data_path)
            print("finish dump train_i_data")
        with open(test_fold_i_data_path, 'w') as jsonl:
            jsonl.writelines(test_i_data)
            print("number of test_i_data", len(test_i_data))
            print("test_i_data_path:", test_fold_i_data_path)
            print("finish dump test_i_data")



def postprocess_data(data_ori_path,compare_fold_craftdroid,compare_fold_fruiter, compare_fold_atm):
    data_new = []
    with open(data_ori_path,'r+',encoding='utf8') as f:
        for item in jsonlines.Reader(f):
            source_widget_name = item['widget1'].split(":")[0].split("/")[0]
            if source_widget_name in compare_fold_craftdroid or source_widget_name in compare_fold_fruiter or source_widget_name in compare_fold_atm:
                continue
            data_new.append(json.dumps(item)+"\n")
    return data_new


def get_train_test_for_Craftdroid_test_Fruiter(true_feature_label_path, false_feature_label_path,csv_save_path):
    # given Craftdroid ori true + false
    # delete Etsy / Geek/ Wish pair
    # split into two parts
    all_data = []
    with open(true_feature_label_path, 'r+', encoding='utf8') as f:
        for item in jsonlines.Reader(f):
            target_widget_name = item['widget2'].split(":")[0].split("/")[0]
            if target_widget_name == 'etsy' or target_widget_name == 'geek' or target_widget_name == 'wish':
                continue
            source_widget_name = item['widget1'].split(":")[0].split("/")[0]
            if source_widget_name == 'etsy' or source_widget_name == 'geek' or source_widget_name == 'wish':
                continue
            all_data.append(json.dumps(item)+"\n")
    with open(false_feature_label_path, 'r+', encoding='utf8') as f:
        for item in jsonlines.Reader(f):
            target_widget_name = item['widget2'].split(":")[0].split("/")[0]
            if target_widget_name == 'etsy' or target_widget_name == 'geek' or target_widget_name == 'wish':
                continue
            source_widget_name = item['widget1'].split(":")[0].split("/")[0]
            if source_widget_name == 'etsy' or source_widget_name == 'geek' or source_widget_name == 'wish':
                continue
            all_data.append(json.dumps(item)+"\n")
    random.seed(42)
    indeces = list(range(len(all_data)))
    random.shuffle(indeces)
    data_random = np.array(all_data)[indeces]
    data_random = data_random.tolist()
    train_max_index = int(len(all_data) * 0.8)
    train_data = data_random[:train_max_index]
    test_data = data_random[train_max_index:]
    train_data_path = csv_save_path + "train_data_" + ".jsonl"
    test_data_path = csv_save_path + "test_data_" + ".jsonl"
    with open(train_data_path, 'w') as jsonl:
        jsonl.writelines(train_data)
        print("number of train_i_data", len(train_data))
        print("train_i_data_path:", train_data_path)
    with open(test_data_path, 'w') as jsonl:
        jsonl.writelines(test_data)
        print("number of test_i_data", len(test_data))
        print("test_i_data_path:", test_data_path)
    return train_data, test_data


def get_train_valid_test_for_ori_groundtruth(groundtruth_true_ori_path,csv_save_path):
    # no use
    df_ori = pd.read_csv(groundtruth_true_ori_path)
    df_mask_1 = df_ori['app_name_y'] in fold_1_fruiter
    df_fold_1 = df_ori[df_mask_1]
    df_mask_2 = df_ori['app_name_y'] in fold_2_fruiter
    df_fold_2 = df_ori[df_mask_2]
    df_mask_3 = df_ori['app_name_y'] in fold_3_fruiter
    df_fold_3 = df_ori[df_mask_3]
    df_mask_4 = df_ori['app_name_y'] in fold_4_fruiter
    df_fold_4 = df_ori[df_mask_4]
    df_mask_5 = df_ori['app_name_y'] in fold_5_fruiter
    df_fold_5 = df_ori[df_mask_5]

    ori_train_data = df_fold_1 + df_fold_2 + df_fold_4
    ori_valid_data = df_fold_3
    ori_test_data = df_fold_5

    # delete some widgets in train and valid (app name in df_fold_5)
    df_train_mask = ori_train_data['app_name_x'] not in fold_5_fruiter
    df_train = ori_train_data[df_train_mask]
    df_valid_mask = ori_valid_data['app_name_x'] not in fold_5_fruiter
    df_valid = ori_valid_data[df_valid_mask]
    df_test = ori_test_data

    train_csv_path = csv_save_path + 'train_data_true_' + ".csv"
    valid_csv_path = csv_save_path + 'valid_data_true_'+ ".csv"
    test_csv_path = csv_save_path + 'test_data_true_' + ".csv"

    df_train.to_csv(train_csv_path)
    df_valid.to_csv(valid_csv_path)
    df_test.to_csv(test_csv_path)



def csv_to_json(groundtruth_true_ori_path, feature_save_path):
    # given groundtruth_true_ori_path, output true_feature_label
    df_ori = pd.read_csv(groundtruth_true_ori_path)
    true_data_ori = []
    for idx in range(len(df_ori)):
        state_name_x = df_ori.loc[idx,'state-name_x']
        if state_name_x!=state_name_x:
            state_name_x = df_ori.loc[idx,'app_name_x']
        add_xpath_x = df_ori.loc[idx,'add-xpath_x']
        if add_xpath_x!=add_xpath_x:
            add_xpath_x = ""
        state_name_y = df_ori.loc[idx,'state-name_y']
        if state_name_y!=state_name_y:
            state_name_y = df_ori.loc[idx,'app_name_y']
        add_xpath_y = df_ori.loc[idx,'add-xpath_y']
        if add_xpath_y!=add_xpath_y:
            add_xpath_y = ""
        src_widget = state_name_x + ":" + add_xpath_x
        tgt_widget = state_name_y + ":" + add_xpath_y
        data = {
            'widget1':src_widget,
            'widget2':tgt_widget,
            'label':1,
        }
        true_data_ori.append(json.dumps(data) + "\n")
    true_data_path = feature_save_path + \
        "true_feature_label_ori" + ".jsonl"
    with open(true_data_path, 'w') as jsonl:
        jsonl.writelines(true_data_ori)
        print("number of true_data_ori", len(true_data_ori))
        print("finish dump true_data_ori")

    print("true_data_ori_path", true_data_path)


def get_cross_validation_data_through_true_false_sample(true_data_path, false_data_path, data_save_path):
    fold_1_data, fold_2_data, fold_3_data, fold_4_data, fold_5_data = split_fold_data(true_data_path, false_data_path)
    test_data_path_list = get_cross_validation_data(fold_1_data, fold_2_data, fold_3_data, fold_4_data, fold_5_data, data_save_path)
    train_data_path_list, valid_data_path_list = get_cross_validation_data_train_valid(fold_1_data, fold_2_data, fold_3_data, fold_4_data, fold_5_data, data_save_path)
    for i in range(len(test_data_path_list)):
        print("test_data %s: %s" % (i + 1, test_data_path_list[i]))
    for i in range(len(train_data_path_list)):
        print("test_data %s: %s" % (i + 1, train_data_path_list[i]))
    for i in range(len(valid_data_path_list)):
        print("test_data %s: %s" % (i + 1, valid_data_path_list[i]))


if __name__ == "__main__":
    """
    """
    true_data_path = 'example/Train/true_false_data/'
    false_data_path = 'example/Train/true_false_data/'
    data_save_path = 'example/Train/train_valid_test_data'
    get_cross_validation_data_through_true_false_sample(true_data_path, false_data_path, data_save_path)

    # """
    # """
    # true_data_path = 'example/Train/true_false_data/'
    # false_data_path = 'example/Train/hard_false_data/'
    # data_save_path = 'train_valid_test_data_stage_2'
    # get_cross_validation_data_through_true_false_sample(true_data_path, false_data_path, data_save_path)

