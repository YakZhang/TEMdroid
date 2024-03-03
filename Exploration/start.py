# coding=utf-8

import argparse
from exploration import input_manager
from exploration import input_policy
from exploration import env_manager
from exploration import DroidBot
import os


"""
"""
src_csv_prefix = 'example/Migration/src_test/'
src_csv_name = 'example.csv'
tgt_apk_path = 'example/Migration/tgt_apk/'
tgt_apk_name = 'example.apk'
output_path = 'example/Migration/tgt_test/'
tgt_xml_prefix = 'example/Migration/temp/'
script_path = None
"""
************************************************************
"""

# """
# """
# src_csv_prefix = None
# src_csv_name = None
# tgt_apk_path = 'example/Migration/tgt_apk/'
# tgt_apk_name = 'example.apk'
# output_path = 'example/Train/tgt_widget/'
# tgt_xml_prefix = 'example/Train/temp/'
# script_path = 'example/Train/tgt_gt_test'
# """
# ************************************************************
# """

def main():
    max_round_num = 1
    for round in range(max_round_num):
        task_name = src_csv_name + "_" + tgt_apk_name
        output_dir = output_path + task_name
        droidbot_policy = input_policy.POLICY_GREEDY_BFS
        if not os.path.exists(tgt_apk_path):
            print("APK does not exist.")
            return
        if not output_dir:
            print("To run in CV mode, you need to specify an output dir (using -o option).")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        droidbot = DroidBot(
            app_path=tgt_apk_path,
            device_serial=None,
            is_emulator=True,
            output_dir=output_dir,
            env_policy=env_manager.POLICY_NONE,
            policy_name=droidbot_policy,
            random_input=False,
            script_path = None,
            event_interval=1,
            timeout=60*10, # seconds (-1 for unlimit)
            event_count=input_manager.DEFAULT_EVENT_COUNT,
            cv_mode=False,
            debug_mode=False,
            keep_app=True,
            keep_env=False,
            profiling_method=None,
            grant_perm=False,
            enable_accessibility_hard=False,
            master=None,
            humanoid=None,
            ignore_ad=False,
            replay_output=None,
            src_csv_prefix=None,
            src_csv_name=None,
            tgt_xml_prefix=tgt_xml_prefix,
            src_tgt_pair_path=None,
            predict_threhold=None,
            new_function_threhold=None,
            dfs_graph=True,
            tgt_apk_name=tgt_apk_name,
            model_path=None,
        )
        droidbot.start()
        print("output_dir",output_dir)


        src_csv_prefix = src_csv_prefix


if __name__ == "__main__":
    main()
