# Instructions

## Training a Matching Model

Input: a sequence of source test cases, the corresponding ground truth target test cases, and the target apps

Output: a matching model

### Environment

- Python env: `pip install -r requirements.txt`
- Emulator: A Pixel 3 emulator, Android 6.0

### Getting started

**Step 1: Generate positive samples and negative samples** 

1. For `oracleEvent/GT_state_pair.py`:
    - Assign the path of `groundtruth.csv` to the variable `groundtruth_file_path`
    - Assign the path of `test_case/` to the variable `test_case_path`
    
1. Run `oracleEvent/GT_state_pair.py` to get the positive samples

1. For `stage2/get_widget_feature.py`:

    - Assign the path of the positive samples to the `ground_truth_true_path`

    - Assign the path of `test_case/` to the variable `test_case_path`

    - Assign the path of saving the negative samples to the `feature_save_path`

1. Run `stage2/get_widget_feature.py` to get the negative samples.

**Step 2: Generate train_data, validation_data, and test_data**

1. For `BERT/preprocess_data_according_app.py`:
    - Assign the paths of the positive samples and the negative samples generated in **Step 1** to the variables `true_data_path` and `false_data_path`
1. Run `BERT/preprocess_data_according_app.py` to generate the train_data, the validation_data, and the test_data

**Step 3: Train a hard-negative sample miner for the Stage 1**

1. For `run2.sh`:
    - Replace `train_data` with the path of the train_data generated in **Step 2**
    - Replace `validation_data` with the path of the validation_data generated in **Step 2**
    - Replace `test_data` with the path of the test_data generated in **Step 2**
1. Run `run2.sh` to train a hard-negative sample miner

**Step 4: Get the hard-negative samples**

1. For `run1.sh`
    - Replace `resume_from_checkpoint` with the path of the hard-negative sample miner generated in **Step 3**
    - Replace `test_file` with the path of the target widgets generated in **Step 1**
    - Replace `output_dir` to a directory to save the hard negative samples
1. Run `run1.sh` to get the predict results
1. For `oracleEvent/groundtruth_true_false_generation.py`:
    - Assign the path of `groundtruth.csv` to the variable `groundtruth_file_path`.
    - Assign the path of the predict results to the variable `widget_path_prefix`.
1. Run `oracleEvent/groundtruth_true_false_generation.py` to get the positive samples and the hard-negative samples

**Step 5: Train a widget matching model for the Stage 2**

1. For `BERT/preprocess_data_according_app.py`:
    - Assign the path of the positive samples and the hard-negative samples generated in **Step 4** to the variables `true_data_path` and `false_data_path`
1. Run `BERT/preprocess_data_according_app.py` to generate the train_data, the validation_data, and the test_data
1. For `run2.sh`:
    - Replace `train_data` with the path of the train_data
    - Replace `validation_data` with the path of the validation_data 
    - Replace `test_data` with the path of the test_data 
1. Run `run2.sh` to train a widget matching model

## Test Case Migration

Input: a source app, a target app, a source test case, and a matching model

Output: a migrated target test case for the target app

### Environment

- Python env: `pip install -r requirements.txt`
- Emulator: A Pixel 3 emulator, Android 6.0

### Preparation

1. For `Exploration/start.py`:
   - Prepare a directory to save the source test case. Assign the path of this directory to the variable `src_csv_prefix`.
   - Prepare a directory to save the apk file of the target app. Assign the path of this directory to the variable `tgt_apk_path`.
   - Prepare a directory to save the migrated test cases. Assign the path of this directory to the variable `output_path`.
   - Prepare a directory to save the intermediate files. Assign the path of this directory to the variable `tgt_xml_prefix`.
1. For `run1.sh`:
   - Prepare a directory to save the matching model. Replace `model_path` with the path of this directory.
1. Make sure the emulator has the source app and the target app installed

### Getting started

1. Launch the emulator
1. Run `Exploration/start.py`, the migrated test case should be generated at the end of this program.

