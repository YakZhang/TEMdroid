# TEMdroid

TEMdroid is a learning-based widget matching approach for test case migration.

It is trained with test case migration data and utilizes BERT to incorporate contextual information instead of static word embeddings.

By learning a matching model rather than relying on manually defined matching functions, TMEdroid can handle complex matching relations and adapt to diverse matching scenarios in real-world apps.

## Overview

- `Code/` contains the source code of TEMdroid.
- `Dataset/` contains the experiment files for TEMdroid in the Craftdroid dataset and the SemFinder dataset.
- `Result/` contains the evaluation results of TEMdroid and baselines.

## Code

- `BERT/` contains the source code of our matching model.
- `oracleEvent/` contains the source code for the data processing in the Craftdroid dataset.
- `stage2/` contains the source code for the general data processing.
- `semfinder/` contains the source code for the data processing in the SemFinder dataset.
- `Exploration/` contains the source code for the dynamic exploration for the target apps.
- `Instruction.md` introduces how to start this project.

## Dataset

- `Craftdroid/` contains the groundtruth (`groundtruth.csv`), the test cases (`test_case/`), the app files (`apk/`), and the app version information of the Craftdroid dataset (`app_subject.xlsx`).
- `SemFinder/` contains the groundtruth (`groundtruth.csv`), the app version information of the SemFinder dataset(`app_subject.xlsx`), and the app files  (`apk/`) .
- `Usefulness_study/` contains the app files (`apk/`) and the app version information of the usefulness study (`app_subject.xlsx`).
- Users can directly use the apk files from `apk/` to find the apps in our experimental subjects. Users can also use the groundtruth information and the test cases to evaluate their own test case migration approaches.

## Results

- `Craftdroid/` contains the results of TEMdroid and baselines in the Craftdroid dataset.
- `SemFinder/` contains the results of TEMdroid and baselines in the SemFinder dataset.
- `Usefulness/` contains the results of TEMdroid for usefulness study.
- `statistics.xlsx` contains detailed results that are not included in the paper due to space constraints. 
- Users can use the results of TEMdroid and baselines to compare with their own approaches. Users can also get detailed information related to TEMdroid and baselines from `statistics.xlsx`.

## Implementation

We implement TEMdroid in Python. The matching model of TEMdroid is implemented based on PyTorch framework and the pretrained BERT module is from Huggingface.  The dynamic exploration of apps is under a Pixel 3 Emulator running Android 6.0. Dependency libraries and versions are available at `Code/requirements.txt`.

## Acknowledgement

The code of `Exploration/` is partly refered to Droidbot.

Li, Yuanchun, el al. ''Droidbot: a lightweight UI-guided test input generator for Android.'' In Proceedings of the 39th International Conference on Software Engineeering Companion (ICSE-C'17). Buenos Aires, Argentina, 2017.

## Related Paper

Yakun Zhang, Wenjie Zhang, Dezhi Ran, Qihao Zhu, Chengfeng Dou, Dan Hao, Tao Xie, and Lu Zhang. 2024. Learning-based Widget Matching for Migrating GUI Test Cases. In Proceedings of the 46th International Conference on Software Engineering (ICSE'24), April 14-20, 2024. Portugal, Lisbon, 13 pages.

If this repository is useful for your research, please cite this paper.