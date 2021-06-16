Temporal Convolution Networks with Positional Encoding for Evoked Expression Estimation
---

:1st_place_medal: **1<sup>st</sup> place** in EEV 2021 challenge: https://sites.google.com/view/auvi-cvpr2021/challenge?authuser=0 

*To be updated more later*

## Feature extraction
* *Dockerfile*: create docker image for feature extraction with TensorFlow Hub.
* Adjust *line 20* in *feature_extractor.py* and *line 19* in *utils.py*, *dataset_root_path* variable to correct folder on your computer. This folder need to contained videos in EEV with 3 separated folders (train, val, test).
* The structure of *dataset_root_path* folder as in my code. Note that eev folder is from https://github.com/google-research-datasets/eev
```
/mnt/Work/Dataset/EEV/
└───dataset
│   └───train
│       │   __18H35fPo8.mp4
│       │   _-6yhPXr_Hw.mp4
│       │   ...
│   └───val
│       │   _-91_iXATY8.mp4
│       │   _-zO8Gg2Kxw.mp4
│       │   ...
│   └───test
│       │   _EARmKIxjxQ.mp4
│       │   _FJgj1e_xo8.mp4
│       │   ...
│   
└───eev
    │   test.csv
    │   train.csv
    │   val.csv
```
## Network training
* *environment.yml*: contain list of packages (using anaconda) for network training ``` conda env create -f environment.yml ```
* Adjust *line 25* in *main.py*, *root_path* parameter to correct folder that contain feature extracted from above.
* Adjust *line 184* in *models.py*, *dataset_root_path* to the same value with *root_path* in *main.py*
* The structure of *root_path* folder as in my code. Note that eev folder is from https://github.com/google-research-datasets/eev
```
/mnt/sXProject/EvokedExpression/
└───dataset
│   └───features_v2
│       └───train
│           │   __18H35fPo8.pt
│           │   _-6yhPXr_Hw.pt
│           │   ...
│       └───val
│           │   _-91_iXATY8.pt
│           │   _-zO8Gg2Kxw.pt
│           │   ...
│       └───test
│           │   _EARmKIxjxQ.pt
│           │   _FJgj1e_xo8.pt
│           │   ...
│   
└───eev
    │   test.csv
    │   train.csv
    │   val.csv
```
