# A Dual-Cycled Cross-View Transformer Network for Unified Road Layout Estimation and 3D Object Detection in the Bird's-Eye-View
#### Curie Kim, Ue-Hwan Kim 

#### [Paper](https://arxiv.org/abs/2209.08844)

## Abstract

The bird's-eye-view (BEV) representation allows robust learning of multiple tasks for autonomous driving including road layout estimation and 3D object detection. However, contemporary methods for unified road layout estimation and 3D object detection rarely handle the class imbalance of the training dataset and multi-class learning to reduce the total number of networks required. To overcome these limitations, we propose a unified model for road layout estimation and 3D object detection inspired by the transformer architecture and the CycleGAN learning framework. The proposed model deals with the performance degradation due to the class imbalance of the dataset utilizing the focal loss and the proposed dual cycle loss. Moreover, we set up extensive learning scenarios to study the effect of multi-class learning for road layout estimation in various situations. To verify the effectiveness of the proposed model and the learning scheme, we conduct a thorough ablation study and a comparative study. The experiment results attest the effectiveness of our model; we achieve state-of-the-art performance in both the road layout estimation and 3D object detection tasks.

## Contributions

*  DCT Architecture: We propose the dual-cycled crossview transformer (DCT) network for unified road layout estimation and 3D object detection for autonomous driving along with the learning scheme to handle the class imbalance.
* Multi-Class Learning: We investigate the effect of multi-class learning in the context of road layout estimation for the first time to the best of our knowledge.
* Ablation Study: We conduct a thorough ablation study and reveal important intuitions for the effect of each design choice.
* SoTA Performance: We achieve state-of-the-art performance on both road layout estimation and 3D object detection in the Argoverse and KITTI 3D Object
datasets, respectively.

## Approach overview

![Overview_final](https://user-images.githubusercontent.com/17980462/191461740-47ae6379-439a-4af3-bed9-ff60cc678b8d.png)

## Repository Structure

```plain
DCTNet/
├── crossView            # Contains scripts for dataloaders and network/model architecture
└── datasets             # Contains datasets
    ├── argoverse        # argoverse dataset
    ├── kitti            # kitti dataset 
├── log                  # Contains a log of network/model
├── losses               # Contains scripts for loss of network/model
├── models               # Contains the saved model of the network/model
├── output               # Contains output of network/model
└── splits
    ├── 3Dobject         # Training and testing splits for KITTI 3DObject Detection dataset 
    ├── argo             # Training and testing splits for Argoverse Tracking v1.0 dataset
    ├── odometry         # Training and testing splits for KITTI Odometry dataset
    └── raw              # Training and testing splits for KITTI RAW dataset(based on Schulter et. al.)
```
## Installation

Our code was tested in virtual environment with Python 3.7, Pytorch 1.7.1, torchvision 0.8.2 and installing all the dependencies listed in the requirements file.

```plain
git clone https://github.com/AutoCompSysLab/DCTNet

cd DCTNet
pip install -r requirements.txt
```
## Datasets

In the paper, we've presented results for KITTI 3D Object, KITTI Odometry, KITTI RAW, and Argoverse 3D Tracking v1.0 datasets. For comparison with [Schulter et. al.](https://cseweb.ucsd.edu/~mkchandraker/pdf/eccv18_occlusionreasoning.pdf?fileGuid=3X8QJDGGJPXyQgW9), We've used the same training and test splits sequences from the KITTI RAW dataset. For more details about the training/testing splits one can look at the `splits` directory. And you can download Ground-truth from [Monolayout](https://github.com/hbutsuak95/monolayout?fileGuid=3X8QJDGGJPXyQgW9). If the link of the road label in Monolayout is invalid, please try these links offered by [JPerciever](https://github.com/sunnyHelen/JPerceiver): [KITTI RAW](https://drive.google.com/file/d/1CN_-WKsEZrUkdLPv-MFD16PHLlOYHhaJ/view) and [KITTI Odometry](https://drive.google.com/file/d/1Z-zqMKiMqlws4s54mEjbKwK9vB_eZ1wE/view).

```plain
# Download KITTI RAW
./data/download_datasets.sh raw

# Download KITTI 3D Object
./data/download_datasets.sh object

# Download KITTI Odometry
./data/download_datasets.sh odometry

# Download Argoverse Tracking v1.0
./data/download_datasets.sh argoverse
```
The above scripts will download, unzip and store the respective datasets in the `datasets` directory.
```plain
datasets/
└── argoverse                          # argoverse dataset
    └── argoverse-tracking
        └── train1
            └── 1d676737-4110-3f7e-bec0-0c90f74c248f
                ├── car_bev_gt         # Vehicle GT
                ├── road_gt            # Road GT
                ├── stereo_front_left  # RGB image
└── kitti                              # kitti dataset 
    └── object                         # kitti 3D Object dataset 
        └── training
            ├── image_2                # RGB image
            ├── vehicle_256            # Vehicle GT
    ├── odometry                       # kitti odometry dataset 
        └── 00
            ├── image_2                # RGB image
            ├── road_dense128  # Road GT
    ├── raw                            # kitti raw dataset 
        └── 2011_09_26
            └── 2011_09_26_drive_0001_sync
                ├── image_2            # RGB image
                ├── road_dense128      # Road GT
```
## Training

1. Prepare the corresponding dataset
2. Run training
```plain
# Road (KITTI Odometry)
python3 train.py --type static --split odometry --data_path ./datasets/odometry/ --model_name <Model Name with specifications>

# Vehicle (KITTI 3D Object)
python3 train.py --type dynamic --split 3Dobject --data_path ./datasets/kitti/object/training --model_name <Model Name with specifications>

# Road (KITTI RAW)
python3 train.py --type static --split raw --data_path ./datasets/kitti/raw/  --model_name <Model Name with specifications>

# Vehicle (Argoverse Tracking v1.0)
python3 train.py --type dynamic --split argo --data_path ./datasets/argoverse/ --model_name <Model Name with specifications>

# Road (Argoverse Tracking v1.0)
python3 train.py --type static --split argo --data_path ./datasets/argoverse/ --model_name <Model Name with specifications>

# Vehicle and Road (Argoverse Tracking v1.0)
python3 train.py --type both --split argo --data_path ./datasets/argoverse/ --model_name <Model Name with specifications> --lr_steps 100  --num_class 3
```

3. The training model are in "models" (default: ./models)

## Evaluation and Saving predictions

1. Prepare the corresponding dataset
2. Download pre-trained models
3. Run evaluation
4. The results are in "output" (default: ./output)
```plain
# Evaluate on KITTI Odometry 
python3 eval.py --type static --split odometry --pretrained_path <path to the model directory> --data_path ./datasets/odometry --out_dir <path to the output directory> 

# Evaluate on KITTI 3D Object
python3 eval.py --type dynamic --split 3Dobject --pretrained_path <path to the model directory> --data_path ./datasets/kitti/object/training --out_dir <path to the output directory> 

# Evaluate on KITTI RAW
python3 eval.py --type static --split raw --pretrained_path <path to the model directory> --data_path ./datasets/kitti/raw/ --out_dir <path to the output directory> 

# Evaluate on Argoverse Tracking v1.0 (Road)
python3 eval.py --type static --split argo --pretrained_path <path to the model directory> --data_path ./datasets/kitti/argoverse/ --out_dir <path to the output directory> 

# Evaluate on Argoverse Tracking v1.0 (Vehicle)
python3 eval.py --type dynamic --split argo --pretrained_path <path to the model directory> --data_path ./datasets/kitti/argoverse --out_dir <path to the output directory> 

# Evaluate on Argoverse Tracking v1.0 (Vehicle and Road)
python3 eval.py --type both --split argo --pretrained_path <path to the model directory> --data_path ./datasets/kitti/argoverse --out_dir <path to the output directory>  --num_class 3
```
4. The results are in "output" (default: ./output)

## Pretrained Models

The following table provides links to the pre-trained models for each dataset mentioned in our paper. The table also shows the corresponding evaluation results for these models.

### Single class learning
| Dataset            | Segmentation Objects | mIOU(%) | mAP(%)| Pretrained Model                                                                                                       | 
| :--------:           | :-----:     | :----:   | :----: | :----:                                                                                                                 |
| KITTI 3D Object     | Vehicle    |  39.44 | 58.89 | [link](https://drive.google.com/drive/folders/1i_dQZO_g-I9pzpJmNKmhz4r03BIHSyy4?usp=sharing) |
| KITTI Odometry     | Road     |  77.15  | 88.28 | [link](https://drive.google.com/drive/folders/181uTXUXZ7zAWUNHwvANgP2QeI1m4Pm-p?usp=sharing) |
| KITTI Raw          | Road     |  65.86  | 86.56 | [link]() |
| Argoverse Tracking | Vehicle    |  48.04  | 68.96 | [link](https://drive.google.com/drive/folders/1_zerPSlgeG-If_HWf1sJrpiElre4Tiim?usp=sharing) |
| Argoverse Tracking | Road    |  76.71  | 88.87 | [link](https://drive.google.com/drive/folders/15PZKyb3nepkYlBsxqT8EbduRbTTj42pX?usp=sharing) |

### Multi class learning
| Dataset            | Segmentation Objects | mIOU(%) | mAP(%)| Pretrained Model                                                                                                       | 
| :--------:           | :-----:     | :----:   | :----: | :----: 
| Argoverse Tracking | Vehicle    |  31.75  | 46.20 | [link for both](https://drive.google.com/drive/folders/1yX3LWLl92YuY2zuiMWXqeUwWqBD0iRxq?usp=sharing) |
| Argoverse Tracking | Road    |  74.73  | 86.76  |  |

## Results
### Single Class Learning
![Qual](https://user-images.githubusercontent.com/17980462/191462025-9de54e5d-dc55-4bda-ac81-b8c788a6be95.png)
### Multi Class Learning
![Qual-Multi](https://user-images.githubusercontent.com/17980462/191462014-22d4f561-7ded-4cdf-9e5d-db7cdbb0aa48.png)
## Contact
If you meet any problems, please describe them in issues or contact:
* Curie Kim: [curie3170@gmail.com](curie3170@gmail.com)

## License
Thanks for the open-source related works. This project partially depends on the sources of [Monolayout](https://github.com/hbutsuak95/monolayout), [PYVA](https://github.com/JonDoe-297/cross-view), and [JPerciever](https://github.com/sunnyHelen/JPerceiver).