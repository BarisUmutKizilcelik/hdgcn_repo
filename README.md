# Action Recognition with Noisy Labels

This is the repository for the CV:HCI practical course with topic: Action Recognition with Noisy Labels. In this repository, you can find the implementation for this project with respect to HDGCN model. 

## Dependencies

We implement our methods by PyTorch on Quadro RTX 6000 and 8000 GPU. The environment is as bellow:

- [Python](https://python.org/), version >= 3.7
- [Ubuntu 16.04 Desktop](https://ubuntu.com/download)
- [PyTorch](https://PyTorch.org/), version = 1.12.1
- [CUDA](https://developer.nvidia.com/cuda-downloads), version = 11.6


## Installation

Please find installation instructions for HDGCN in [README.md](HD-GCN/README.md) in HD-GCN folder and follow the instructions in [README.md](HD-GCN/README.md) to prepare the NTU60 dataset for HDGCN.

## Experiments

We verify the effectiveness of the PNP method and Robust Early-Learning method on simulated noisy datasets. 

In this repository, we provide the subset we used for this project. You should download the NTU60 dataset and create the subset according to the csv files. The dataset should be put into the same folder of labels as the instructions in [README.md](HD-GCN/README.md).

To generate noise labels, you can run the [add_noise.py](HD-GCN/add_noise.py) in the script folder with any noise proportion.

To run HDGCN using NTU60 dataset with noisy labels, please follow the instructions in [README.md](HD-GCN/README.md) in HD-GCN folder.

Here is a training example for HDGCN: 
```bash
python main.py --config HD-GCN/config/nturgbd60-cross-view/bone_com_1.yaml --device 0
```
Here is a test example for HDGCN: 
```bash
python main.py --config <work_dir>/config.yaml --work-dir <work_dir> --phase test --save-score True --weights <work_dir>/xxx.pt --device 0
```

To run Robust EL using NTU60 dataset with noisy labels, please follow the instructions in [README.md](HD-GCN/CDR/README.md) in CDR folder.

Here is a training example for Robust EL: 
```bash
python3 main.py --dataset ntu60
```

To run PNP, firstly you need create a new dataset for AimCLR. You can run [aimclr_gen_data_for_hdgcn.py](HD-GCN/aimclr_gen_data_for_hdgcn.py) to create new NTU60 dataset. Then please follow [README.md](HD-GCN/hdgcn_PNP/PNP/README.md).

Here is a training example for PNP: 
```bash
bash scripts/hdgcn.sh 
```


## Visualization

To visualize the resultant embeddings of your model, you can first perform test and set the Task as t-sne and save the output png file. You need to change path of model for the current training. Then you can run the [tsne_hdgcn.ipynb](HD-GCN/tsne.ipynb) to visualiza them in 2d or 3D via t-SNE.
