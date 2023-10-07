# ArSDM

This is the official implementation of [ArSDM: Colonoscopy Images Synthesis with Adaptive Refinement Semantic Diffusion Models](https://arxiv.org/abs/2309.01111) at MICCAI-2023.
<p align="center">
<img src=assets/framework.png />
</p>

## Table of Contents
- [Requirements](#requirements)
- [Dataset Preparation](#dataset-preparation)
- [Sampling with ArSDMs](#sampling-with-arsdms)
- [Training Your Own ArSDMs](#training-your-own-arsdms)
- [Downstream Evaluation](#downstream-evaluation)
- [Acknowledgement](#acknowledgement)
- [Citations](#citations)

## Requirements
To get started, ensure you have the following dependencies installed:
```bash
conda create -n ArSDM python=3.8
conda activate ArSDM
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
pip install -r requirements.txt
```

## Dataset Preparation
You can download the dataset from [this repository](https://github.com/DengPingFan/PraNet).

Please organize the dataset with the following structure:
```angular2
├── ${data_root}
│ ├── ${train_data_dir}
│ │ ├── images
│ │ │ ├── ***.png
│ │ ├── masks
│ │ │ ├── ***.png
│ ├── ${test_data_dir}
│ │ ├── images
│ │ │ ├── ***.png
│ │ ├── masks
│ │ │ ├── ***.png
```
## Sampling with ArSDMs
### Model Zoo
We provide pre-trained models for various configurations:

| Ada. Loss | Refinement | Saved Epoch | Batch Size | GPU           | Link                                                                                                                                                  |                                      
|-----------|------------|-------------|------------|---------------|-------------------------------------------------------------------------------------------------------------------------------------------------------|
| x         | x          | 94          | 8          | 2 A100 (80GB) | [OneDrive](https://cuhko365-my.sharepoint.com/:u:/g/personal/118010053_link_cuhk_edu_cn/EUcEvfckWOFHtjGAjGNjQZIBrpggLNXiOm5dZL3lpgcx-w?e=F1bog8)      |
| &#10004;  | x          | 100         | 8          | 1 A100 (80GB) | [OneDrive](https://cuhko365-my.sharepoint.com/:u:/g/personal/118010053_link_cuhk_edu_cn/EcPG7yBMUBRMjejlOv4QCSMBdHaT4hQ1HmnMRPhlXPf-jA?e=JlPkeR)      |
| x         | &#10004;   | 2           | 8          | 1 A100 (80GB) | [OneDrive](https://cuhko365-my.sharepoint.com/:u:/g/personal/118010053_link_cuhk_edu_cn/EWQ0gnBeWNlKlGLgoku44tUB89U2df_HzjaYlXefV4TzQQ?e=28mKHO)      |
| &#10004;  | &#10004;   | 3           | 8          | 1 A100 (80GB) | [OneDrive](https://cuhko365-my.sharepoint.com/:u:/g/personal/118010053_link_cuhk_edu_cn/EaRh_sxw4zVIq2sa_uNsnQ0BhKiihz2M2T0z67nkdj3y0Q?e=HIqY7n)      |  


Download the pre-trained weights above or follow the [next section](#train-your-own-arsdms) to train your own models.


Specify the ```CKPT_PATH``` and ```RESULT_DIR``` in the ```sample.py``` file and run the following command:

```bash
python sample.py
```

Illustrations of generated samples with the corresponding masks and original images for comparison reference are shown below:

<p align="center">
<img src=assets/samples.png />
</p>

## Train Your Own ArSDMs
To train your own ArSDMs, follow these steps:

1. Specify the ```train_data_dir``` and ```test_data_dir``` in the corresponding ```ArSDM_xxx.yaml``` file in the ```configs``` folder.
2. Specify the ```CONFIG_FILE_PATH``` in the ```main.py``` file.
3. Run the following command:

```bash
python main.py
```

If you intend to train models with ```refinement```, ensure that you have trained or downloaded diffusion model weights and the [PraNet](https://github.com/DengPingFan/PraNet) weights. Specify the ```ckpt_path``` and ```pranet_path``` in the ```ArSDM_xxx.yaml``` config file.

For example, if you want to train models with ```adaptive loss``` and ```refinement``` (**ArSDM**), first train a diffusion model with ```adaptive loss``` only using ```ArSDM_adaptive.yaml```. Then, specify the trained weights path with ```ckpt_path``` and use ```ArSDM_our.yaml``` to train the final model.

Please note that all experiments were conducted using NVIDIA A100 (80GB) with a batch size of 8. If you have GPUs with lower memory, please reduce the ```batch_size``` in the config files accordingly.



## Downstream Evaluation
To perform downstream evaluation, follow the steps in the [Sampling with ArSDMs](#sampling-with-arsdms) section to sample image-mask pairs and create a new training dataset for downstream polyp segmentation and detection tasks. For training these tasks, refer to the official repositories:

#### Polyp Segmentation:
* [PraNet](https://github.com/DengPingFan/PraNet)
* [SANet](https://github.com/weijun88/SANet)
* [Polyp-PVT](https://github.com/DengPingFan/Polyp-PVT)

#### Polyp Detection:
* [CenterNet](https://github.com/xingyizhou/CenterNet)
* [Sparse R-CNN](https://github.com/PeizeSun/SparseR-CNN)
* [Deformable DETR](https://github.com/fundamentalvision/Deformable-DETR)


## Acknowledgement
This project is built upon the foundations of several open-source codebases, including [LDM](https://github.com/CompVis/latent-diffusion), [guided-diffusion](https://github.com/openai/guided-diffusion) and [SDM](https://github.com/WeilunWang/semantic-diffusion-model). We extend our gratitude to the authors of these codebases for their invaluable contributions to the research community.


## Citations
If you find **ArSDM** useful for your research, please consider citing our paper:
```angular2
@inproceedings{du2023arsdm,
  title={ArSDM: Colonoscopy Images Synthesis with Adaptive Refinement Semantic Diffusion Models},
  author={Du, Yuhao and Jiang, Yuncheng and Tan, Shuangyi and Wu, Xusheng and Dou, Qi and Li, Zhen and Li, Guanbin and Wan, Xiang},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={339--349},
  year={2023},
  organization={Springer}
}
```
