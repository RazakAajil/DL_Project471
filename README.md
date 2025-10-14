# A Closer Look at Skeleton-based Continuous Sign Language Recognition

🏆 Official repository for A Closer Look at Skeleton-based Continuous Sign Language Recognition, the winner (1st place) in both the [Signer-Independent](https://codalab.lisn.upsaclay.fr/competitions/22899) and [Unseen Sentences](https://codalab.lisn.upsaclay.fr/competitions/22900) tasks of the ICCV 2025 [SignEval 2025: The First Multimodal Sign Language Recognition Challenge](https://multimodal-sign-language-recognition.github.io/ICCV-2025/). This implementation is largely built upon [the CoSign framework](https://openaccess.thecvf.com/content/ICCV2023/html/Jiao_CoSign_Exploring_Co-occurrence_Signals_in_Skeleton-based_Continuous_Sign_Language_Recognition_ICCV_2023_paper.html).


## Prerequisites

- This project is implemented in Pytorch (better ==2.0.0 to be compatible with ctcdecode or these may exist errors). Thus, please install Pytorch first.
- ctcdecode==0.4 [[parlance/ctcdecode]](https://github.com/parlance/ctcdecode), for beam search decode.
- sclite [[kaldi-asr/kaldi]](https://github.com/kaldi-asr/kaldi), install the kaldi tool to get sclite for evaluation. After installation, create a soft link to the sclite:  
```
mkdir ./software
ln -s PATH_TO_KALDI/tools/sctk-2.4.10/bin/sclite ./software/sclite
```

## Setup Instructions

1. **Download the dataset** [[download link]](https://www.kaggle.com/competitions/continuous-sign-language-recognition-iccv-2025/data) and place the dataset in the `./datasets` folder.

2. **Download the annotation** [[download link]](https://github.com/gufranSabri/Pose86K-CSLR-Isharah/tree/main/annotations_v2) and place them in the `./preprocess/mslr2025` folder.

3. **Preprocess the dataset**. Run the command to generate gloss dict, dataset info and groundtruth for evaluation.

```
cd ./preprocess/mslr2025
python mslr_process.py
```

## Running the Model

We provide the pretrained models for inference, you can download them from:

| Task                   | Baseline Test (WER) | Weight                                                                                               |
| ---------------------- | ------------------- | ---------------------------------------------------------------------------------------------------- |
| **Signer Independent** | 7.44%               | [GoogleDrive](https://drive.google.com/file/d/1KMXkr3UG_1Cl2AtCSCUK4eopujPxzqxg/view?usp=drive_link) |
| **Unseen Sentences**   | 28.20%              | [GoogleDrive](https://drive.google.com/file/d/1v9NYOH6ms0DyPcGw1cMjaDGmc_-tBaQ5/view?usp=drive_link) |

| Task                   | Baseline Dev (WER) | Weight                                                                                               |
| ---------------------- | ------------------ | ---------------------------------------------------------------------------------------------------- |
| **Signer Independent** | 2.2%               | [GoogleDrive](https://drive.google.com/file/d/1rGc6MqYeEm_JR6AZWweEWrX8e_X26v_p/view?usp=drive_link) |
| **Unseen Sentences**   | 35.6%              | [GoogleDrive](https://drive.google.com/file/d/1ezEFG-xMOyzwpon_XAN_35lV3Tpl9DzW/view?usp=drive_link) |

**Note:** Different tasks are suited for different data augmentation strategies during the training phase. Change the strategy in `./datasets/skeleton_feeder.py` on line 194.

### Signer Independent

- **Train:** running the command

```
python main.py --config ./configs/Double_Cosign_si.yaml
```

- **Test:** running the command

```
python main.py --config ./configs/Double_Cosign_si.yaml --phase test --load-weights PATH_TO_PRETRAINED_MODEL
```

### Unseen Sentences

- **Train:** download the pretrained weight from [here](https://drive.google.com/file/d/1oTbcL3gev4DftIFdjahJeMBbLux9Q3Y7/view?usp=drive_link), place it in the `./` folder and running the command

```
python main.py --config ./configs/Double_Cosign_us.yaml --load-weights PATH_TO_PRETRAINED_MODEL --ignore-weights classifier_static.weight classifier_motion.weight classifier_fusion.weight
```

- **Test:** running the command

```
python main.py --config ./configs/Double_Cosign_us.yaml --phase test --load-weights PATH_TO_PRETRAINED_MODEL
```

## Citation

If you find this repo useful in your research works, please consider citing:

```latex
@inproceedings{min2025closer,
title={A Closer Look at Skeleton-based Continuous Sign Language Recognition},
author={Min, Yuecong and Yang, Yifan and Jiao, Peiqi and Nan, Zixi and Chen, Xilin},
booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision Workshops},
year={2025}

}

@inproceedings{jiao2023cosign,
  title={Cosign: Exploring co-occurrence signals in skeleton-based continuous sign language recognition},
  author={Jiao, Peiqi and Min, Yuecong and Li, Yanan and Wang, Xiaotao and Lei, Lei and Chen, Xilin},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={20676--20686},
  year={2023}
}
```
