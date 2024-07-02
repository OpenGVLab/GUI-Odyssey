# GUI Odyssey

**This repository is the official implementation of GUI Odyssey.**

> [GUI Odyssey: A Comprehensive Dataset for Cross-App GUI Navigation on Mobile Devices](https://arxiv.org/abs/2406.08451)  
> Quanfeng Lu, Wenqi Shao✉️⭐️, Zitao Liu, Fanqing Meng, Boxuan Li, Botong Chen, Siyuan Huang, Kaipeng Zhang, Yu Qiao, Ping Luo✉️  
> ✉️  Wenqi Shao (shaowenqi@pjlab.org.cn) and Ping Luo (pluo@cs.hku.hk) are correponding authors.   
> ⭐️ Wenqi Shao is project leader.   


## 💡 News

- `2024/06/24`: The data of [GUI Odyssey](https://arxiv.org/pdf/2406.08451) is released! Please check out [OpenGVLab/GUI-Odyssey](https://huggingface.co/datasets/OpenGVLab/GUI-Odyssey)!
- `2024/06/13`: The paper of [GUI Odyssey](https://arxiv.org/pdf/2406.08451) is released! 
<!-- And check our [project page]()! -->


## 🔆 Introduction
GUI Odyssey is a comprehensive dataset for training and evaluating **cross-app** navigation agents. GUI Odyssey consists of 7,735 episodes from 6 mobile devices, spanning 6 types of cross-app tasks, 201 apps, and 1.4K app combos.
![overview](assets/dataset_overview.jpg)


## 🛠️ Data collection pipeline 
GUI Odyssey comprises six categories of navigation tasks. For each category, we construct instruction templates with items and apps selected from a predefined pool, resulting in a vast array of unique instructions for annotating GUI episodes. Human demonstrations on an Android emulator capture the metadata of each episode in a comprehensive format. After rigorous quality checks, GUI Odyssey includes 7,735 validated cross-app GUI navigation episodes.
![pipeline](assets/pipeline.png)


## 📝 Statistics

<center>

Splits                      | # Episodes        | # Unique Prompts  | # Avg. Steps     | Data location | OdysseyAgent
:---------:                 | :---------:       | :-----------:     | :--------------: | :-----------: | :-----------:
**Total**                   | **7,735**         | **7,735**         | **15.4**         | [GUI-Odyssey](https://huggingface.co/datasets/OpenGVLab/GUI-Odyssey)                | [OdysseyAgent]([https://huggingface.co/datasets/OpenGVLab/GUI-Odyssey](https://huggingface.co/collections/hflqf88888/gui-odyssey-6683bac37ad6fe37b1215c18))
Train-Random \& Test-Random | 5,802 / 1,933     | 5,802 / 1,933     | 15.4 / 15.2      | [random_split.json](https://huggingface.co/datasets/OpenGVLab/GUI-Odyssey/tree/main/splits)   | [OdysseyAgent-Random](https://huggingface.co/hflqf88888/OdysseyAgent-random)
Train-Task \& Test-Task     | 6,719 / 1,016     | 6,719 / 1,016     | 15.0 / 17.6      | [task_split.json](https://huggingface.co/datasets/OpenGVLab/GUI-Odyssey/tree/main/splits)     | [OdysseyAgent-Task](https://huggingface.co/hflqf88888/OdysseyAgent-task)
Train-Device \& Test-Device | 6,473 / 1,262     | 6,473 / 1,262     | 15.4 / 15.0      | [device_split.json](https://huggingface.co/datasets/OpenGVLab/GUI-Odyssey/tree/main/splits)   | [OdysseyAgent-Device](https://huggingface.co/hflqf88888/OdysseyAgent-device)
Train-App \& Test-App       | 6,596 / 1,139     | 6,596 / 1,139     | 15.4 / 15.3      | [app_split.json](https://huggingface.co/datasets/OpenGVLab/GUI-Odyssey/tree/main/splits)      | [OdysseyAgent-App](https://huggingface.co/hflqf88888/OdysseyAgent-app)

</center>

## 💫 Dataset Access

The whole GUI Odyssey is hosted on [Huggingface](https://huggingface.co/datasets/OpenGVLab/GUI-Odyssey). 

Clone the entire dataset from Huggingface:
```shell
git clone https://huggingface.co/datasets/OpenGVLab/GUI-Odyssey
```
And then move the cloned dataset into `./data` directory. After that, the structure of `./data` should look like this:


```
GUI-Odyssey
├── data
│   ├── annotations
│   │   └── *.json
│   ├── screenshots
│   │   └── data_*
│   │        └── *.png
│   ├── splits
│   │   ├── app_split.json
│   │   ├── device_split.json
│   │   ├── random_split.json
│   │   └── task_split.json
│   ├── format_converter.py
│   └── preprocessing.py
└── ...
```

Then organize the screenshots folder:

```shell
cd data
python preprocessing.py
```

Finally, the structure of `./data` should look like this:

```
GUI-Odyssey
├── data
│   ├── annotations
│   │   └── *.json
│   ├── screenshots
│   │   └── *.png
│   ├── splits
│   │   ├── app_split.json
│   │   ├── device_split.json
│   │   ├── random_split.json
│   │   └── task_split.json
│   ├── format_converter.py
│   └── preprocessing.py
└── ...
```


## ⚙️ Detailed Data Information
Please refer to [this](introduction.md).


## 🚀 Quick Start

Please refer to [this](Quickstart.md) to quick start.

## 📖 Release Process

- [x] Dataset
  - [x] Screenshots of GUI Odyssey
  - [x] annotations of GUI Odyssey
  - [x] split files of GUI Odyssey
- [x]  Code
  - [x] data preprocessing code
  - [x] inference code
- [x]  Models


## 🖊️ Citation 
If you feel GUI Odyssey useful in your project or research, please kindly use the following BibTeX entry to cite our paper. Thanks!
```bib
@misc{lu2024gui,
      title={GUI Odyssey: A Comprehensive Dataset for Cross-App GUI Navigation on Mobile Devices}, 
      author={Quanfeng Lu and Wenqi Shao and Zitao Liu and Fanqing Meng and Boxuan Li and Botong Chen and Siyuan Huang and Kaipeng Zhang and Yu Qiao and Ping Luo},
      year={2024},
      eprint={2406.08451},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

<!-- ## 📢 Disclaimer

We develop this repository for RESEARCH purposes, so it can only be used for personal/research/non-commercial purposes. -->
