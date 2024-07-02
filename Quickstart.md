# 🚀Quick Start

## Data preprocessing

Please follow the **Dataset Access** section of the [README.md](README.md) to prepare the data, and run the `preprocessing.py` script as instructed. Ensure that the structure of the `./data` directory is as shown below:

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

Next, run the following command to generate chat-format data for training and testing. The `his_len` parameter can be set to specify the length of historical information:

```shell
cd data
python format_converter.py --his_len 4
```

## Build OdysseyAgent upon Qwen-VL-Chat

The OdysseyAgent is bulit upon [Qwen-VL](https://github.com/QwenLM/Qwen-VL).

Before running, set up the environment and install the required packages:

```shell
cd src
pip install -r requirements.txt
```

Next, initialize `OdysseyAgent` using the weights from `Qwen-VL-Chat`:

```shell
python merge_weight.py
```

Further, we also provide four variants of OdysseyAgent: 
- [OdysseyAgent-Random](https://huggingface.co/hflqf88888/OdysseyAgent-random)
- [OdysseyAgent-Task](https://huggingface.co/hflqf88888/OdysseyAgent-task)
- [OdysseyAgent-Device](https://huggingface.co/hflqf88888/OdysseyAgent-device)
- [OdysseyAgent-App](https://huggingface.co/hflqf88888/OdysseyAgent-app)

Each fine-tuned on `Train-Random`, `Train-Task`, `Train-Device`, and `Train-App` respectively.

### Fine-tuning

Specify the path to the `OdysseyAgent` and the chat-format training data generated in the  `Data preprocessing`  stage (one of the four splits) in the `script/train.sh` file. Then, run the following command:

```shell
cd src
bash script/train.sh
```

### Evalutaion

Specify the path to the checkpoint and dataset split (one of `app_split`, `device_split`, `random_split`, `task_split`) in the `script/eval.sh` file. Then, run the following command:

```shell
cd src
bash script/eval.sh
```
