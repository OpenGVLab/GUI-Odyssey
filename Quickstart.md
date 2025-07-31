# 🚀Quick Start

## Data preprocessing

Please follow the **Dataset Access** section of the [README.md](README.md) to prepare the data, and ensure that the structure of the `./data` directory is as shown below:

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
│   └── format_converter.py
└── ...
```

Next, run the following command to generate chat-format data for training and testing.
You can adjust the following parameters as needed:
* `--his_len` specifies the length of historical information to include (default: 4).
* `--level` sets the instruction granularity, with choices of 'high' or 'low' (default: 'high').
* `--type` sets the annotation type, with choices of 'semantic' or 'standard' (default: 'standard').

```shell
cd data
python format_converter.py --his_len 4 --level high --type standard
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

Further, we also provide two variants of OdysseyAgent trained on `Train-Random` with semantic annotation: `OdysseyAgent-random-high` and `OdysseyAgent-random-low`, which are trained with high-level and low-level instructions, respectively.
- [OdysseyAgent-random-high](https://huggingface.co/hflqf88888/OdysseyAgent-random-high)
- [OdysseyAgent-random-low](https://huggingface.co/hflqf88888/OdysseyAgent-random-low)


### Fine-tuning

Specify the path to the `OdysseyAgent` and the chat-format training data generated in the  `Data preprocessing`  stage in the `script/train.sh` file. Then, run the following command:

```shell
cd src
bash script/train.sh
```

### Evalutaion

Specify the path to the checkpoint and dataset split (one of `low_app_split`, `low_device_split`, `low_random_split`, `low_task_split` `high_app_split`, `high_device_split`, `high_random_split`, `high_task_split`) in the `script/eval.sh` file. Then, run the following command:

```shell
cd src
bash script/eval.sh
```
