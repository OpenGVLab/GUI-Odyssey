from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
from transformers import modeling_utils
import torch

QWENVL_PATH = 'Qwen/Qwen-VL-Chat'

import sys
sys.path.append('../OdysseyAgent')
from configuration_qwen import QWenConfig
from modeling_qwen import QWenLMHeadModel

torch.manual_seed(1234)
import json, random
import time

device = 'cpu'

def load_qwen(model_name=QWENVL_PATH):
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map=None, trust_remote_code=True)
    return model

def load_qwen_tokenizer(model_name=QWENVL_PATH):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    return tokenizer


def load_new_qwen(bp='../OdysseyAgent/config.json'):
    cfg = QWenConfig(**json.load(open(bp)))
    model = QWenLMHeadModel(cfg)
    return model

def merge_weight(qwen, odysseyAgent):
    qwen_dict = qwen.state_dict()
    odysseyAgent_dict = odysseyAgent.state_dict()
    for k in qwen_dict.keys():
        if k in odysseyAgent_dict:
            odysseyAgent_dict[k] = qwen_dict[k]
    odysseyAgent.load_state_dict(odysseyAgent_dict)
    return odysseyAgent


def copy_QwenVL(bp='../OdysseyAgent'):
    tokenizer = load_qwen_tokenizer()
    tokenizer.save_pretrained(bp)
    qwen_model = load_qwen()
    new_qwen_model = load_new_qwen()
    print('start merging weight...')
    new_model = merge_weight(qwen_model, new_qwen_model)
    print('saving...')
    new_model.save_pretrained(bp)

if __name__ == '__main__':
    copy_QwenVL()
    