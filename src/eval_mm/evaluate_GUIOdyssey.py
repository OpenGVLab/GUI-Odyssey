import argparse
import itertools
import json
import os, sys
import torch
import random
import time
from functools import partial
from tqdm import tqdm
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
import warnings
from GUIOdyssey_action_matching import action_matching
import numpy as np

warnings.filterwarnings("ignore")
current_path = os.path.abspath(__file__)
DATA_DIR = os.path.dirname(os.path.dirname(os.path.dirname(current_path)))

sys.path.append(os.path.join(DATA_DIR, 'OdysseyAgent'))
sys.path.append(os.path.join(DATA_DIR, 'src'))
from qwen_generation_utils import make_context, decode_tokens


IMAGE_HISTORY = True

ds_collections = {
    'high_app_split': {
        'test': '../data/test_anno/high_app_split.json',
        'metric': 'macro'
    },
    'high_device_split': {
        'test': '../data/test_anno/high_device_split.json',
        'metric': 'macro'
    },
    'high_random_split': {
        'test': '../data/test_anno/high_random_split.json',
        'metric': 'micro'
    },
    'high_task_split': {
        'test': '../data/test_anno/high_task_split.json',
        'metric': 'macro'
    },
    'low_app_split': {
        'test': '../data/test_anno/low_app_split.json',
        'metric': 'macro'
    },
    'low_device_split': {
        'test': '../data/test_anno/low_device_split.json',
        'metric': 'macro'
    },
    'low_random_split': {
        'test': '../data/test_anno/low_random_split.json',
        'metric': 'micro'
    },
    'low_task_split': {
        'test': '../data/test_anno/low_task_split.json',
        'metric': 'macro'
    }
}


def simple_decode(gt):
    gts = gt.split(':')
    gt_action = gts[0].strip()
    if len(gts) > 1:
        action = gt_action
        info = gts[1].strip()
        if action in ['CLICK', "LONG_PRESS"]:
            info = eval(info)
    else:
        action = gt_action
        info = ""
    return {"action": action, "info": info}

def stat_result(eval_dict, metric):
    text_correct = sum([1 for _ in eval_dict if _['info'] == 'type_correct'])
    type_correct = sum([1 for _ in eval_dict if _['info'] != 'action_fail'])
    text_total = sum([1 for _ in eval_dict if _['info'].startswith('type_')])
    
    if metric == 'macro':
        action_correct = sum([1 for _ in eval_dict if _['is_correct'] == 'yes'])
        AMS = round(action_correct / len(eval_dict) * 100, 2)
        SR_cnt, SR_tot, SR = check_SR(eval_dict)
    elif metric == 'micro':
        task_cate_dict = {}
        acc_list = []
        SR_list = []
        for sample in eval_dict:
            cat = sample['more_info']['category']
            if cat not in task_cate_dict:
                task_cate_dict[cat] = []
            task_cate_dict[cat].append(sample)
        assert len(task_cate_dict) == 6
        for k, v in task_cate_dict.items():
            SR_cnt, SR_tot, SR = check_SR(v)
            SR_list.append((SR))
            acc = round(sum([1 for x in v if x['is_correct'] == 'yes']) / len(v) * 100, 2)
            acc_list.append(acc)
            print(f'category: {k}, AMS: {acc}, SR: {SR}')
        
        AMS = np.round(np.mean(acc_list), 2)  
        SR = np.round(np.mean(SR_list), 2)
        
    else:
        raise ValueError(f'No metric {metric} found.')
    
    info = {
        'AMS': AMS,
        'SR': SR,
        'total': len(eval_dict),
        'action_type': '{} / {} = {:.2f}'.format(type_correct, len(eval_dict), type_correct / len(eval_dict) * 100),
        'text': '{} / {} = {:.2f}'.format(text_correct, text_total, text_correct / text_total * 100),
    }
        
    print(info)
    return info

def action_matching_evaluation(pred_output, metric='macro'):
    eval_dict = []
    for idx, sample in enumerate(pred_output):
        question, pred, gt, more_info = sample['question'], sample['pred'], sample['gt'], sample['more_info']
        sample_eval_dict = {'question': question, 'pred': str(pred), 'gt': str(gt), 'more_info': more_info}
        sam2_bbox = more_info['sam2_bbox']
        
        gt_simple_info = simple_decode(gt)
        gt_action = gt_simple_info['action']
        gt_info = gt_simple_info['info']
        
        try:
            pred_simple_info = simple_decode(pred)
            pred_action = pred_simple_info['action']
            pred_info = pred_simple_info['info']
        except:
            print('eval err:', idx, pred)
            log_info = {'is_correct': 'no', 'info': 'invalid'}
            sample_eval_dict.update(log_info)
            eval_dict.append(sample_eval_dict)
            continue
        
        try:
            check_match = action_matching(pred_action, pred_info, gt_action, gt_info, sam2_bbox)
        except Exception as exc:
            print('eval err:', gt, pred, exc)
            check_match = {'is_correct': 'no', 'info': 'invalid'}

        sample_eval_dict.update(check_match)
        eval_dict.append(sample_eval_dict)
        
    
    info = stat_result(eval_dict, metric)
    metrics = {"info": info, "pred": eval_dict}
    return metrics



def check_SR(eval_dict):
    episode_dict = {}
    steps_map = {}
    for data in eval_dict:
        if 'img' in data: img = data['img']
        elif 'image' in data: img = data['image']
        else: img = data['question'].split('</img>')[0].split('<img>')[1]
        img = os.path.basename(img)
        tail = img.split('_')[-1]
        episode = img.replace(f'_{tail}', '')
        if episode not in episode_dict: 
            episode_dict[episode] = []
        else: 
            assert steps_map[episode] == data['more_info']['step_length']
        
        info = data['is_correct']
        episode_dict[episode].append(info)
        steps_map[episode] = data['more_info']['step_length']
        
    cnt, tot = 0, 0
    for k, v in episode_dict.items():
        if len(v) != steps_map[k]:
            print(f'step length of {k} does not match.')
            continue
        tot += 1
        v = list(set(v))
        if len(v) == 1 and v[0] == 'yes':
            cnt += 1
    
    if tot == 0: SR = 0
    else: SR = round(cnt / tot * 100, 2)
    print(f'total episode: {tot}, successful episode: {cnt}, SR: {SR}')
    return cnt, tot, SR


def rank0_print(*args):
    if torch.distributed.get_rank() == 0:
        print(*args)


def collate_fn(batches, tokenizer):
    question = [_['question'] for _ in batches]
    raw_texts = [_['raw_text'] for _ in batches]
    gt = [_['gt'] for _ in batches]
    more_info = [_['more_info'] for _ in batches]
    input_ids = tokenizer(raw_texts, return_tensors='pt', padding='longest')

    return question, raw_texts, input_ids.input_ids, input_ids.attention_mask, gt, more_info


class LazySupervisedDataset(torch.utils.data.Dataset):
    def __init__(self, datapath, tokenizer: transformers.PreTrainedTokenizer, his_len, max_window_size, chat_format):
        super(LazySupervisedDataset, self).__init__()
        self.all_data = json.load(open(datapath))
        self.tokenizer = tokenizer
        self.max_window_size = max_window_size
        self.chat_format = chat_format
        self.his_len = his_len

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, idx):
        data = self.all_data[idx]
        img = data['image']
        question = f"Picture 1: <img>{img}</img>\nI'm looking for guidance on how to {data['question']}"
        answer = data['answer']
        
        history_action = eval(data['history_action'])[-self.his_len:]
        if IMAGE_HISTORY:
            if len(history_action) > 0:
                his_img = f'\nPrevious screenshots: <img>image-history: {img}</img>'
                his_str = '\nPrevious Actions: '
                for idx, hi in enumerate(history_action):
                    his_str += f"{idx+1}. {hi}\n"
                
                question = f"{question}{his_img}{his_str}"
            else:
                question += f'\nPrevious screenshots: None'
                question += f'\nPrevious Actions: None'
        else:
            if len(history_action) > 0:
                his_str = '\nPrevious Actions: '
                for idx, hi in enumerate(history_action):
                    his_str += f"{idx+1}. {hi}\n"
                
                question = f"{question}{his_str}"
        
        question += '\nProvide the command-style action directly.'
        raw_text, _ = make_context(self.tokenizer, question, system="You are a helpful assistant.", max_window_size=self.max_window_size, chat_format=self.chat_format)
        more_info = {'category': data['category'], 'step_length': data['step_length'], 'sam2_bbox': data['sam2_bbox']}
        return {
            'raw_text': raw_text,
            'question': question,
            'gt': answer,
            'more_info': more_info
        }


class InferenceSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, size):
        self._size = int(size)
        assert size > 0
        self._rank = torch.distributed.get_rank()
        self._world_size = torch.distributed.get_world_size()
        self._local_indices = self._get_local_indices(size, self._world_size, self._rank)

    @staticmethod
    def _get_local_indices(total_size, world_size, rank):
        shard_size = total_size // world_size
        left = total_size % world_size
        shard_sizes = [shard_size + int(r < left) for r in range(world_size)]

        begin = sum(shard_sizes[:rank])
        end = min(sum(shard_sizes[:rank + 1]), total_size)
        return range(begin, end)

    def __iter__(self):
        yield from self._local_indices

    def __len__(self):
        return len(self._local_indices)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='/path/to/model')
    parser.add_argument('--dataset', type=str, default='random_split')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--num-workers', type=int, default=12)
    parser.add_argument('--seed', type=int, default=2024)
    parser.add_argument('--output-path', type=str, default='output_res')
    parser.add_argument('--image-history', type=str, default='yes')
    parser.add_argument('--his_len', type=int, default=4)
    args = parser.parse_args()
    
    if args.image_history == 'no':
        IMAGE_HISTORY = False
    else:
        IMAGE_HISTORY = True

    torch.distributed.init_process_group(backend='nccl', world_size=int(os.getenv('WORLD_SIZE', '1')), rank=int(os.getenv('RANK', '0')),)
    torch.cuda.set_device(int(os.getenv('LOCAL_RANK', 0)))
    rank0_print(args)
    rank0_print('load model...')
    model = AutoModelForCausalLM.from_pretrained(args.checkpoint, device_map='cuda', trust_remote_code=True).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, trust_remote_code=True)
    tokenizer.padding_side = 'left'
    tokenizer.pad_token_id = tokenizer.eod_id

    rank0_print('init test set...')
    random.seed(args.seed)
    datapath = ds_collections[args.dataset]['test']
    dataset = LazySupervisedDataset(datapath=datapath, tokenizer=tokenizer, his_len=args.his_len, max_window_size=6144, chat_format='chatml')

    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        sampler=InferenceSampler(len(dataset)),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=partial(collate_fn, tokenizer=tokenizer),
    )

    rank0_print(f'len of dataloader: {len(dataloader)}')
    outputs = []
    for _, (question, raw_texts, input_ids, attention_mask, gt, more_info) in tqdm(enumerate(dataloader)):
        try:
            batch_input_ids = input_ids.to(model.device)
            batch_input_attention_mask = attention_mask.to(model.device)
            
            batch_out_ids = model.generate(
                input_ids=batch_input_ids,
                attention_mask=batch_input_attention_mask,
                do_sample=False,
                num_beams=1,
                length_penalty=1,
                num_return_sequences=1,
                use_cache=True,
                pad_token_id=tokenizer.eod_id,
                eos_token_id=tokenizer.eod_id,
                min_new_tokens=1,
                max_new_tokens=30,
                )

            padding_lens = [batch_input_ids[i].eq(tokenizer.pad_token_id).sum().item() for i in range(batch_input_ids.size(0))]
            batch_response = [decode_tokens(batch_out_ids[i][padding_lens[i]:], tokenizer, raw_text_len=len(raw_texts[i]), context_length=(batch_input_ids[i].size(0)-padding_lens[i]),
            chat_format="chatml", verbose=False, errors='replace') for i in range(len(raw_texts))]
            
            for q, pred, _gt, info in zip(question, batch_response, gt, more_info):
                outputs.append({
                        'question': q,
                        'pred': str(pred),
                        'gt': _gt,
                        'more_info': info,
                    })
        except Exception as e:
            print('error', e)
            print(_)
            continue

    print(f'Rank {torch.distributed.get_rank()}: inference finished.')
    torch.distributed.barrier()

    world_size = torch.distributed.get_world_size()
    merged_outputs = [None for _ in range(world_size)]
    torch.distributed.all_gather_object(merged_outputs, json.dumps(outputs))

    merged_outputs = [json.loads(_) for _ in merged_outputs]
    merged_outputs = [_ for _ in itertools.chain.from_iterable(merged_outputs)]

    if torch.distributed.get_rank() == 0:
        print(f"Saving predicted result ...")
        # time_prefix = time.strftime('%y%m%d%H%M%S', time.localtime())
        os.makedirs(args.output_path, exist_ok=True)
        model_name = str(args.checkpoint).replace('/', '_')
        savefile = os.path.join(args.output_path, f'{model_name}_{args.dataset}.json')
        json.dump(merged_outputs, open(savefile, 'w', encoding='utf-8', errors='ignore'), indent=4, ensure_ascii=False)
        
        print(f"Evaluating {args.dataset} ...")
        metrics = action_matching_evaluation(merged_outputs, metric=ds_collections[args.dataset]['metric'])
        
        output_data = {'dataset': args.dataset, 'model': model_name, 'metrics': metrics}
        json.dump(output_data, open(savefile, 'w', encoding='utf-8', errors='ignore'), indent=4, ensure_ascii=False)
        
    torch.distributed.barrier()
