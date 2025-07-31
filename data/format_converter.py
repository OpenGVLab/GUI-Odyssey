import os, json
import argparse
import numpy as np
import random

### 
current_path = os.path.abspath(__file__)
DATA_DIR = os.path.dirname(current_path)
pic_base = os.path.join(DATA_DIR, 'screenshots')
anno_base = os.path.join(DATA_DIR, 'annotations')
train_anno_base = os.path.join(DATA_DIR, 'train_anno')
test_anno_base = os.path.join(DATA_DIR, 'test_anno')
split_base = os.path.join(DATA_DIR, 'splits')

PROMPT = "I'm looking for guidance on how to "

###

def decode_action(action, info):
    if action == 'CLICK' or action == "LONG_PRESS":
        if info == 'KEY_HOME':
            gt = 'PRESS_HOME'
        elif info == 'KEY_BACK':
            gt = 'PRESS_BACK'
        elif info == 'KEY_APPSELECT':
            gt = 'PRESS_RECENT'
        elif type(info) == list:
            gt = f"{action}: {tuple(info[0])}"
        else:
            raise ValueError(f'Unknown click action {info}')

    elif action == 'SCROLL':
        start = np.array(info[0])
        end = np.array(info[1])
        delta = end - start
        delta_abs = np.abs(delta)
        lr = 'LEFT' if delta[0] < 0 else 'RIGHT'
        ud = 'UP' if delta[1] < 0 else 'DOWN'
        if delta_abs[0] > delta_abs[1]:
            gt = f"SCROLL: {lr}"
        else:
            gt = f"SCROLL: {ud}"

    elif action == 'TEXT': 
        gt = f'TYPE: {info}'
    elif action == 'COMPLETE':
        gt = action
    elif action == 'INCOMPLETE':
        gt = 'IMPOSSIBLE'
    else:
        raise ValueError(f'Unknown action {action}')
    return gt


def build_train_chat(split_fp='./splits/random_split.json', his_len=4, instr_level='high'):
    os.makedirs(train_anno_base, exist_ok=True)
    name = os.path.basename(split_fp).split('.')[0]
    train_split = json.load(open(split_fp))['train']
    res = []
    idx = 0
    for f in train_split:
        this_res = []
        fp = os.path.join(anno_base, f)
        data = json.load(open(fp))
        high_level_instruction = data['task_info']['instruction']
        steps = data['steps']
        
        history_screenshot, history_action = [], []

        for step in steps:
            image = step['screenshot']
            action = step['action']
            info = step['info']
            low_level_instruction = step['low_level_instruction']
            if instr_level == 'high':
                instruction = high_level_instruction
            elif instr_level == 'low':
                instruction = low_level_instruction
                

            gt = decode_action(action, info)
            img_abs_path = os.path.join(pic_base, image)
            value = f"Picture 1: <img>{img_abs_path}</img>\n{PROMPT}{instruction}"

            his_str = ''
            for hidx, act in enumerate(history_action[-his_len:]):
                his_str += f'{hidx + 1}. {act}\n'

            if len(history_action) > 0 and his_len > 0:
                value += f'\nPrevious screenshots: <img>image-history: {img_abs_path}</img>'
                value += f'\nPrevious Actions: {his_str}'
            else:
                value += f'\nPrevious screenshots: None'
                value += f'\nPrevious Actions: None'

            value = value + '\nProvide the command-style action directly.'
            conversations = [{"from": "user", "value": value}, {"from": "assistant", "value": gt}]

            this_res.append({
                'id': f'GUIOdyssey_{name}_{idx}',
                'image': img_abs_path,
                'conversations': conversations,
                'history': str(history_screenshot),
            })
            idx += 1

            history_screenshot.append(img_abs_path)
            history_action.append(gt)

        res.extend(this_res)

    json.dump(res, open(os.path.join(train_anno_base, f'{instr_level}_' + os.path.basename(split_fp)), 'w'), indent=4, ensure_ascii=False)


def build_train_semantic_chat(split_fp='./splits/random_split.json', his_len=4, instr_level='high'):
    os.makedirs(train_anno_base, exist_ok=True)
    name = os.path.basename(split_fp).split('.')[0]
    train_split = json.load(open(split_fp))['train']
    res = []
    idx = 0
    for f in train_split:
        this_res = []
        fp = os.path.join(anno_base, f)
        data = json.load(open(fp))
        high_level_instruction = data['task_info']['instruction']
        steps = data['steps']
        
        history_screenshot, history_action = [], []
        
        for step in steps:
            image = step['screenshot']
            action = step['action']
            info = step['info']
            description = step['description']
            intention = step['intention']
            context = step['context']
            low_level_instruction = step['low_level_instruction']
            if instr_level == 'high':
                instruction = high_level_instruction
            elif instr_level == 'low':
                instruction = low_level_instruction
            
            gt = decode_action(action, info)
            img_abs_path = os.path.join(pic_base, image)
            
            ## desc
            value = f"Picture 1: <img>{img_abs_path}</img>\nProvide a 2-3 sentence summary of the screenshot content."
            conversations = [{"from": "user", "value": value}, {"from": "assistant", "value": description}]
            this_res.append({
                'id': f'GUIOdyssey_{name}_{idx}_desc',
                'image': img_abs_path,
                'conversations': conversations,
                'history': str(history_screenshot),
            })
            
            value = f"Picture 1: <img>{img_abs_path}</img>\n{PROMPT}{instruction}"
            
            his_str = ''
            for hidx, act in enumerate(history_action[-his_len:]):
                his_str += f'{hidx + 1}. {act}\n'
            
            if len(history_action) > 0 and his_len > 0:
                value += f'\nPrevious screenshots: <img>image-history: {img_abs_path}</img>'
                value += f'\nPrevious Actions: {his_str}'
            else:
                value += f'\nPrevious screenshots: None'
                value += f'\nPrevious Actions: None'
            
            value1 = value + '\nProvide the command-style action directly.'
            value2 = value + '\nExplain the intention of the next action.'
            value3 = value + '\nWhat steps have been taken so far for this task?'
            conversations1 = [{"from": "user", "value": value1}, {"from": "assistant", "value": gt}]
            conversations2 = [{"from": "user", "value": value2}, {"from": "assistant", "value": intention}]
            conversations3 = [{"from": "user", "value": value3}, {"from": "assistant", "value": context}]
            
            this_res.append({
                'id': f'GUIOdyssey_{name}_{idx}_action',
                'image': img_abs_path,
                'conversations': conversations1,
                'history': str(history_screenshot),
            })
            
            this_res.append({
                'id': f'GUIOdyssey_{name}_{idx}_intention',
                'image': img_abs_path,
                'conversations': conversations2,
                'history': str(history_screenshot),
            })
            
            this_res.append({
                'id': f'GUIOdyssey_single_{name}_{idx}_context',
                'image': img_abs_path,
                'conversations': conversations3,
                'history': str(history_screenshot),
            })
        
            history_screenshot.append(img_abs_path)
            history_action.append(gt)
            
        res.extend(this_res)
    random.shuffle(res)
    json.dump(res, open(os.path.join(train_anno_base, f'{instr_level}_sementic_' + os.path.basename(split_fp)), 'w'), indent=4, ensure_ascii=False)


def build_test(split_fp='./splits/random_split.json', his_len=4, instr_level='high'):
    os.makedirs(test_anno_base, exist_ok=True)
    name = os.path.basename(split_fp).split('.')[0]
    test_split = json.load(open(split_fp))['test']
    res = []
    idx = 0
    for f in test_split:
        this_res = []
        fp = os.path.join(anno_base, f)
        data = json.load(open(fp))
        high_level_instruction = data['task_info']['instruction']
        steps = data['steps']
        category = data['task_info']['category']
        step_length = data['step_length']

        history_screenshot = []
        history_action = []

        for step in steps:
            image = step['screenshot']
            low_level_instruction = step['low_level_instruction']
            if instr_level == 'high':
                instruction = high_level_instruction
            elif instr_level == 'low':
                instruction = low_level_instruction
            img_abs_path = os.path.join(pic_base, image)
            action = step['action']
            info = step['info']
            sam2_bbox = step['sam2_bbox']
            gt = decode_action(action, info)

            this_res.append({
                'id': f'GUIOdyssey_{name}_{idx}',
                'image': img_abs_path,
                'question': instruction,
                'answer': gt,
                'category': category,
                'step_length': step_length,
                'history_action': str(history_action),
                'history_screenshot': str(history_screenshot),
                'sam2_bbox': sam2_bbox,
            })
            idx += 1

            history_screenshot.append(img_abs_path)
            history_action.append(gt)

        res.extend(this_res)

    json.dump(res, open(os.path.join(test_anno_base, f'{instr_level}_' + os.path.basename(split_fp)), 'w'), indent=4, ensure_ascii=False)


def make_his_idx(train_base=train_anno_base, test_base=test_anno_base):
    savep = './his_index.json'
    his_dict = {}
    for subsplit in os.listdir(train_base):
        subp = os.path.join(train_base, subsplit)

        data_all = json.load(open(subp))
        for data in data_all: 
            img = data['image']
            history = eval(data['history'])
            if img in his_dict:
                assert his_dict[img] == history
            else:
                his_dict[img] = history
            
    for subsplit in os.listdir(test_base):
        subp = os.path.join(test_base, subsplit)
        data_all = json.load(open(subp))
        for data in data_all: 
            img = data['image']
            history = eval(data['history_screenshot'])
            if img in his_dict:
                assert his_dict[img] == history
            else:
                his_dict[img] = history
            
    print(len(his_dict))
    json.dump(his_dict, open(savep, 'w'), indent=4, ensure_ascii=False)
    
    
def main(args):
    for f in os.listdir(split_base):
        fp = os.path.join(split_base, f)
        if args.type == 'semantic':
            build_train_semantic_chat(fp, args.his_len, args.level)
        elif args.type == 'standard':
            build_train_chat(fp, args.his_len, args.level)
        build_test(fp, args.his_len, args.level)
        
    make_his_idx()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--his_len', type=int, default=4)
    parser.add_argument('--level', type=str, choices=['high', 'low'], default='high')
    parser.add_argument('--type', type=str, choices=['semantic', 'standard'], default='standard')
    args = parser.parse_args()
    main(args)