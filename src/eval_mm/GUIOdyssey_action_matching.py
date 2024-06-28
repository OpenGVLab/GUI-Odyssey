import numpy as np

TEXT_ANLS_THRESHOLD = 0.5
CLICK_COORD_THRESHOLD = 0.14

def levenshtein_distance(s1, s2):
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2+1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]


def text_matching(gt, pred):
    gt = gt.strip()
    pred = pred.strip()
    if gt in pred or pred in gt:
        return True
    
    dist = levenshtein_distance(gt, pred)
    length = max(len(gt), len(pred))
    value =  0.0 if length == 0 else float(dist) / float(length)
    value = 1 - value
    return value >= TEXT_ANLS_THRESHOLD


def click_matching(gt_info, pred_info):
    if type(pred_info) == str:
        pred_info = eval(pred_info)
    if type(gt_info) == str:
        gt_info = eval(gt_info)
    
    pred = np.asarray(pred_info) / 1000
    gt = np.asarray(gt_info) / 1000
        
    return np.linalg.norm(pred - gt) <= CLICK_COORD_THRESHOLD
    


def action_matching(pred_action, pred_info, gt_action, gt_info):
    pred_action = pred_action.strip()
    if type(pred_info) == str:
        pred_info = pred_info.strip()
    gt_action = gt_action.strip()
    if type(gt_info) == str:
        gt_info = gt_info.strip()
    
    if pred_action != gt_action:
        return {'is_correct': 'no', 'info': 'action_fail'}
    
    if gt_action not in ['SCROLL', 'CLICK', 'TYPE', 'LONG_PRESS']:
        return {'is_correct': 'yes', 'info': 'action_correct'}
    
    elif gt_action == 'TYPE':
        text_flag = text_matching(gt_info, pred_info)
        
        if text_flag:
            return {'is_correct': 'yes', 'info': 'type_correct'}
        else:
            return {'is_correct': 'no', 'info': 'type_fail'}
    
    elif gt_action == 'SCROLL':
        if gt_info.lower() == pred_info.lower():
            return {'is_correct': 'yes', 'info': 'scroll_correct'}
        else:
            return {'is_correct': 'no', 'info': 'scroll_fail'}        
    
    elif gt_action == 'CLICK' or gt_action == 'LONG_PRESS':
        click_flag = click_matching(gt_info, pred_info)
        
        if click_flag:
            return {'is_correct': 'yes', 'info': 'click_correct'}
        else:
            return {'is_correct': 'no', 'info': 'click_fail'}
    
    else:
        raise ValueError('Invalid action type')