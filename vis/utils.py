import os
import torch
import json
import jsonlines
import numpy as np
import seaborn as sns  
import random
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import matplotlib as mpl

def get_shape(obj):
    """
    该函数递归地推断嵌套 tensor / list / tuple 结构的维度信息，
    返回类似 shape 的多维长度列表，本质是对规则嵌套结构的维度展开。
    """
    if isinstance(obj, torch.Tensor):
        return list(obj.shape)
    elif isinstance(obj, (list, tuple)):
        return [len(obj)] + get_shape(obj[0]) if len(obj) > 0 else []
    else:
        return []


def load_acd_scores(file_path,pt_name):
    acd_score_path = file_path+pt_name
    check_result_path = file_path+'/output_judge.jsonl'
    acd_scores = torch.load(acd_score_path)
    check_result = []
    with jsonlines.open(check_result_path) as reader:
        for obj in reader:
            check_result.append({'id': obj['id'], 'result': obj['result_type']})
    return acd_scores, check_result


def read_acd_scores_all(file_paths,pt_name):
    layer_acd = {}
    for dataset,file_path in file_paths.items():
        acd_scores, check_result = load_acd_scores(file_path,pt_name)
        layer_acd_scores = [np.mean(acd_scores[key],axis=-1) for key in acd_scores.keys()]
        layer_acd[dataset] = layer_acd_scores
    return layer_acd

def mean_of_2d_list(lst):
    flat_list = [item for sublist in lst for item in sublist]

    if not flat_list:
        return 0
    mean_value = sum(flat_list) / len(flat_list)
    return mean_value