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
from utils import *

mpl.rcParams['font.family'] = 'serif' 
mpl.rcParams['font.serif'] = ['Times New Roman']  
mpl.rcParams['font.size'] = 12

mpl.rcParams['text.color'] = 'black'

mpl.rcParams['font.weight'] = 'normal'
from scipy.stats import pearsonr

def plot_acd_lineplot(layer_acd):
    plt.figure(figsize=(12, 8))
    all_curves = []  
    colors = ['blue', 'green', 'red', 'orange']
    linestyles = ['-', '-', '-', '-']
    markers = ['o', 'o', 'o', 'o']

    for idx, key in enumerate(layer_acd):
        mean_acd_scores = np.mean(layer_acd[key], axis=0)[:-1]
        x = np.arange(len(mean_acd_scores))
        plt.plot(x, mean_acd_scores,
                 label=key,
                 marker=markers[idx % len(markers)],
                 linestyle=linestyles[idx % len(linestyles)],
                 color=colors[idx % len(colors)],
                 linewidth=1,      
                 alpha=0.5,        
                    markersize=1.5)
        all_curves.append(mean_acd_scores)

def plot_acd_ribbon(layer_acd):
    fig, ax = plt.subplots(figsize=(15, 10))
    all_curves = []
    for key in layer_acd:
        mean_scores = np.mean(layer_acd[key], axis=0)[:-1]
        all_curves.append(mean_scores)
    all_curves = np.array(all_curves)
    overall_mean = np.mean(all_curves, axis=0)
    overall_std = np.std(all_curves, axis=0)

    x = np.arange(len(overall_mean))
    ax.grid(True, axis='y', linestyle='--', linewidth=1.2, color='gray', alpha=0.3)
    ax.grid(False, axis='x')
    ax.plot(x, overall_mean, label='Overall Mean', color='blue', linestyle='--', linewidth=3, alpha=1)
    ax.fill_between(x, overall_mean - overall_std*1.5, overall_mean + overall_std*1.5,
                    color='blue', alpha=0.15, label='Std Dev')
    
    ax.set_title("Layer-wise ICR Score Evolution", fontsize=42, pad=15)
    ax.set_xlabel("Layer", fontsize=40, labelpad=15) 
    ax.set_ylabel("ICR Score", fontsize=40, labelpad=15)
    
    plt.xticks(fontsize=37)
    plt.yticks(fontsize=35)

    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')

    leg = ax.legend(frameon=True, facecolor='white', edgecolor='gray', fontsize=33, loc='upper right')
    
    plt.tight_layout()
    plt.savefig('layerwise_ICR_Score.png')

if __name__ == "__main__":
    model = "qwen3"
    file_paths = {
        'squad2': '/home/sjx/hallucination/ICR_Probe/saves/halu_eval/qwen3_halu_eval_qa_KQA',
        'halu_eval': '/home/sjx/hallucination/ICR_Probe/saves/squad2/qwen3_squad2_train_KQA'
    }
    pt_name = '/icr_score.pt'
    layer_acd = read_acd_scores_all(file_paths, pt_name)
    plot_acd_ribbon(layer_acd)
