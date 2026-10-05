# Copyright (c) 2026 Yuanjie Li. All rights reserved.
# Author: Yuanjie Li
# See LICENSE in the project root for terms.

import numpy as np 
import matplotlib.pyplot as plt 

def showCurrent(data: np.ndarray, mv: np.ndarray = None, que_seq: str = None, ishow = False):
    fig = plt.figure(figsize=(10, 6))
    gs = plt.GridSpec(2, 1, height_ratios=[1, 10], hspace=0)
    bar_ax = fig.add_subplot(gs[0])
    ax = fig.add_subplot(gs[1], sharex=bar_ax)
    ax.plot(data, c='#7f7f7f', linewidth=0.8, alpha=1.0, label='Raw Data')
    if mv is not None:
        change = np.diff(mv, prepend=-1) != 0
        starts = np.where(change)[0]
        ends = np.r_[starts[1:], len(mv)]
        seg_sum = np.add.reduceat(data, starts)
        num = ends - starts 
        mean = seg_sum / num 
        mean = np.repeat(mean, num)
        ax.plot(mean, c='#2ca02c', linewidth=0.5, label='Segment Mean')
        print(f"Step Number: {len(num)}")
    if que_seq is not None:
        bar_ax.axis("off")
        colors = ["#d62728", "#1f77b4", "#2ca02c", "#ff7f0e", "#7f7f7f", "#bcbd22", "#9467bd"]
        labels = ["A", "T", "C", "G", "N", "R", "D"]
        label_map = {"A": 0, "T": 1, "C": 2, "G": 3, "N": 4, "R": 5, "D": 6}
        for start, end in zip(starts, ends):
            if (mv[start] >= len(que_seq)):
                continue
            i = label_map[que_seq[mv[start]]]
            rect = plt.Rectangle((start, 0), end - start, 1, facecolor=colors[i], edgecolor='none', alpha=0.85)
            bar_ax.add_patch(rect)
            bar_ax.text((start + end) / 2, 0.5, labels[i], ha="center", va="center", fontsize=10, color="white")
        
    plt.setp(bar_ax.get_xticklabels(), visible=False)
    plt.setp(bar_ax.get_yticklabels(), visible=False)
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    bar_ax.spines['top'].set_visible(False)
    bar_ax.spines['right'].set_visible(False)
    bar_ax.spines['left'].set_visible(False)
    bar_ax.spines['bottom'].set_visible(False)
    ax.set_xlabel("Index / Time (samples)", labelpad=3)
    ax.set_ylabel("Current (pA)", labelpad=3)
    ax.tick_params(direction='out', length=3, width=0.6, colors='black')
    plt.tight_layout()
    if ishow:
        plt.show()