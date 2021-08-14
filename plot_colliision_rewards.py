#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 01:15:15 2020

@author: dl-asoro
"""
import pickle
import numpy as np
from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'

import matplotlib.pyplot as plt
plt.rc('font', family='serif')

from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
from matplotlib.lines import Line2D


def smooth(scalars, weight):  # Weight between 0 and 1
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        last = smoothed_val                                  # Anchor the last smoothed value

    return smoothed


gt_collisions = {'eth': 253.0, 'hotel': 288.0, 'univ': 8.0, 'zara1': 253.0, 'zara2': 245.0}
runs = [
        'run-10_hdim64_elayrs8_dlayrs2_zdim256_batch64_lr0.0003_kld0.1_edg0.0_mse1.0_cri1.0_gwt1.0_cwt1.0_gdist0.4_center_scale_mse_adv*val1',
        'run-10_hdim64_elayrs8_dlayrs2_zdim256_batch64_lr0.0003_kld0.1_edg0.0_mse1.0_cri1.0_gwt0.0_cwt1.0_center_scale_mse_adv*val1',
        'run-10_hdim64_elayrs8_dlayrs2_zdim256_batch64_lr0.0003_kld0.1_edg0.0_mse1.0_cri0.0_gwt0.0_cwt0.0_center_scale_mse_adv*val1',
        ]
out_dir = './out/SC_GCN/trial_16/'

num_points = 100
datasets = ['eth', 'hotel', 'univ', 'zara1', 'zara2']

plt.close('all')
fig, ax = plt.subplots()

# cmap = plt.get_cmap(name='Dark2', lut=len(runs))
# cmap = {dset:cmap(i) for i, dset in enumerate(datasets)}
cmap = ['green', 'orange', 'gray']
legends_handles = []
for i, run in enumerate(runs):
    run_dir = out_dir + run + '/'
    
    with open(run_dir + 'args.pkl', 'rb') as f: 
        args = pickle.load(f)
            
    # datasets = 'zara2'
    all_rewards = []
    all_collisions = []
    all_train_mse = []
    all_val_mse = []
    for dset_name in datasets:        
            
        hist_path = run_dir + dset_name 
        with open(run_dir + dset_name + '/history.pkl', 'rb') as f:
            hist = pickle.load(f)
        
        train_rewards = hist.get('train_collision_rewards')
        all_rewards.append(train_rewards[:num_points])
        
        train_collisions = hist.get('train_collisions')
        all_collisions.append(train_collisions[:num_points])
        
        train_mse = hist.get('train_mse_loss')
        all_train_mse.append(train_mse[:num_points])
        
        val_mse = hist.get('val_mse_loss')
        all_val_mse.append(val_mse[:num_points])
        
        # x = np.arange(num_points)
        # y = train_rewards[:num_points]
        # # ci = 1.96 * np.std(y)/np.sqrt(num_points)
        # y = smooth(y, 0.4)
        
        # ls = 'solid' if args.collision_rewards_wt>0 else 'dashed'
        # ax.plot(x, y, linestyle=ls, color=cmap[dset_name], label=dset_name)
        # # ax.fill_between(x, (y-ci), (y+ci), color=cmap[dset_name], alpha=0.1)
        
    x = np.arange(num_points)
    y = np.mean(all_rewards, axis=0)
    ci = 1.96 * np.std(y)/np.sqrt(num_points)
    y = smooth(y, 0.4)
    
    # color = cmap(i)
    color = cmap[i]
    line, = ax.plot(x, y, linestyle='solid', color=color, lw=1)
    ax.fill_between(x, (y-ci), (y+ci), color=color, alpha=0.4)
    legends_handles.append(line)
    
    # plot train mse
    x = np.arange(0, num_points, 2)
    y = np.mean(all_train_mse, axis=0)[x]
    line, = ax.plot(x, 1/(y+1), linestyle='dashed', color=color, lw=1)

# ax.legend(handles=legends_handles, labels= ('GGCN', 'SC-GGCN(C)', 'SC-GGCN(GC)'))
# ax.set_ylabel('Collision Rewards', fontsize=12)
# ax.set_xlabel('Training Steps', fontsize=12)
ax.grid(which='both', ls='dashed')
# ax.set_ylim([0.8, 1])
ax.set_xlim([0, num_points-1])


#create legend for data lines
legend_elem = [Line2D([0], [0], color='k', linestyle='dashed', label='Train Accuracy'),
               Line2D([0], [0], color='k', linestyle='solid', label='Train Collision Rewards')]
legends = plt.legend(handles=legend_elem, loc='upper right', bbox_to_anchor=(0.55, 0.3))
# Add the legend manually to the current Axes.
plt.gca().add_artist(legends)

# create another legend for model GGCN
legend_elem = [Line2D([0], [0], color='green', linestyle='solid', label='GatedGCN'),
               Line2D([0], [0], color='orange', linestyle='solid', label='SC-GatedGCN(C)'),
               Line2D([0], [0], color='gray', linestyle='solid', label='SC-GatedGCN(GC)')]
legends = ax.legend(handles=legend_elem, loc='upper left', bbox_to_anchor=(0.58, 0.3))
plt.gca().add_artist(legends)

# # create another legend for model SC-GGCN(C)
# legend_elem = [Line2D([0], [0], color='k', linestyle='solid', label='SC-GGCN(C)')]
# legends = ax.legend(handles=legend_elem, loc='center', bbox_to_anchor=(0.65, 0.05))
# plt.gca().add_artist(legends)


fig.savefig('collision_rewards_over_training_steps.jpg', frameon=False, bbox_inches='tight')
    
    # break

