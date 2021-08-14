#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 15:16:21 2020

@author: dl-asoro
"""
import os
import sys
import yaml
import time
import math
import torch
import shutil

import pickle
import numpy as np
import matplotlib.pyplot as plt
from copy import copy
from IPython import get_ipython
from tqdm import tqdm
from misc import create_new_dir, setup_gpu, video_from_images
from evaluate import get_model, test_epoch, get_loader


#x_min, x_max is actually y_min y_max in eth and hotel
data_stats = {'eth': {'x_min': -7.69, 'x_max': 14.42, 'y_min': -3.17, 'y_max': 13.21},
              'hotel': {'x_min': -3.25, 'x_max': 4.35, 'y_min': -10.31, 'y_max': 4.31},
              'univ': {'x_min': -0.462, 'x_max': 15.469, 'y_min': -0.318, 'y_max': 13.892},
              'zara1': {'x_min': -0.14, 'x_max': 15.481, 'y_min': -0.375, 'y_max': 12.386},
              'zara2': {'x_min': -0.358, 'x_max': 15.558, 'y_min': -0.274, 'y_max': 13.943}
              }

def plot_path(obsv_traj, target_traj, pred_traj_list=None, ade_dict=None, top_k=1, ped_ids=None, data_min_max=None, 
              counter=0, frame=None, save_dir='./plots', legend=False, axis_off=False, fprefix=None, dtext=''):
    '''
    data_dict: data for one sequence/scene
    '''        
    if data_min_max is not None:
        pad=1
        x_min, x_max = data_min_max['x_min']-pad, data_min_max['x_max']+pad
        y_min, y_max = data_min_max['y_min']-pad, data_min_max['y_max']+pad
        
    #create canvass
    plt.close('all')
    w, h = 8, 5
    fig = plt.figure(frameon=False, figsize=(w, h))
    ax = plt.axes()
    if axis_off:
        ax.axis('off')
    fig.add_axes(ax)
    
    if frame is not None:
        if data_min_max is not None:
            extent = [x_min, x_max, y_min, y_max]
            ax.imshow(frame, aspect='auto', extent=extent) #extents = (left, right, bottom, top)
        else:
            print('Showing frame without extent, may not render properly. Provide x_min, x_max, y_min, y_max to define the extent of the frame')
            ax.imshow(frame, aspect='auto')
        
    if data_min_max is not None:
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
    
    num_peds = obsv_traj.shape[1]
    # cmap = np.random.uniform(0, 1, size=(num_peds+1, 3))
    cmap = plt.cm.get_cmap(name='Set1', lut=num_peds)
    # pred_cmap = plt.cm.get_cmap(name='Set3', lut=num_peds)
    legend_handles = []
    legend_labels = []
    
    for p in range(num_peds):
        color = cmap(p)
        lm = 'o'
        lw = 3
        ms = 6 # marker size
        mw = 1   # mamrker edge width
        # obsv tracks
        xs, ys = obsv_traj[:, p, 0], obsv_traj[:, p, 1]
        
        # start markers
        start_mark = ax.scatter(xs[0:1], ys[0:1], c=[color], label='start', marker=lm, edgecolors='k', s=lw**3, zorder=3)
        # plot obsv tracks
        obsv_line, = ax.plot(xs, ys, color=color, ls='solid', lw=lw, zorder=2, 
                             # marker=lm, markersize=ms, fillstyle='full', mfc='w', mec='k', mew=mw,
                             )
    
        #target tracks
        xs, ys = target_traj[:, p, 0], target_traj[:, p, 1]
        #plot target tracks
        target_line, = ax.plot(xs, ys, color=color, linestyle='solid', linewidth=lw, zorder=3,
                               # marker=lm, markersize=ms, fillstyle='full', mfc='cyan', mec='k', mew=mw,
                               )
        #end marker
        # end_mark = ax.scatter(xs[-1:], ys[-1:], color=[color], marker='*', s=24, zorder=2)
        ax.quiver(xs[-1], ys[-1], (xs[-1]-xs[-2])+1e-3, (ys[-1]-ys[-2])+1e-3, color=color,zorder=3, 
                        angles='xy', scale_units='xy', scale=1,
                        width=0.012*(ys[-1]-y_min)/(y_max-y_min),
                        # headwidth=3, headlength=4, headaxislength=3,
                        )
        
        if pred_traj_list is not None:
            if ade_dict is not None:
                #get the top k tracks using ade
                top_k_idx = np.argsort(ade_dict[p])[:top_k]
                top_k_preds = [pred_traj_list[idx][:, p, :] for idx in top_k_idx]
            else:
                top_k_preds = [pred_traj_list[idx][:, p, :] for idx in range(top_k)]
            
            for k in range(top_k):
                xs, ys = top_k_preds[k][:, 0], top_k_preds[k][:, 1]
                pred_line, = ax.plot(xs, ys, color=color, linestyle='--', linewidth=lw, zorder=1,
                                     # marker=lm, markersize=ms, fillstyle='full', mfc=color, mec='k', mew=mw,
                                     )
                ax.quiver(xs[-1], ys[-1], (xs[-1]-xs[-2])+1e-3, (ys[-1]-ys[-2])+1e-3, color=color, zorder=3,
                        angles='xy', scale_units='xy', scale=1,
                        width=0.012*(ys[-1]-y_min)/(y_max-y_min),
                        # headwidth=3, headlength=4, headaxislength=3,
                        )
        if ped_ids is not None:
            legend_handles.append(pred_line)
            legend_labels.append('{}:pid{}'.format(p, int(ped_ids[p][0])))
            
    if legend:
        legend_handles.extend([start_mark, target_line, pred_line])
        legend_labels.extend(['Start', 'GT', 'Pred'])
        
        ax.legend(legend_handles, legend_labels, handlelength=4)
        
    ax.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False,
                   labelbottom=False, labeltop=False, labelleft=False, labelright=False)
    plt.tight_layout()
    plt.text(0.05, 0.96, '%s, frame:%s'%(dtext, counter), transform=ax.transAxes, fontsize=16, color='yellow', va='top',)
    if save_dir is not None:
        if prefix is not None:
            file_path = save_dir + '{}_frame_{}.jpeg'.format(fprefix, counter)
        else:
            file_path = save_dir + 'frame_{}.jpeg'.format(counter)
        fig.savefig(file_path , bbox_inches='tight',dpi=300)
    # plt.show()
    return fig

# plot_path(obsv, target, pred_list, m_dict, top_k=top_k, data_min_max=data_stats[dset_name], counter=counter, frame=frame, save_dir=plot_dir, prefix=prefix, legend=True, ped_ids=None)
frames_to_plot_dic = {1: {'eth': [910, 1130, 8980, 12120, 12130, 12140, 12150, 12160, 12170, 12180, 12190, 12200, 12210, 12220, 12230, 12240, 12250, 12260, 12270],
                          'hotel':[],
                          'univ': [80, 90, 100, 110, 120, 130, 140, 150, 160, 180, 190, 200, 210, 220, 230, 240, 250, 260, 270, 280, 290, 300], 
                          'zara1':[],
                          'zara2':[6140, 6150, 6160, 6170, 6180, 6190, 6200, 6210, 6220, 6230, 6240, 6250, 6260, 6270, 6280, 6290, 6300, 6310,]
                  },
                  20: {'eth': [910, 1130, 8980, 12120, 12130, 12140, 12150, 12160, 12170, 12180, 12190, 12200, 12210, 12220, 12230, 12240, 12250, 12260, 12270],
                      'hotel': [],
                      'univ': [4520, 4530, 4540, 4550, 4560, 4570, 4580, 4890, 4600, 4610, 4620, 4630, 4640, 4650, 4660, 4670, 4680, 4690, 4700, 4710],
                      'zara1': [],
                      'zara2':[6140, 6150, 6160, 6170, 6180, 6190, 6200, 6210, 6220, 6230, 6240, 6250, 6260, 6270, 6280, 6290, 6300, 6310,]
                      }
                  }

peds_idx_dic = {1: {'eth':[],
                    'hotel':[],
                    'univ':[],
                    'zara1':[],
                    'zara2':[] },
            20: {'eth':[],
                 'hotel':[],
                 'univ':[0, 1, 2, 3, 4],
                 'zara1':[],
                 'zara2':[]}
            }


if __name__ == '__main__':
    #load model    
    out_dir = './out/SC_GCN/trial_16/'
    runs = [
        'run-10_hdim64_elayrs8_dlayrs2_zdim256_batch64_lr0.0003_kld0.1_edg0.0_mse1.0_cri0.0_gwt0.0_cwt0.0_center_scale_mse_adv*val1',
        'run-10_hdim64_elayrs8_dlayrs2_zdim256_batch64_lr0.0003_kld0.1_edg0.0_mse1.0_cri1.0_gwt0.0_cwt1.0_center_scale_mse_adv*val1',
        'run-10_hdim64_elayrs8_dlayrs2_zdim256_batch64_lr0.0003_kld0.1_edg0.0_mse1.0_cri1.0_gwt1.0_cwt0.0_gdist0.4_center_scale_mse_adv*val1',
        'run-10_hdim64_elayrs8_dlayrs2_zdim256_batch64_lr0.0003_kld0.1_edg0.0_mse1.0_cri1.0_gwt1.0_cwt1.0_gdist0.4_center_scale_mse_adv*val1'
        ]
    for run in runs:
        run_dir = out_dir + run + '/'
        print(run_dir)
        
        device = setup_gpu(gpu_id=0)
    
        with open(run_dir + 'args.pkl', 'rb') as f: 
            args = pickle.load(f)
            
        with open(run_dir + "%s.yaml"%args.model_name, 'r') as f:
            model_params = yaml.load(f, Loader = yaml.FullLoader)
        
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if device.type == 'cuda':
            torch.cuda.manual_seed(args.seed)

        phase='test'
        model_ckpt = 'best_test_fde'
        # model_ckpt = 'model_epoch_100'
        # model_ckpt = 'entire_model_100'
        
        num_samples = 20
        background = True
        top_k = 1 #number of samples to plot
        rank_metric = 'ade' #metric to rank traj
        
        if args.collision_rewards_wt>0 and args.goal_rewards_wt>0:
            mname = 'SC-GatedGCN(GC)'
        elif args.collision_rewards_wt>0:
            mname = 'SC-GatedGCN(C)'
        elif args.goal_rewards_wt>0:
            mname = 'SC-GatedGCN(G)'
        else:
            mname = 'GatedGCN'
        print('----------------------------')
        print('samples:', num_samples)
        print('top k:', top_k)
        print('rank metric:', rank_metric)
        print('model ckpt:', model_ckpt)
        print('model name:', mname)
        print('----------------------------')
        
        frames_to_plot = frames_to_plot_dic[1]
        peds_idx = peds_idx_dic[1]
            
        frames_list = []
        # datasets = ['eth', 'hotel', 'univ', 'zara1', 'zara2']
        datasets = ['eth']
        for dset_name in datasets:

            if not len(frames_to_plot[dset_name])>0:
                continue
        
            prefix = phase + '_batch_size%s'%args.batch_size + '_K%s'%num_samples
            raw_data_path = create_new_dir(run_dir + dset_name + '/eval_results/' + model_ckpt + '/' + prefix + '/')
            try:
                #load raw data dumped by evaluate.py
                print('Loading raw data....')
                with open(raw_data_path + 'raw_data_dict.pkl', 'rb') as f: 
                    ade, fde, raw_data_dict = pickle.load(f)
            except Exception as e:
                print(e)
                print('Evaluating model...')
                ckpt_path = run_dir + dset_name + '/' + model_ckpt + '.pth'
                model = get_model(args, model_params, ckpt_path, device)        
                data_dir = './datasets/{}/'.format(dset_name)
                dataloader = get_loader(args, data_dir, phase=phase)        
                ade, fde, raw_data_dict = test_epoch(args, model, dataloader, device, KSTEPS=num_samples)
                
                with open(raw_data_path + 'raw_data_dict.pkl', 'wb') as f:
                    pickle.dump([ade, fde, raw_data_dict], f)
                
            print('Plotting traj...', phase, dset_name)
            plot_dir = run_dir + dset_name + '/plot_traj/{}/{}_background{}_K{}_top{}_{}/'.format(phase, model_ckpt, background, num_samples, top_k, rank_metric)
            # plot_dir = './videos/' + mname + '/' + dset_name + '/' 
            if os.path.exists(plot_dir):
                shutil.rmtree(plot_dir)
            create_new_dir(plot_dir)
                
            frames_path = '../eth_ucy_frames/' + dset_name + '/frames/'
            prefix = 'gwt{}_cwt{}_{}_{}_K{}_top{}{}'.format(args.goal_rewards_wt, args.collision_rewards_wt, dset_name, model_ckpt, num_samples, top_k, rank_metric, )
            start = time.time()        
            step = 0
            unique_counter_list = []
            # pbar = tqdm(total=len(raw_data_dict), position=0)
            for k, raw_data in raw_data_dict.items():
        
                # obsv_frames = raw_dataunique_counter_list['obsv_frames'].cpu().numpy()
                obsv_traj = raw_data['obsv'].cpu().numpy() #[obs_len, K, 2]
                
                target_frames = raw_data['trgt_frames'].cpu().numpy()
                target_traj = raw_data['trgt'].cpu().numpy() #[pred_len, K, 2]
            
                pred_traj_list = raw_data['pred'] #list of sampled pred [pred_len, K, 2]
                pred_traj_list = [traj.cpu().numpy() for traj in pred_traj_list]
                
                ade_dict = raw_data['ade_dict']
                fde_dict = raw_data['fde_dict']
                seq_start_end = raw_data['seq_start_end']
                
                frames_list.append(np.unique(target_frames))
                for start, end in seq_start_end:
                    frame_num = int(target_frames[start:end, :][0][0])
                    
                    if frame_num not in frames_to_plot[dset_name]:
                        continue

                    counter = frame_num if not phase=='train' else step #frame_num cab be repeated in train data
                    if frame_num in unique_counter_list:
                        continue
                        # counter = str(frame_num) + '100' #to prevent overwriting image. This problem occur while testing univ, as it contains two sequence 
                    unique_counter_list.append(counter)
                    
                    ped_id = raw_data['ped_id'][start:end, :].numpy()

                    obsv = obsv_traj[:, start:end, :]
                    target = target_traj[:, start:end, :]
                    pred_list = [traj[:, start:end, :] for traj in pred_traj_list]

                    if rank_metric=='ade':
                        m_dict = {i:ade_dict[k] for i,k in enumerate(range(start, end))}
                    if rank_metric=='fde':
                        m_dict = {i:fde_dict[k] for i,k in enumerate(range(start, end))}
                        
                    ped_list = peds_idx[dset_name]
                    if len(ped_list)>0:
                        obsv = obsv[:, ped_list, :]
                        target = target[:, ped_list, :]
                        pred_list = [traj[:, ped_list, :] for traj in pred_list]
                        m_dict = {i:m_dict[p] for i, p in enumerate(ped_list)}
                
                    if dset_name=='eth' or dset_name=='hotel':
                        cx = (data_stats[dset_name]['x_max'] - abs(data_stats[dset_name]['x_min']))/2
                        cy = (data_stats[dset_name]['y_max'] - abs(data_stats[dset_name]['y_min']))/2
                        T = np.array([[cx, cy]])
                        alpha = 90 * math.pi / 180
                        M = np.array([[math.cos(alpha), -math.sin(alpha)], [math.sin(alpha), math.cos(alpha)]])
                        target = np.dot(target-T, M) + T
                        obsv = np.dot(obsv-T, M) + T
                        pred_list = [np.dot(traj-T, M)+T for traj in pred_list]
                    
                    # if dset_name=='univ': # this is used for second set of UNIV
                    #     cx = (data_stats[dset_name]['x_max'] - abs(data_stats[dset_name]['x_min']))/2
                    #     cy = (data_stats[dset_name]['y_max'] - abs(data_stats[dset_name]['y_min']))/2
                    #     T = np.array([[cx, cy]])
                    #     alpha = -90 * math.pi / 180
                    #     M = np.array([[math.cos(alpha), -math.sin(alpha)], [math.sin(alpha), math.cos(alpha)]])
                    #     target = np.dot(target-T, M) + T
                    #     obsv = np.dot(obsv-T, M) + T
                    #     pred_list = [np.dot(traj-T, M)+T for traj in pred_list]
                    
                        
                    # try:
                    frame = plt.imread(frames_path + 'frame{}.png'.format(frame_num)) if background else None
                    plot_path(obsv, target, pred_list, m_dict, top_k=top_k, data_min_max=data_stats[dset_name], 
                              counter=counter, frame=frame, save_dir=plot_dir, fprefix=prefix, legend=False, ped_ids=None, dtext=mname)

                    # sys.exit()
                    step+=1
                    
            # pbar.close()
            print('Finished plotting {} traj in {:.6f}s'.format(dset_name, time.time()-start))
            # get_ipython().run_line_magic('matplotlib', 'qt5')
            # video part
            video_from_images(img_dir=plot_dir, dest_dir='./videos', fname='./videos/{}_{}_K{}_top{}_{}.mp4'.format(mname, dset_name, num_samples, top_k, rank_metric), frame_rate=2.5)


                


