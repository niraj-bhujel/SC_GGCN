#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 11:29:58 2020

@author: dl-asoro
"""


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import csv
import time
import copy
import json
import yaml
import shutil
import pickle
import random
import traceback
import numpy as np
from datetime import datetime
from collections import defaultdict


from early_stopping import EarlyStopping

from misc import *
from utils import TrajectoryDataset, MEAN_, STD_
from metrics import final_l2, average_l2, pairwise_euclidean_dist
from config import parse_argument


import torch
import torch.optim as optim
import torch.distributions.multivariate_normal as torchdist

from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from model import gnn_model

epsilon = 1e-20

def standardize_output(rel_vel, center, scale):
    assert rel_vel.shape[2]==2, 'Channel dims must be 2, but got {}'.format(rel_vel.shape[2])
    if center:
        rel_vel[:, :, 0] += MEAN_[0]
        rel_vel[:, :, 1] += MEAN_[1]
        
    if scale:
        rel_vel[:, :, 0] *= STD_[0]
        rel_vel[:, :, 1] *= STD_[1]
        
    return rel_vel
                    
def rel_to_abs(rel_vel, init_pos):
    '''
    Parameters
    ----------
    rel_vel : relative velocity of pedestrian
        Description. [seq_len, num_peds, 2]  
    init_pos : initial pos of the pedestrians [num_peds, 2]

    Returns
    -------
    abs_pos: [seq_len, num_peds, 2] 
    absolute position of the pedestrians
    '''
    abs_pos = torch.zeros_like(rel_vel).to(rel_vel.device)
    
    for s in range(rel_vel.shape[0]):
        # x_t = x0 + displacement_t 
        abs_pos[s, :, :] = torch.sum(rel_vel[:s+1, :, :], dim=0)  + init_pos
        
    return abs_pos
            
def organize_tensor(tensor, data_format, channel_dim=None, reshape=False, time_major=False):

    '''
    reshape tensor first and convert to time major if reshape flag=True and time_major=True
    Tensor can be converted to time_major directly without reshaping as well (reshape=False, time_major=True).  
    '''
    if reshape:
        if not channel_dim:
            raise Exception('channel_dim must be provided to reshape')
            
    if data_format=='channel_first':
        channel_axis=1
        time_axis = 2
        if reshape:
            tensor = tensor.view(tensor.shape[0], channel_dim, -1)
        
    elif data_format =='channel_last':
        channel_axis = 2
        time_axis = 1
        if reshape:
            tensor = tensor.view(tensor.shape[0], -1, channel_dim)
    else:
        raise Exception('Unknown data format')
        
    if time_major:
        tensor = tensor.permute(time_axis, 0, channel_axis)
        
    return tensor


def track_ade_fde(target_pos, sampled_traj):
    '''
    Parameters
    (x, y) are considered as features
    ----------
    target_pos : Tensor array of shape [pred_len, num_peds, 2] 
        DESCRIPTION. ground truth relative velocity of the target
    sampled_traj : list of sampled numpy array of  [[pred_len, num_peds, 2]...]
        DESCRIPTION. predicted absolute pos of the target
    Returns
    -------
    final_ade : float
    final_fde : float
    '''
    
    ade_dict = defaultdict(list)
    fde_dict = defaultdict(list)

    for pred_pos in sampled_traj:
        
        for n in range(pred_pos.shape[1]):
            n_pred =  pred_pos[:, n, :] #[pred_len, 2]        
            n_target = target_pos[:, n, :] #[pred_len, 2]
            
            ade_dict[n].append(average_l2(n_pred, n_target))
            fde_dict[n].append(final_l2(n_pred, n_target))
    
    #take the minimum val
    final_ade = np.mean([np.min(val) for k, val in ade_dict.items()])
    final_fde = np.mean([np.min(val) for k, val in fde_dict.items()])

    return final_ade, final_fde


def edge_loss(y_true, y_pred):
    '''
    pred_edge : num_edges x 12
    target_edge : num_edges x 12
    '''
    return torch.mean(torch.abs(y_true-y_pred))


def collect_collision_rewards(y_pred, y_ids, y_frames, collision_thresh=0.2, verbose=0):
    '''
    y_pred: sampled pos [pred_len, num_peds, 2]
    frames: frame numbers for each peds [num_peds, pred_len]
    ped_id: array containing id of each sample [num_peds, 1]
    '''

    if verbose>0:
        print('collecting collision rewards for %d samples'%y_pred.shape[1])
        start=time.time()
        
    y_ids = y_ids.cpu().numpy().flatten()
    y_frames = y_frames.cpu().numpy()
    
    collision_peds = []
    for t in range(12):
        
        paths = y_pred[t, :, :]
        # peds_dist = torch.cdist(paths, paths)
        peds_dist = pairwise_euclidean_dist(paths, paths)
        collision_peds_t = (peds_dist<=2*collision_thresh).type_as(peds_dist) * peds_dist # >0 mean collision, otherwise non-collision
        collision_peds.append(collision_peds_t)
    
    collision_peds = torch.stack(collision_peds, dim=0).sum(dim=0)

    #NOTE: By default same pedestrian will have collisions, as their distance is less than 0.2
    # Also multiple samples/row may have same ids, so need to flag of these pedestrian from the collision_matrix, 
    # Once the mask is computed, invert it to flag off those similar pedestrian and
    peds_mask = np.array([id_ == y_ids for id_ in y_ids]) #1-same peds, 0-different peds
    collision_peds *= torch.from_numpy(~peds_mask).type_as(collision_peds) #same peds will not have collision
    
    #NOTE! Collision cannot occur in different frames. ONLY frame at same time step is considered. 
    frames_mask = np.array([np.any(f == y_frames, axis=1) for f in y_frames]) #1-same frames, 0-different frames
    collision_peds *= torch.from_numpy(frames_mask).type_as(collision_peds) #only same frames will have collision i.e 1
    
    collision_rewards = collision_peds.sum(dim=1, keepdim=True).clone()
    
    for i in range(collision_rewards.numel()):
        if collision_rewards[i]>0:
            collision_rewards[i]=0
        else:
            collision_rewards[i]=1
    
    num_collisions = torch.sum(collision_peds>0).type(torch.float)*0.5
    if verbose>0:
        print('%f secs, samples: %d, collisions: %d, rewards:%f'%(time.time()-start, y_pred.shape[1], 
                                                                   num_collisions, 
                                                                   collision_rewards.mean()
                                                                   ))

    return collision_rewards, num_collisions
  
def collect_goal_rewards(y_true, y_pred, dist_thresh=0.2):
    
    assert y_true.shape==y_pred.shape
    
    goal_dist = torch.sqrt((y_true[-1, :, :] - y_pred[-1, :, :]).pow(2).sum(1, keepdim=True)) #[num_peds, 1]

    # goal_rewards = torch.zeros_like(goal_dist).to(goal_dist.device)
    goal_rewards = goal_dist.clone()
    
    for i in range(goal_rewards.numel()):
        if goal_dist[i]<dist_thresh:
            goal_rewards[i] = 1
        else:
            goal_rewards[i] = 0
    
    return goal_rewards, goal_dist
#%%
def test_epoch(args, model, dataloader, device, KSTEPS=20, history=None):
    
    model.eval()
    
    hist = defaultdict(list) if not history else history  
    raw_data_dict = defaultdict(dict)

    epoch_ade = 0
    epoch_fde = 0
    
    with torch.no_grad():
        for iter, (obsv_graph, target_graph) in enumerate(dataloader): 
            obsv_p = obsv_graph.ndata['pos'].to(device)
            obsv_v = obsv_graph.ndata['vel'].to(device)  # [K, obsv_len, 2]
            obsv_e = obsv_graph.edata['dist'].to(device)    #[K, obsv_len]
            
            target_p = target_graph.ndata['pos'].to(device)
            
            #reshape
            obsv_p = organize_tensor(obsv_p, args.data_format, time_major=True) #[obsv_len, num_peds, 2]
            obsv_v = obsv_v.view(-1, obsv_v.shape[1]*obsv_v.shape[2]) #[K, obsv_len*2]
    
            #predict k samples
            pred_pos_list = []
            for k in range(KSTEPS):
                #predict
                logits_v, logits_e, _, _  = model(obsv_graph, obsv_v, obsv_e, device=device)
            
                logits_v = organize_tensor(logits_v, args.data_format, 2, reshape=True, time_major=True)
                logits_v = standardize_output(logits_v, args.center, args.scale)
                pred_pos = rel_to_abs(logits_v, obsv_p[-1, :, :])
                pred_pos_list.append(pred_pos.cpu().numpy())
                
            target_pos = organize_tensor(target_p, args.data_format, reshape=False, time_major=True).cpu().numpy()
            ade, fde = track_ade_fde(target_pos, pred_pos_list) 
            
            epoch_ade += ade
            epoch_fde += fde
        
    hist['test_ade'].append(epoch_ade/(iter+1))
    hist['test_fde'].append(epoch_fde/(iter+1))
    
    return hist, raw_data_dict
    
    
def eval_epoch(args, model, dataloader, device, history=None):
    
    model.eval()
    
    hist = defaultdict(list) if not history else history  
    
    epoch_loss = 0
    epoch_log_loss = 0
    epoch_mse_loss = 0
    epoch_edge_loss = 0
    criterion = torch.nn.MSELoss()
    with torch.no_grad():
        for iter, (obsv_graph, target_graph) in enumerate(dataloader):
            # ped_id = obsv_graph.ndata['pid'].to(device).long()
            # obsv_p = obsv_graph.ndata['pos'].to(device)
            obsv_v = obsv_graph.ndata['vel'].to(device)  
            obsv_e = obsv_graph.edata['dist'].to(device)    
            
            # target_p = target_graph.ndata['pos'].to(device) 
            target_v = target_graph.ndata['vel'].to(device) 
            target_e = target_graph.edata['dist'].to(device)
            
            #reshape inputs
            obsv_v = obsv_v.view(-1, obsv_v.shape[1]*obsv_v.shape[2]) #[K, 2*obsv_len]
            target_v = target_v.view(-1, target_v.shape[1]*target_v.shape[2])#[K, 2*pred_len] 
            
            #predict
            logits_v, logits_e, _, _= model(obsv_graph, obsv_v, obsv_e, gy=target_graph, yy=target_v, ey=target_e, device=device)
            
            loss = 0
            #gaussian log loss
            if args.log_loss_wt>0:
                log_loss = bivariate_loss(target_v, logits_v, data_format=args.data_format)
                epoch_log_loss+= log_loss.detach().item()
                loss += log_loss
            
            if args.edge_loss_wt>0:
                edge_loss = criterion(target_e - logits_e)
                epoch_edge_loss += edge_loss.detach().item()
                loss += edge_loss
                
            #mse loss
            if args.mse_loss_wt>0:
                mse_loss = criterion(target_v, logits_v)
                epoch_mse_loss += mse_loss.detach().item()
                loss += mse_loss
                
            epoch_loss += loss.item()
            
    hist['val_loss'].append(epoch_loss/(iter+1))
    hist['val_log_loss'].append(epoch_log_loss/(iter+1))
    hist['val_edge_loss'].append(epoch_edge_loss/(iter+1))
    hist['val_mse_loss'].append(epoch_mse_loss/(iter+1))
    
    return hist

def train_epoch(args, model, optimizer, dataloader, device, epoch, history=None):
    
    model.train()
        
    hist = defaultdict(list) if not history else history  
    
    epoch_loss = 0
    epoch_kld_loss = 0
    epoch_log_loss = 0
    epoch_mse_loss = 0
    epoch_edge_loss = 0
    epoch_est_values = 0
    epoch_critic_loss = 0
    epoch_goal_rewards = 0
    epoch_goal_dist = 0
    epoch_collision_rewards = 0
    epoch_collisions = 0
    epoch_rewards = 0
    eporch_rewards_loss = 0
    epoch_grad_norm = 0
    
    epoch_gt_collision_rewards = 0
    epoch_gt_collisions = 0
    
    criterion = torch.nn.MSELoss()
    
    for iter, (obsv_graph, target_graph) in enumerate(dataloader):
        # print("iter", iter)
        ped_id = obsv_graph.ndata['pid'].to(device)
        obsv_p = obsv_graph.ndata['pos'].to(device)  
        obsv_v = obsv_graph.ndata['vel'].to(device)  
        obsv_e = obsv_graph.edata['dist'].to(device)   
        
        target_p = target_graph.ndata['pos'].to(device) 
        target_v = target_graph.ndata['vel'].to(device) 
        target_e = target_graph.edata['dist'].to(device) 
        target_f = target_graph.ndata['frames'].to(device)
        
        #reshape inputs
        obsv_v = obsv_v.view(-1, obsv_v.shape[1]*obsv_v.shape[2]) #[K, obsv_len*2]
        target_v = target_v.view(-1, target_v.shape[1]*target_v.shape[2])#[K, pred_len*2] 
        
        #predict
        logits_v, logits_e, values, KLD = model(obsv_graph, obsv_v, obsv_e, gy=target_graph, yy=target_v, ey=target_e, device=device)
        
        epoch_kld_loss += KLD.detach().item()
        loss = KLD*args.kld_loss_wt
        
        if args.edge_loss_wt>0:
            edge_loss = criterion(target_e - logits_e)
            epoch_edge_loss += edge_loss.detach().item()
            loss += edge_loss * args.edge_loss_wt
            
        #NOTE: mse loss works only with channel last forma_fde:0.52873, traint
        if args.mse_loss_wt>0:
            mse_loss = criterion(target_v, logits_v) 
            epoch_mse_loss += mse_loss.detach().item()
            loss += mse_loss * args.mse_loss_wt
            
        #relative to abs
        pred_vel = organize_tensor(logits_v, args.data_format, 2, reshape=True, time_major=True)
        pred_vel = standardize_output(pred_vel.clone(), args.center, args.scale)
        obsv_pos = organize_tensor(obsv_p, args.data_format, time_major=True) #[obsv_len, num_peds, 2]
        pred_pos = rel_to_abs(pred_vel, obsv_pos[-1, :, :])
        
        goal_rewards = 0
        collision_rewards = 0
        if args.goal_rewards_wt>0:
            #compute goal rewards
            target_pos = organize_tensor(target_p, args.data_format, reshape=False, time_major=True)
            goal_rewards, goal_dist = collect_goal_rewards(target_pos, pred_pos, dist_thresh=args.gdist_thresh)
            epoch_goal_dist += goal_dist.mean().detach().item()
            epoch_goal_rewards += goal_rewards.mean().detach().item()
        
        if args.collision_rewards_wt>0:
            #compute gt_collision
            # gt_collision_rewards, num_gt_collisions = collect_collision_rewards(target_pos, ped_id, target_f, args.collision_thresh)
            #compute collision rewards
            collision_rewards, num_collisions = collect_collision_rewards(pred_pos, ped_id, target_f, args.collision_thresh)
            epoch_collision_rewards += collision_rewards.mean().detach().item()
            epoch_collisions += num_collisions.item()
        
        rewards = args.goal_rewards_wt*goal_rewards + args.collision_rewards_wt*collision_rewards
        
        if args.critic_loss_wt>0:
            epoch_est_values += values.mean().detach().item()
            #critic loss
            critic_loss = criterion(rewards , values) * values.mean()#*(rewards-values).mean()
            epoch_critic_loss += critic_loss.item()
            loss += critic_loss * args.critic_loss_wt
                
            if args.goal_rewards_wt>0 or args.collision_rewards_wt>0:
                value_loss = criterion(values, rewards)
            
                #zero grad
                optimizer.zero_grad()
                #freeze all excepts critic
                for param_name, param in model.named_parameters():
                    if 'critic' not in param_name:
                        param.requires_grad=False
                    
                #compute gradients
                value_loss.backward(retain_graph=True)
                #update critic with value_loss
                optimizer.step()
    
                #unfreeze all
                for param in model.parameters():
                    param.requires_grad=True

            #freeze critic if critic_loss_wt>0
            for param in model.critic.parameters():
                param.requires_grad=False
                
        #zero grad
        optimizer.zero_grad()
        #compute gradients
        loss.backward(retain_graph=True)
        epoch_loss += loss.detach().item()        
        
        if args.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        # parameters = list(filter(lambda p: p.grad is not None, model.parameters()))
        # total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach().to(device), 2) for p in parameters]), 2)
        # epoch_grad_norm += total_norm.cpu().numpy()            
        optimizer.step()
        
        #unfreeze  all (critic specially)
        for param in model.parameters():
            param.requires_grad=True        

            
        
    
    hist['lr'].append(optimizer.param_groups[0]['lr'])
    hist['grad_norm'].append(epoch_grad_norm/(iter+1))
    hist['train_loss'].append(epoch_loss/(iter+1))
    hist['train_kld_loss'].append(epoch_kld_loss/(iter+1))
    # hist['train_log_loss'].append(epoch_log_loss/(iter+1))
    # hist['train_edge_loss'].append(epoch_edge_loss/(iter+1))
    hist['train_mse_loss'].append(epoch_mse_loss/(iter+1))
    
    hist['train_est_values'].append(epoch_est_values/(iter+1))
    hist['train_critic_loss'].append(epoch_critic_loss/(iter+1))
    hist['train_goal_dist'].append(epoch_goal_dist/(iter+1))
    hist['train_goal_rewards'].append(epoch_goal_rewards/(iter+1))
    hist['train_collisions'].append(epoch_collisions/(iter+1))
    hist['train_collision_rewards'].append(epoch_collision_rewards/(iter+1))
    
    return hist

def run(args, model_params, device):
    ############# Prepare dir #########
    if args.edge_loss_wt>0:
        model_params['past_dec']['mlp_readout_edge']=True
        model_params['critic']['mlp_readout_edge']=True
    
    args.z_dim = model_params['z_dim']
    args.hidden_dim = model_params['hidden_dim'] 
    args.enc_layers = model_params['enc_layers']
    args.dec_layers = model_params['dec_layers']
            

    trial = 'trial_{}/'.format(args.trial)
    run = 'run-{}_hdim{}_elayrs{}_dlayrs{}_zdim{}_batch{}_lr{}_kld{}_edg{}_mse{}_cri{}_gwt{}_cwt{}'.format(
        args.run, model_params['hidden_dim'], model_params['enc_layers'], model_params['dec_layers'], model_params['z_dim'],
        args.batch_size, args.lr, args.kld_loss_wt, args.edge_loss_wt, args.mse_loss_wt, args.critic_loss_wt,
        args.goal_rewards_wt, args.collision_rewards_wt)
    if args.goal_rewards_wt>0:
        run += '_gdist{}'.format(args.gdist_thresh)
    if args.grad_clip is not None:
        run += '_clip{}'.format(args.grad_clip)
    if args.center:
        run += '_center'
    if args.scale:
        run += '_scale'
    if args.in_feat_dropout>0:
        run += '_indrop{}'.format(args.in_feat_dropout)
    if args.pos_enc:
        run += '_penc{}_pdim{}'.format( args.pos_enc, args.pos_enc_dim)
    
    if len(args.prefix)>0:
        run = run + '_{}'.format(args.prefix)
        
    args.out_dir = './out/' + args.model_name + '/'
    args.out_path = args.out_dir + trial + run + '/' + args.dataset + '/'

    print('Starting trial_{}, {}'.format(args.trial, args.out_path))
    
    if os.path.exists(args.out_path):
        if args.overwrite:
            print('Run exists! Overwritting existing run...')
        else:
            raise Exception('Run exists! Continuing to next run...')
            return None            
    else:
        create_new_dir(args.out_path)

    #backup
    # shutil.copy(__file__, os.path.join(args.out_path, os.path.basename(__file__)))    
    backup_files = ['model.py', 'train.py', 'evaluate.py', 'utils.py', 'metrics.py', 
                    'config.py', 'early_stopping.py', 'misc.py', '%s.yaml'%args.model_name]
    for file in backup_files:
        dest_fpath = os.path.join(args.out_dir, trial, run, file)
        if os.path.exists(dest_fpath):
            if args.overwrite:
                shutil.copy(file, dest_fpath)
        else:
            shutil.copy(file, dest_fpath)
                
    with open(args.out_dir + trial + run + '/' + 'args.pkl', 'wb') as fp:
        pickle.dump(args, fp)
        
    with open(args.out_dir + trial + run + '/' + 'model_params.pkl', 'wb') as fp:
        pickle.dump(model_params, fp)
        
    args.summary_dir = create_new_dir(args.out_path + 'summary/')
    #verify log_interval is less than epochs
    args.log_interval = min(args.log_interval, args.epochs)
    
    if args.resume_training:
        assert args.epochs>args.resume_epoch, 'Cannot perform training. Make sure epochs is greater than resume epoch'
        
    print('Preparing datasets...', args.dataset)
    data_dir = './datasets/'+args.dataset+'/'
    datasets = {phase: TrajectoryDataset(data_dir, obs_len=args.obs_seq_len, pred_len=args.pred_seq_len, 
                                         phase=phase, preprocess=args.preprocess) for phase in ['train', 'val', 'test']}
    #shift to origin and scale
    for phase in ['train', 'val', 'test']:
        datasets[phase]._standardize_inputs(args.center, args.scale, args.data_format)
        
        #add positional encoding
        if args.pos_enc:
            datasets[phase]._add_positional_encodings(args.pos_enc_dim)
        
    dataloaders = {phase:DataLoader(datasets[phase], batch_size=int(args.batch_size),
                                    shuffle=True if phase=='train' else False,
                                    collate_fn=datasets[phase].collate) for phase in ['train', 'val', 'test']}
    for phase in ['train', 'val', 'test']:
        print('{} dataset: {} samples, {} batches'.format(phase, len(datasets[phase]), len(dataloaders[phase])))
        
    #create model            
    print("Creating model ....")
    model = gnn_model(args.model_name, model_params, args)
    model = model.double().to(device)
    args.num_parameters = model_parameters(model, verbose=0)
    for m_name, module in model.named_children():
        print(m_name, model_attributes(module, verbose=0), '\n')
    
    #optimizer
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)

    #lr scheculer
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=args.lr_reduce_factor,
                                                     patience=args.lr_patience, verbose=True)

    # early_stopping
    early_stopping = EarlyStopping(patience=args.early_stop_patience, metric_name=args.early_stop_metric,
                                   ckpt_path=args.out_path)
        
    #clear summary events
    if not args.resume_training:
        for filename in os.listdir(args.summary_dir):
            os.unlink(args.summary_dir + filename)
    writer = SummaryWriter(log_dir=args.summary_dir)
    
    h = defaultdict(list)
    
    if args.resume_training:
        print('Loading state from pretrained model...')
        model, optimizer, scheduler = load_ckpt(model, optimizer, scheduler, args.resume_epoch, args.out_path)
        args.initial_epoch = args.resume_epoch+1
        with open(args.out_path +'history.pkl', 'rb') as fp:
            h = pickle.load(fp)
    try:
        for epoch in range(args.initial_epoch, args.epochs+1, 1):
            start = time.time()      
            print('\nEpoch {}/{}'.format(epoch, args.epochs))
            h = train_epoch(args, model, optimizer, dataloaders['train'], device, epoch, history=h)
    
            h = eval_epoch(args, model, dataloaders['val'], device, history=h)
    
            if epoch%args.test_interval==0 or epoch==1:
                h, raw_data_dict = test_epoch(args, model, dataloaders['test'], device, KSTEPS=args.k_samples, history=h)
                early_stopping(h[args.early_stop_metric][-1], model)
            else:
                h['test_ade'].append(h['test_ade'][-1])
                h['test_fde'].append(h['test_fde'][-1])
            
            scheduler.step(h[args.lr_scheduler_metric][-1])
            
            h['epoch'].append(epoch)
            h['eta'].append((time.time()-start))
            
            #update every epoch
            for key, val in h.items():
                if 'de' not in key:
                    writer.add_scalar(key, val[-1], epoch)
                #write ade, fde every test interveal only
                elif (epoch)%args.test_interval==0 or epoch==1: 
                    writer.add_scalar(key, val[-1], epoch)
    
                if np.isnan(val[-1]):
                    raise Exception('nan value in {}'.format(key))
                    break
                
            print(', '.join(['{}:{:.5f}'.format(k, v[-1]) for k, v in sorted(h.items())]))
            
            if early_stopping.early_stop:
                print("Early stopping at epoch {}".format(epoch))
                break
    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early because of KeyboardInterrupt')
        
    writer.add_scalar('paramaters', args.num_parameters, 0)
    writer.close()
    
    print('Logging results...')
    save_history(h, args.out_path)
    
    with open(args.out_path +'history.pkl', 'wb') as fp:
        pickle.dump(h, fp)
        
    with open(args.out_dir + trial + run + '/test_results.txt', 'a+') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(['{}, Model:{}, Dataset:{}, K:{}, ADE:{}, FDE:{}, MinADE:{}, MinFDE:{}'.format(
            datetime.now(), args.model_name, args.dataset, args.k_samples, h['test_ade'][-1], h['test_fde'][-1], 
            np.min(h['test_ade']), np.min(h['test_fde']) )])
    
    #save metrics to csv
    args.final_epoch = epoch
    params_to_save = ['model_name', 'num_parameters', 'dataset', 'trial', 'run', 'batch_size',
                      'z_dim', 'hidden_dim', 'enc_layers', 'dec_layers', 'kld_loss_wt', 'mse_loss_wt', 'critic_loss_wt', 
                      'gdist_thresh', 'collision_thresh', 'goal_rewards_wt', 'collision_rewards_wt',
                      'lr', 'grad_clip', 'center', 'scale', 'pos_enc', 'epochs', 'final_epoch']
    metrics_to_save = ['train_loss', 'val_loss', 'test_ade', 'test_fde', 'min_test_ade', 'min_test_fde']
    
    h['min_test_ade'].append(np.min(h['test_ade']))
    h['min_test_fde'].append(np.min(h['test_fde']))
    
    row_data = [vars(args)[k] for k in params_to_save] + [h[k][-1] for k in metrics_to_save]
    
    log_details = {k:v for k, v in zip(params_to_save+metrics_to_save, row_data)}
    with open(args.out_path + 'log_results.json', 'w') as f:
        json.dump(log_details, f)
    
    save_to_excel(args.out_dir + 'log_results_trial_{}.xlsx'.format(args.trial), 
                  row_data, header=params_to_save+metrics_to_save)
    
    #save final models
    save_ckpt(model, optimizer, scheduler, epoch, args.out_path)   
    torch.save(model, args.out_path + 'entire_model_{}.pth'.format(epoch))
    
    return model, dataloaders, optimizer, h

    
if __name__=='__main__':
    
    args = parse_argument()
    # print(vars(args))
    device = setup_gpu(args.gpu_id, memory=args.gpu_memory)

    with open("./%s.yaml"%args.model_name, 'r') as file:
        model_params = yaml.load(file, Loader = yaml.FullLoader)
        
    #setup seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(args.seed)
        
    if args.mode=='search':
        for i in range(200):
            model_params['z_dim'] = int(np.random.choice([32, 64, 128, 256]))
            model_params['hidden_dim'] = int(np.random.choice([32, 64, 128, 256]))
            model_params['enc_layers'] = int(np.random.choice([1, 2, 4, 8]))
            model_params['dec_layers'] = int(np.random.choice([1, 2, 4, 8,]))

            if model_params['hidden_dim']==256 and model_params['enc_layers']>4:
                continue
            
            args.batch_size = int(np.random.choice([16, 32, 64, 128, 256]))
            
            # for args.lr in [0.0005, 0.0001, 0.00005]:
            # all_datasets = 'eth, hotel, univ, zara1, zara2'
            all_datasets = 'eth, univ'
            for args.dataset in all_datasets.split(', '):
                try:
                    result = run(args, model_params, device)
                except KeyboardInterrupt:
                    print('Exiting from training early because of KeyboardInterrupt')
                    sys.exit()
                except RuntimeError: #RuntimeError: CUDA out of memory.
                    traceback.print_exc()
                    break #if CUDA out of memory, no need to run with this learning rate again
                except ValueError: #something wrong with the model
                    traceback.print_exc()
                    break
                except Exception:
                    traceback.print_exc()
                    continue
                
    if args.mode=='train':
        if args.dataset=='all':
            for args.dataset in 'eth, hotel, univ, zara1, zara2'.split(', '):
                try:
                    result = run(args, model_params, device)
                except KeyboardInterrupt:
                    print('Exiting from training early because of KeyboardInterrupt')
                    sys.exit()
                except Exception:
                    traceback.print_exc()
                    continue
        else:
            result = run(args, model_params, device)



    