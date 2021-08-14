#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 15:16:21 2020

@author: dl-asoro
"""
import os
import csv
import sys
import pickle
import yaml
import time
import numpy as np
from collections import defaultdict
from misc import setup_gpu, create_new_dir
from utils import TrajectoryDataset
from evaluate import test_epoch, get_model, get_loader

import torch
from torch.utils.data import DataLoader


def collision_torch(y_pred, y_ids=None, y_frames=None, collision_thresh=0.2, verbose=0):
    '''
    y_pred: sampled pos [pred_len, num_peds, 2]
    frames: frame numbers for each peds [num_peds, pred_len]
    ped_id: array containing id of each sample [num_peds, 1]
    '''
    def pairwise_euclidean_dist(x, y):
        """
        Args:
          x: pytorch Variable, with shape [m, d]
          y: pytorch Variable, with shape [n, d]
        Returns:
          dist: pytorch Variable, with shape [m, n]
        """
        m, n = x.size(0), y.size(0)
        xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
        yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
        dist = xx + yy
        dist.addmm_(1, -2, x, y.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        return dist

    if verbose>0:
        print('collecting collision rewards for %d samples'%y_pred.shape[1])
        start=time.time()
    
    collision_peds = []
    for t in range(y_pred.shape[0]):
        paths = y_pred[t, :, :]
        # peds_dist = torch.cdist(paths, paths)
        peds_dist = pairwise_euclidean_dist(paths, paths)
        collision_peds_t = (peds_dist<=2*collision_thresh).type_as(peds_dist) # >0 mean collision, otherwise non-collision
        # collision_peds_t[collision_peds_t>1e-3]=1
        collision_peds.append(collision_peds_t)
    collision_matrix_unmasked = torch.stack(collision_peds, dim=0)
    collision_peds = torch.stack(collision_peds, dim=0).sum(dim=0)
    #NOTE: By default same pedestrian will have collisions, as their distance is less than 0.2
    # Also multiple samples/row may have same ids, so need to flag of these pedestrian from the collision_matrix, 
    # Once the mask is computed, invert it to flag off those similar pedestrian and
    if y_ids is not None:
        y_ids = y_ids.cpu().numpy().flatten()
        peds_mask = np.array([id_ == y_ids for id_ in y_ids]) #1-same peds, 0-different peds
        collision_peds *= torch.from_numpy(~peds_mask).type_as(collision_peds).to(collision_peds.device) #same peds will not have collision    
    #NOTE! Collision cannot occur in different frames. ONLY frame at same time step is considered. 
    if y_frames is not None:
        y_frames = y_frames.cpu().numpy()
        frames_mask = np.array([np.any(f == y_frames, axis=1) for f in y_frames]) #1-same frames, 0-different frames
        collision_peds *= torch.from_numpy(frames_mask).type_as(collision_peds).to(collision_peds.device) #only same frames will have collision i.e 1
    
    num_collisions = torch.sum(collision_peds>0).type(torch.float)*0.5
    if verbose>0:
        print('%f secs, samples: %d, collisions: %d'%(time.time()-start, y_pred.shape[1], 
                                                                   num_collisions, 
                                                                   ))
        
    return num_collisions.item(), collision_peds, collision_matrix_unmasked.cpu().numpy()

def collision(path1, path2, frames1=None, frames2=None, n_predictions=12, person_radius=0.1, inter_parts=2):
    """Check if there is collision or not
    Parameters
    ----------
    path1 : array of [T, 2] 
    path2 : Tarray of [T, 2]     
    """

    assert len(path1) >= n_predictions
    
    if frames1 is not None:
        if frames2 is not None:
            common_frames = np.intersect1d(frames1, frames2)
        
            if common_frames.size==0:
                return False

            path1 = np.array([path1[i] for i in range(len(path1)) if frames1[i] in common_frames])
            path2 = np.array([path2[i] for i in range(len(path2)) if frames2[i] in common_frames])
    
    def getinsidepoints(p1, p2, parts=2):
        """return: equally distanced points between starting and ending "control" points"""

        return np.array((np.linspace(p1[0], p2[0], parts + 1),
                         np.linspace(p1[1], p2[1], parts + 1))) #[2, parts+1]
    num_collisions = 0
    for i in range(len(path1) - 1):
        p1, p2 = [path1[i][0], path1[i][1]], [path1[i + 1]                #         path1 = target_p[:, i, :].cpu().numpy()
                #         path2 = target_p[:, j, :].cpu().numpy()
                #         num_gt_collisions += 2*collision(path1, path2, person_radius=collision_thresh, inter_parts=1)[0], path1[i + 1][1]] #[(x1, y1), (x2, y2)]
        p3, p4 = [path2[i][0], path2[i][1]], [path2[i + 1][0], path2[i + 1][1]] #[(x1, y1), (x2, y2)]
        
        inside_points1 = getinsidepoints(p1, p2, inter_parts)
        inside_points2 = getinsidepoints(p3, p4, inter_parts)
        
        inter_parts_dist = np.linalg.norm(inside_points1 - inside_points2, axis=0)
        
        if np.min(inter_parts_dist) <= 2 * person_radius:
            num_collisions+=1
            return num_collisions
        
    return num_collisions

if __name__=='__main__':
    device = setup_gpu(gpu_id=0)
    #load model
    out_dir = './out/SC_GCN/trial_16/'
    
    run = 'run-10_hdim64_elayrs8_dlayrs2_zdim256_batch64_lr0.0003_kld0.1_edg0.0_mse1.0_cri0.0_gwt0.0_cwt0.0_center_scale_mse_adv*val1'
    # run = 'run-10_hdim64_elayrs8_dlayrs2_zdim256_batch64_lr0.0003_kld0.1_edg0.0_mse1.0_cri1.0_gwt0.0_cwt1.0_center_scale_mse_adv*val1'
    # run = 'run-10_hdim64_elayrs8_dlayrs2_zdim256_batch64_lr0.0003_kld0.1_edg0.0_mse1.0_cri1.0_gwt1.0_cwt1.0_gdist0.4_center_scale_mse_adv*val1'
    run_dir = out_dir + run + '/'
    print(run)
    phase = 'test'    
    model_ckpt = 'best_test_fde'
    # model_ckpt = 'model_epoch_100'
    # model_ckpt = 'entire_model_125'
    
    with open(run_dir + 'args.pkl', 'rb') as f: 
        args = pickle.load(f)
        args.batch_size = 64
    
    with open(run_dir + "%s.yaml"%args.model_name, 'r') as file:
        model_params = yaml.load(file, Loader = yaml.FullLoader)
    
    if args.edge_loss_wt<=0:
        model_params['past_dec']['mlp_readout_edge']=False
        model_params['critic']['mlp_readout_edge']=False
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(args.seed)
    
    KSTEPS = 1
    collision_thresh = 0.1
    rank_traj = True
    rank_metric = 'ade'
    pred_steps = 8
    verbose = 0
    
    print('----------------------------')
    print('samples:', KSTEPS)
    print('rank traj:', rank_traj)
    print('rank metric:', rank_metric)
    print('model:', model_ckpt)
    print('pred_steps:', pred_steps)
    print('collision thresh:', collision_thresh)
    print('----------------------------')

    total_ade = []
    total_fde = []
    all_collision = {}
    all_collision['pred'] = {}
    all_collision['gt'] = {}
    total_traj = defaultdict(list)
    static_traj = defaultdict(list)
    
    gt_peds = defaultdict(list)
    peds_per_frame = defaultdict(list)
    gt_collision_per_frame = defaultdict(list)
    pred_collision_per_frame = defaultdict(list)
    
    datasets = ['eth', 'hotel', 'univ', 'zara1', 'zara2']
    # datasets = ['eth']computed
    for dset_name in datasets:
        # dset_name = 'univ'
        # print(dset_name)
        
        prefix = phase + '_batch_size%s'%args.batch_size + '_K%s'%KSTEPS
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
            ade, fde, raw_data_dict = test_epoch(args, model, dataloader, device, KSTEPS=KSTEPS)
            
            with open(raw_data_path + 'raw_data_dict.pkl', 'wb') as f:
                pickle.dump([ade, fde, raw_data_dict], f)
        
 
        print('Evaluating collision on:{} (ADE:{:.2f}, FDE:{:.2f})'.format(dset_name, ade, fde))
        total_ade.append(ade)
        total_fde.append(fde)
        
        total_gt_collision = 0
        total_pred_collision = 0
        
        for k, data_dict in raw_data_dict.items():
            
            seq_start_end = data_dict['seq_start_end']
            total_traj[dset_name].append(seq_start_end[-1][-1])
            
            frame_count = 0
            for start, end in seq_start_end: # each start-end corresponds to single frame
                frame_count += 1
                
                if frame_count%1!=0: #only use this when batch size is greater than 12
                    continue
                
                # obsv_f = data_dict['obsv_frames'][start:end, :]
                target_f = data_dict['trgt_frames'][start:end, :]
                start_f, end_f = int(target_f[0][0]), int(target_f[0][-1])
                # print(start_f)
                
                ped_id = data_dict['ped_id'][start:end, :]
                computed
                target_p = data_dict['trgt'][:, start:end, :] #[12, num_peds, 2]
            
                ade_dict = data_dict['ade_dict'] # key indexed with ped_id, value is a list of 20 ades
                ade_dict = {i:ade_dict[k] for i, k in enumerate(range(start, end))}

                fde_dict = data_dict['fde_dict']
                fde_dict = {i:fde_dict[k] for i, k in enumerate(range(start, end))}
                                
                traj_list = [traj[:, start:end, :] for traj in data_dict['pred']]
                
                target_p = target_p[:pred_steps, :, :]
                if KSTEPS==1:
                    pred_p = traj_list[0]
                    pred_p = pred_p[:pred_steps, :, :]
                    num_pred_collisions, pred_collision_mat, _ = collision_torch(pred_p, y_ids=ped_id, collision_thresh=collision_thresh)
                elif rank_traj:
                    traj_list_ = []
                    for p in range(end-start):
                        top_k_idx = np.argsort(ade_dict[p])[0] if rank_metric=='ade' else np.argsort(fde_dict[p])[0]#top 1
                        top_k_pred = traj_list[top_k_idx][:, p, :]
                        traj_list_.append(top_k_pred)
                    pred_p = torch.stack(traj_list_).permute(1, 0, 2)
                    num_pred_collisions, pred_collision_mat, _ = collision_torch(pred_p, y_ids=ped_id, collision_thresh=collision_thresh)
                else: # take average of samples
                    num_pred_collisions = []
                    for pred_p in traj_list:
                        curr_num_pred_collisions, pred_collision_mat, _ = collision_torch(pred_p, y_ids=ped_id, collision_thresh=collision_thresh)
                        num_pred_collisions.append(curr_num_pred_collisions)
                    num_pred_collisions = np.mean(num_pred_collisions)
                    
                total_pred_collision+=num_pred_collisions    
                
                num_gt_collisions, gt_collision_mat, _ = collision_torch(target_p, ped_id, collision_thresh=collision_thresh)
                total_gt_collision += num_gt_collisions
                
                # num_gt_collisions = 0
                # for i in range(len(ped_id)):
                #     for j in range(i+1, len(ped_id)):
                #         path1 = target_p[:, i, :].cpu().numpy()
                #         path2 = target_p[:, j, :].cpu().numpy()
                #         num_gt_collisions += 2*collision(path1, path2, person_radius=collision_thresh, inter_parts=1)
                # total_gt_collision += int(num_gt_collisions)
                
                
                peds_per_frame[dset_name].append(len(ped_id))
                gt_collision_per_frame[dset_name].append(num_gt_collisions)
                pred_collision_per_frame[dset_name].append(num_pred_collisions)
                

                avg_displacement = torch.sqrt(torch.sum((target_p[-1] - target_p[0])**2, dim=-1))
                static_traj[dset_name].append((avg_displacement<0.2).sum().item()) # distance travel less than 0.2m is considered static
                gt_peds[dset_name].append(ped_id.numpy())

                if verbose>0:
                    print('\nFrame[{}]\n'.format(target_f[0][0]))
                    gt_colliding_peds = (ped_id.repeat((1, ped_id.shape[0]))[gt_collision_mat>0]).tolist()
                    pred_colliding_peds = []
                    for i in range(pred_collision_mat.shape[0]):
                        for j in range(i+1, pred_collision_mat.shape[1]):
                            if pred_collision_mat[i][j]>0:
                                pred_colliding_peds.append([ped_id[i].item(), ped_id[j].item()])
                    if num_gt_collisions>0:
                        print('{} collision in gt traj between: {}'.format(num_gt_collisions, gt_colliding_peds))
                    # print('{} collision in pred traj between: {}'.format(num_pred_collisions.item(), pred_colliding_peds))
                
                # sys.exit()

            
        # print('\n')
        print('Number of collisions in GT Traj:', int(total_gt_collision))
        print('Number of collisions in Predicted Traj:', int(total_pred_collision))

        
        # with open(run_dir + '/collisions.txt', 'a+') as f:
        #     csv_writer = csv.writer(f)
        #     csv_writer.writerow(['Test Model:{}, Dataset:{}, BS:{}, GT collisions :{}, Pred Collisions:{}'.format(
        #         model_ckpt, dset_name, args.batch_size, total_gt_collision.item(), total_pred_collision.item())])
    if verbose>0:computed
        print('**************************')
        print('Total ADE:{}, Total FDE:{}'.format(np.mean(total_ade), np.mean(total_fde)))    
        print('Total Traj:', {k:np.sum(val) for k, val in total_traj.items()})
        print('Static Traj:', {k:np.sum(val) for k, val in static_traj.items()})
        print('Total Number of Peds:', {k:len(np.unique(np.concatenate(val))) for k, val in gt_peds.items()})
        print('Ped Density (peds/frame):', {k:sum(val)/len(val) for k, val in peds_per_frame.items()})
        print('**************************')
    
    for k in gt_collision_per_frame.keys():
        gt_percent_coll_per_frame = [c/p*100 for c, p in zip(gt_collision_per_frame[k], peds_per_frame[k])]
        pred_pecent_coll_per_frame = [c/p*100 for c, p in zip(pred_collision_per_frame[k], peds_per_frame[k])]
        
        print('-'*10, k, '-'*10)
        print('Total collisions: {} | {}'.format(sum(gt_collision_per_frame[k]), sum(pred_collision_per_frame[k])))
        
        # print('Average Collision per Frame (GT):', np.mean(gt_collision_per_frame[k]))
        # print('Average Collision per Frame (Pred):', np.mean(pred_collision_per_frame[k]))
        
        print('Avg percentage of colliding peds per frame: {:.6f} | {:.6f}'.format(np.mean(gt_percent_coll_per_frame), np.mean(pred_pecent_coll_per_frame)))
        
        # print('percentage of avergage colliding peds (GT):', sum(gt_collision_per_frame[k])/sum(peds_per_frame[k])*100)
        # print('percentage avergage colliding peds (Pred):', sum(pred_collision_per_frame[k])/sum(peds_per_frame[k])*100)
overall_gt_collisions = np.sum([sum(v) for v in gt_collision_per_frame.values()])
overall_pred_collisions = np.sum([sum(v) for v in pred_collision_per_frame.values()])
print('Overall Collisions: {} | {} ({:.6f})'.format(overall_gt_collisions, overall_pred_collisions, overall_pred_collisions/overall_gt_collisions))
