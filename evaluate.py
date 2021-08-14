#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import yaml
import time
import pickle
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import dgl
import torch
from torch.utils.data import DataLoader

from utils import TrajectoryDataset 
from train import organize_tensor, rel_to_abs, standardize_output
from misc import setup_gpu, create_new_dir
from metrics import average_l2, final_l2
from model import gnn_model

def test_epoch(args, model, dataloader, device, KSTEPS=20, history=None):
    
    model.eval()
    
    raw_data_dict = defaultdict(dict)

    epoch_ade = 0
    epoch_fde = 0
    total_time = 0
    pbar = tqdm(total=len(dataloader), position=0) 
    with torch.no_grad():
        for iter, (obsv_graph, target_graph) in enumerate(dataloader): 
            pbar.update(1)
            
            _len = [g.number_of_nodes() for g in dgl.unbatch(obsv_graph)]
            cum_start_idx = [0] + np.cumsum(_len).tolist()
            seq_start_end = [[start, end] for start, end in zip(cum_start_idx, cum_start_idx[1:])]
            
            obsv_p = obsv_graph.ndata['pos'].to(device)
            obsv_v = obsv_graph.ndata['vel'].to(device)  # [K, obsv_len, 2]
            obsv_e = obsv_graph.edata['dist'].to(device)    #[K, obsv_len]
            
            target_p = target_graph.ndata['pos'].to(device)
            # target_v = target_graph.ndata['vel'].to(device)
            
            #reshape
            obsv_p = organize_tensor(obsv_p, args.data_format, time_major=True) #[obsv_len, num_peds, 2]
            obsv_v = obsv_v.view(-1, obsv_v.shape[1]*obsv_v.shape[2]) #[K, obsv_len*2]
            
            pred_pos_list = []
            
            for k in range(KSTEPS):
                start=time.time()
                #predict
                logits_v, _, _, _  = model(obsv_graph, obsv_v, obsv_e, device=device) 
                
                total_time+= time.time()-start
                
                logits_v = organize_tensor(logits_v, args.data_format, 2, reshape=True, time_major=True)
                logits_v = standardize_output(logits_v, args.center, args.scale)
                
                pred_pos = rel_to_abs(logits_v, obsv_p[-1, :, :])
                pred_pos_list.append(pred_pos)
                
                
            target_pos = organize_tensor(target_p, args.data_format, reshape=False, time_major=True)
            
            ade_dict = defaultdict(list)
            fde_dict = defaultdict(list)
        
            for pred_pos in pred_pos_list:
                
                for n in range(pred_pos.shape[1]):
                    n_pred =  pred_pos[:, n, :].cpu().numpy() #[pred_len, 2]        
                    n_target = target_pos[:, n, :].cpu().numpy() #[pred_len, 2]
                    
                    ade_dict[n].append(average_l2(n_pred, n_target))
                    fde_dict[n].append(final_l2(n_pred, n_target))
            
            ade = np.mean([np.min(val) for k, val in ade_dict.items()])
            fde = np.mean([np.min(val) for k, val in fde_dict.items()])
            
            epoch_ade += ade
            epoch_fde += fde
            
            raw_data_dict[iter]['ped_id'] = target_graph.ndata['pid']
            raw_data_dict[iter]['obsv_frames'] = obsv_graph.ndata['frames']
            raw_data_dict[iter]['trgt_frames'] = target_graph.ndata['frames']
            raw_data_dict[iter]['obsv'] = obsv_p
            raw_data_dict[iter]['trgt'] = target_pos
            raw_data_dict[iter]['pred'] = pred_pos_list
            raw_data_dict[iter]['ade_dict'] = ade_dict
            raw_data_dict[iter]['fde_dict'] = fde_dict
            raw_data_dict[iter]['seq_start_end'] = seq_start_end
    
    pbar.close()
    
    ade = epoch_ade/(iter+1)
    fde = epoch_fde/(iter+1)
    print('Total Inference Time per Batch:', total_time/(iter+1))
    return ade, fde, raw_data_dict

def get_model(args, model_params, ckpt_path, device):
    
    # model_params['z_sigma'] = 1.8
    if args.edge_loss_wt<=0:
        model_params['past_dec']['mlp_readout_edge']=False
        model_params['critic']['mlp_readout_edge']=False
        
    if 'entire_model' in ckpt_path:
        model = torch.load(ckpt_path)
    elif 'epoch' in ckpt_path:
        checkpoint = torch.load(ckpt_path)
        model = gnn_model(args.model_name, model_params, args)
        model.load_state_dict(checkpoint['state_dict'])
    elif 'best' in ckpt_path:
        model = gnn_model(args.model_name, model_params, args)
        model.load_state_dict(torch.load(ckpt_path), strict=False)
    else:
        raise Exception('Incorrect checkpoint!')
    model = model.double().to(device)
    return model

def get_loader(args, data_dir, phase, force_preprocess=False):
    dataset = TrajectoryDataset(data_dir, phase=phase, preprocess=force_preprocess)
    dataset._standardize_inputs(args.center, args.scale, args.data_format)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=dataset.collate)
    return dataloader

#%%
if __name__=='__main__':
    
    out_dir = './out/SC_GCN/trial_16/'
    # run = 'run-10_hdim64_elayrs1_dlayrs1_zdim128_batch64_lr0.0003_kld0.1_edg0.0_mse1.0_cri0.0_gwt0.0_cwt0.0_center_scale_gcn'
    run = 'run-10_hdim64_elayrs8_dlayrs2_zdim256_batch64_lr0.0003_kld0.1_edg0.0_mse1.0_cri0.0_gwt0.0_cwt0.0_center_scale_mse_adv*val1'
    # run = 'run-10_hdim64_elayrs8_dlayrs2_zdim256_batch64_lr0.0003_kld0.1_edg0.0_mse1.0_cri1.0_gwt0.0_cwt1.0_center_scale_mse_adv*val1'
    # run = 'run-10_hdim64_elayrs8_dlayrs2_zdim256_batch64_lr0.0003_kld0.1_edg0.0_mse1.0_cri1.0_gwt1.0_cwt0.0_gdist0.4_center_scale_mse_adv*val1'
    # run = 'run-10_hdim64_elayrs8_dlayrs2_zdim256_batch64_lr0.0003_kld0.1_edg0.0_mse1.0_cri1.0_gwt1.0_cwt1.0_gdist0.4_center_scale_mse_adv*val1'
    run_dir = out_dir + run + '/'
    
    device = setup_gpu(gpu_id=0)
    
    with open(run_dir + 'args.pkl', 'rb') as f: 
        args = pickle.load(f)
        # args.batch_size=1

    with open(run_dir + "%s.yaml"%args.model_name, 'r') as f:
        model_params = yaml.load(f, Loader = yaml.FullLoader)
        
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(args.seed)

    average_ade = []
    average_fde = []

    model_ckpt = 'best_test_fde'
    # model_ckpt = 'model_epoch_100'
    # model_ckpt = 'model_epoch_60'
    # model_ckpt = 'entire_model_60'
    print('Using model:', model_ckpt)
    
    KSTEPS=1
    phase = 'test'
    test_datasets = 'eth, hotel, univ, zara1, zara2'
    # test_datasets = 'eth'
    for dset_name in test_datasets.split(', '):
        # print('Loading model from {}'.format(run_dir + dset_name + '/' + model_ckpt))
        ckpt_path = run_dir + dset_name + '/' + model_ckpt + '.pth'
        model = get_model(args, model_params, ckpt_path, device)
        
        data_dir = './datasets/{}/'.format(dset_name)
        dataloader = get_loader(args, data_dir, phase=phase, force_preprocess=False)
        
        
        print('Number of samples:', KSTEPS)
        print("Testing {} {} dataset....".format(phase, dset_name))
        start = time.time()
        ade, fde, raw_data_dict = test_epoch(args, model, dataloader, device, KSTEPS)
        print("ADE:{:.6f}, FDE:{:.6f}, Test Time:{:.1f}s".format(ade, fde, time.time()-start))
        
        average_ade.append(ade)
        average_fde.append(fde)
        
        prefix = phase + '_batch_size%s'%args.batch_size + '_K%s'%KSTEPS
        dump_path = create_new_dir(run_dir + dset_name + '/eval_results/' + model_ckpt + '/' + prefix + '/')
        with open(dump_path + 'raw_data_dict.pkl', 'wb') as f:
            pickle.dump([ade, fde, raw_data_dict], f)
            
        # with open(run_dir + 'test_results.txt', 'a+') as f:
        #     writer = csv.writer(f)
        #     writer.writerow(['Model:{}, dataset:{}, z_sigma:{}, K:{}, mADE:{}, mFDE:{}'.format(
        #         model_ckpt, dset_name, model_params['z_sigma'], KSTEPS, ade, fde)])
        
    #write average of all dataset
    print('Average ADE :{}, Average FDE:{}'.format(np.mean(average_ade), np.mean(average_fde)))
    # with open(run_dir + 'test_results.txt', 'a+') as f:
    #         writer = csv.writer(f)
    #         writer.writerow(['Average ADE :{}, Average FDE:{}'.format(np.mean(average_ade), np.mean(average_fde))])
                
                    
        
        
    
    
    

            
        
        