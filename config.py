#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 00:27:03 2020

@author: dl-asoro
"""
import argparse
        

def parse_argument(cmd=None):
    parser = argparse.ArgumentParser()
    
    # model    
    parser.add_argument('--model_name', default='SC_GCN')
    parser.add_argument('--layer_type', default='gated_gcn')
    parser.add_argument('--in_feat_dropout', type=float, default=0.)
    parser.add_argument('--dropout', type=float, default=0.)
    parser.add_argument('--batch_norm', action='store_true', default=True)
    parser.add_argument('--residual', action='store_true', default=True)
    
    # Loss
    parser.add_argument('--kld_loss_wt', type=float, default=0.1)
    parser.add_argument('--edge_loss_wt', type=float, default=0.0)
    parser.add_argument('--mse_loss_wt', type=float, default=1.0)
    parser.add_argument('--log_loss_wt', type=float, default=0.0)
    parser.add_argument('--critic_loss_wt', type=float, default=0.3)
    parser.add_argument('--goal_rewards_wt', type=float, default=0.0)
    parser.add_argument('--collision_rewards_wt', type=float, default=1.)
    parser.add_argument('--gdist_thresh', type=float, default=0.4, help='threshold to consider for goal reward')
    parser.add_argument('--collision_thresh', type=float, default=0.1, help='threshold to consider for collision')

    # Data
    parser.add_argument('--dataset', default='hotel', help='eth, hotel, univ, zara1, zara2')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--obs_seq_len', type=int, default=8)
    parser.add_argument('--pred_seq_len', type=int, default=12)
    parser.add_argument('--skip', type=int, default=1)
    parser.add_argument('--preprocess', dest='preprocess', action='store_false', default=False)
    parser.add_argument('--data_format', default='channel_first', help='channel_first, channel_last')
    parser.add_argument('--force_preprocess', dest='preprocess', action='store_true', default=True)
    parser.add_argument('--pos_enc', action='store_true', default=False)
    parser.add_argument('--pos_enc_dim', type=int, default=20)
    parser.add_argument('--center', action='store_true', default=True)
    parser.add_argument('--scale', action='store_true', default=True)
    
    # Training and evaluation
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--k_samples', type=int, default=20, help='Number of samples used for evaluations')
    parser.add_argument('--initial_epoch', type=int, default=1, help='Start epoch number')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--early_stop_patience', type=int, default=20, help='each step is epoch * test_interval')
    parser.add_argument('--early_stop_metric', default='test_fde', help='metric to use for early stopping')
    parser.add_argument('--resume_training', action='store_false', default=False)
    parser.add_argument('--resume_epoch', type=int, default=70, help='initial_epoch - 1')
    parser.add_argument('--overwrite', dest='overwrite', action='store_true', default=True)
    parser.add_argument('--no_overwrite', dest='overwrite', action='store_false', default=False)
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--gpu_memory', type=int, default=512) 
    
    # Optimizer
    parser.add_argument('--lr', type=float, default=0.0003,  help='Initial learning rate,') 
    parser.add_argument('--weight_decay', type=float, default=0., help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.0, help='momemtum factor')
    parser.add_argument('--grad_clip', type=float, default=None, help='gadient clipping')
    
    
    #--Plateau scheduler
    parser.add_argument('--lr_patience', type=int, default=5)
    parser.add_argument('--lr_reduce_factor', type=int, default=0.1, help='factor to drop the lr')
    parser.add_argument('--lr_scheduler_metric', default='test_fde', help='metric to monitor')
    
    #--logging
    parser.add_argument('--trial', type=int, default=1, help='tial number for trianing')
    parser.add_argument('--run', type=int, default=0, help='Run number for trianing')
    parser.add_argument('--prefix', default='', help='Prefix for trianing')
    parser.add_argument('--test_interval', type=int, default=5, help='Test model (ade, fde) every test_interval')
    parser.add_argument('--log_interval', type=int, default=40, help='Save model every log_interval')
    parser.add_argument('--mode', default='train', help='Normal train mode or search param mode [param_search, train]')
    
    args = parser.parse_args(cmd)
    
    return args

if __name__=='__main__':
    args = parse_argument()
    # args = parse_argument(["--data_format", "channel_last"])
    # args = parse_argument(["--overwrite"])
    args = parse_argument(["--no_overwrite"])
    print(args.overwrite)