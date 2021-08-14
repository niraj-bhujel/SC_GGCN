#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 21:20:35 2020

@author: dl-asoro
"""


import torch
import torch.nn as nn
import torch.nn.functional as F


"""
    GCN: Graph Convolutional Networks
    Thomas N. Kipf, Max Welling, Semi-Supervised Classification with Graph Convolutional Networks (ICLR 2017)
    http://arxiv.org/abs/1609.02907
"""
from layers.gated_gcn_layer import GatedGCNLayer, GatedGCNLayerIsotropic
from layers.mlp_readout_layer import MLPReadout


class GatedGCNNet(nn.Module):
    
    def __init__(self, net_params, **kwargs):
        
        super().__init__()

        self.in_dim_node = net_params['in_dim_node']
        self.in_dim_edge = net_params['in_dim_edge']
        self.hidden_dim = net_params['hidden_dim']
        self.n_layers = net_params['num_layers']
        self.out_dim_node = net_params['out_dim_node']
        self.out_dim_edge = net_params['out_dim_edge']
        self.embed =  net_params['embed']
        self.mlp_readout_node =  net_params['mlp_readout_node']
        self.mlp_readout_edge = net_params['mlp_readout_edge']
        
        #common to all model
        self.in_feat_dropout = kwargs.get('in_feat_dropout', 0.0)
        self.dropout = kwargs.get('dropout', 0.0)
        self.batch_norm = kwargs.get('batch_norm', True)
        self.residual = kwargs.get('residual', True)
        self.pos_enc = kwargs.get('pos_enc')
        self.pos_enc_dim = kwargs.get('pos_enc_dim')

        self.layer_type = kwargs.get('layer_type')
        if self.layer_type =='gcn':
            Layer = GatedGCNLayerIsotropic
        else:
            Layer = GatedGCNLayer

        if self.pos_enc:
            self.embedding_pos_enc = nn.Linear(self.pos_enc_dim, self.hidden_dim)
            
        if self.embed:
            self.embedding_h = nn.Linear(self.in_dim_node, self.hidden_dim) # node feat is an integer
            self.embedding_e = nn.Linear(self.in_dim_edge, self.hidden_dim) # edge feat is a float
        
        self.layers = nn.ModuleList([Layer(self.hidden_dim, self.hidden_dim, self.dropout,
                                                    self.batch_norm, self.residual) for _ in range(self.n_layers) ])
        if self.mlp_readout_node:
            self.MLP_nodes = MLPReadout(self.hidden_dim, self.out_dim_node)
        
        if self.mlp_readout_edge:
            self.MLP_edges = MLPReadout(self.hidden_dim*2, self.out_dim_edge)
        

    def forward(self, g, h, e, h_pos_enc=None):
        
        # input embedding
        if self.embed:
            h = self.embedding_h(h)
            e = self.embedding_e(e)
            
        if self.pos_enc:
            h_pos_enc = self.embedding_pos_enc(h_pos_enc.float()) 
            h = h + h_pos_enc
            
        h = F.dropout(h, self.in_feat_dropout, training=self.training)
        
        # res gated convnets
        for conv in self.layers:
            h, e = conv(g, h, e)
        
        #update graph 
        g.ndata['h'] = h
        
        # node output
        if self.mlp_readout_node:
            h = self.MLP_nodes(h)
        
        #edge output
        if self.mlp_readout_edge:
            def _edge_feat(edges):
                e = torch.cat([edges.src['h'], edges.dst['h']], dim=1)
                e = self.MLP_edges(e)
                return {'e': e}
            g.apply_edges(_edge_feat)
            e = g.edata['e']

        return g, h, e


class SC_GCNNet(nn.Module):
    
    def __init__(self, net_params, **kwargs):
        super().__init__()
        #assign hidden_dim
        net_params['past_enc']['hidden_dim'] = net_params['hidden_dim']
        net_params['target_enc']['hidden_dim'] = net_params['hidden_dim']
        net_params['past_dec']['hidden_dim'] = net_params['hidden_dim']
        net_params['critic']['hidden_dim'] = net_params['hidden_dim']
        
        #adjsut in_dim node for decoders
        net_params['past_dec']['in_dim_node'] = net_params['hidden_dim'] + net_params['z_dim']
        # net_params['critic']['in_dim_node'] = net_params['hidden_dim'] + net_params['z_dim']
        net_params['critic']['in_dim_node'] = net_params['hidden_dim'] + net_params['z_dim'] + net_params['past_dec']['out_dim_node']
        
        #adjust in_dim_edges for decoders
        net_params['past_dec']['in_dim_edge'] = net_params['hidden_dim']
        net_params['critic']['in_dim_edge'] = net_params['hidden_dim'] + net_params['hidden_dim']
        
        #provide layers for enc
        net_params['past_enc']['num_layers'] = net_params['enc_layers']
        net_params['target_enc']['num_layers'] = net_params['enc_layers']
        
        # provide layers for decoder
        net_params['past_dec']['num_layers'] = net_params['dec_layers']
        net_params['critic']['num_layers'] = net_params['dec_layers']      
        
        self.z_sigma = net_params['z_sigma']
        self.z_dim = net_params['z_dim']
        self.critics = True if kwargs['critic_loss_wt']>0 else False
        
        #past encoder
        self.past_enc = GatedGCNNet(net_params['past_enc'], **kwargs)
        
        self.prior_latent = nn.Linear(net_params['past_enc']['hidden_dim'], self.z_dim*2)
        
        #past decoder
        self.past_dec = GatedGCNNet(net_params['past_dec'], **kwargs)

        #target encoder
        self.target_enc = GatedGCNNet(net_params['target_enc'], **kwargs) 
        
        self.post_latent = nn.Linear(net_params['target_enc']['hidden_dim'], self.z_dim*2)
        
        if self.critics:
            #NOTE! embed must be true for this decoder since it concatenate with past_encoder
            self.critic = GatedGCNNet(net_params['critic'], **kwargs) 
    
    def _reparameterize(self, mean, logvar, device):
        var = logvar.mul(0.5).exp_()    
        eps = torch.DoubleTensor(var.size()).normal_()
        eps = eps.to(device)
        z = eps.mul(var).add_(mean)
        return z

    def _kld(self, mean1, logvar1, mean2, logvar2):
        x1 = torch.sum((logvar2 - logvar1), dim=1)
        x2 = torch.sum(torch.exp(logvar1 - logvar2), dim=1)
        x3 = torch.sum((mean1 - mean2).pow(2) / (torch.exp(logvar2)), dim=1)
        kld_element = x1 - mean1.size(1) + x2 + x3
        return torch.mean(0.5 * kld_element)
    
    def _onrm(self, param, device):
        param_flat = param.view(param.shape[0], -1)
        sym = torch.mm(param_flat, torch.t(param_flat))
        sym -= torch.eye(param_flat.shape[0]).double().to(device)
        return sym.abs().sum()
                
    def forward(self, gx, xx, ex, x_pos_enc=None, gy=None, yy=None, ey=None, y_pos_enc=None, device=torch.device('cpu')):        
        xx_, ex_ = xx, ex #for later
        
        #Encode X~p(X)
        gx, xx, ex = self.past_enc(gx, xx, ex, h_pos_enc=x_pos_enc)
        
        prior_latent = self.prior_latent(xx)
        prior_mean = prior_latent[:, :self.z_dim]
        prior_logvar = prior_latent[:, self.z_dim:]
        
        V = 0
        KLD = 0
        if self.training:
            
            #Encode X, Y ~ q(X, Y)
            yy = torch.cat([xx_, yy], dim=1)
            ey = torch.cat([ex_, ey,], dim=1) 
                
            gy, yy, ey = self.target_enc(gy, yy, ey, h_pos_enc=y_pos_enc)

            #CVAE
            post_latent = self.post_latent(yy)
            post_mean = post_latent[:, :self.z_dim] # 2-d array
            post_logvar = post_latent[:, self.z_dim:] # 2-d array 
            
            KLD = self._kld(post_mean, post_logvar, prior_mean, prior_logvar)
            
            z = self._reparameterize(post_mean, post_logvar, device)

            # KLD += self._onrm(z, device)
                            
        else:
            # z = torch.DoubleTensor(xx.size(0), self.z_dim).to(device)
            # z.normal_(0, self.z_sigma)
            z = self._reparameterize(prior_mean, prior_logvar, device)
        
        #decoder input
        xx = torch.cat([xx, z], dim = 1)                    
        
        xx_ = xx # for critic
        ex_ = ex # for critic
        #decoder
        gx, xx, ex = self.past_dec(gx, xx, ex, h_pos_enc=x_pos_enc)

        if self.critics:
            xx_ = torch.cat([xx_, xx], dim=1)
            ex_ = torch.cat([ex_, ex], dim=1)
            _, V, _= self.critic(gx, xx_, ex_, h_pos_enc=y_pos_enc)
            # V = F.relu(V) #relu perform sligmoid
        
        return xx, ex, V, KLD
    
        
        
def gnn_model(model_name, model_params, args):
    
    models = {'GatedGCN': GatedGCNNet,
              'SC_GCN':SC_GCNNet,
              }
    return models[model_name](model_params, **vars(args))


if __name__=='__main__':
    
    import numpy as np
    import networkx as nx
    import dgl
    from config import parse_argument
    from misc import *
    import yaml
    
    args = parse_argument()
    device = setup_gpu(args.gpu_id, memory=args.gpu_memory)


    num_nodes = 3
    #create dgl graph
    gx = dgl.DGLGraph()
    gy = dgl.DGLGraph()
    #add nodes
    gx.add_nodes(num_nodes)
    gy.add_nodes(num_nodes)
    #add edges
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i!=j:
                gx.add_edges(i, j)
                gy.add_edges(i, j)
    nx.draw(gx.to_networkx().to_undirected(), with_labels=True)
    
    batch_graphs = dgl.batch([gx])
    xx = torch.rand((gx.number_of_nodes(), 16)).double().to(device)
    ex = torch.rand(gx.number_of_edges(), 8).double().to(device)
    
    yy = torch.rand((gy.number_of_nodes(), 24)).double().to(device)
    ey = torch.rand(gy.number_of_edges(), 12).double().to(device)
    
    gx_pos_enc = torch.rand((gx.number_of_nodes(), 20)).double().to(device)
    gy_pos_enc = torch.rand((gy.number_of_nodes(), 20)).double().to(device)

    args = parse_argument()
        
    model_name = "SC_GCN"
    with open("./%s.yaml"%model_name, 'r') as file:
        model_params = yaml.load(file, Loader = yaml.FullLoader)
    if args.edge_loss_wt<=0:
        model_params['past_dec']['mlp_readout_edge']=False
        model_params['critic']['mlp_readout_edge']=False
        
    model = gnn_model(model_name, model_params, args)
    
    # model_attributes(model)
    model_parameters(model, verbose=0)
    # h_out, e_out = net(g, h, e)
    model = model.double().to(device)
    
    model_out = model(gx, xx, ex, gy=gy, yy=yy, ey=ey, device=device)
    # print(h_out)
    