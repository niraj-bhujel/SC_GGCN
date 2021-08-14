import os
import math
import time
import pickle
import numpy as np
from tqdm import tqdm
import networkx as nx

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import dgl
from scipy import sparse as sp

MEAN_ = [-0.0076892033542976974, -0.01959083636764868]
STD_ = [0.3510376460695608, 0.2940771666240619]
        
        
def poly_fit(traj, traj_len, threshold):
    """
    Input:
    - traj: Numpy array of shape (2, traj_len)
    - traj_len: Len of trajectory
    - threshold: Minimum error to be considered for non linear traj
    Output:
    - int: 1 -> Non Linear 0-> Linear
    """
    t = np.linspace(0, traj_len - 1, traj_len)
    res_x = np.polyfit(t, traj[0, -traj_len:], 2, full=True)[1]
    res_y = np.polyfit(t, traj[1, -traj_len:], 2, full=True)[1]
    if res_x + res_y >= threshold:
        return 1.0
    else:
        return 0.0
    
def read_file(_path, delim='\t'):
    data = []
    if delim == 'tab':
        delim = '\t'
    elif delim == 'space':
        delim = ' '
    with open(_path, 'r') as f:
        for line in f:
            line = line.strip().split(delim)
            line = [float(i) for i in line]
            data.append(line)
    return np.asarray(data)

def anorm(p1,p2): 
    NORM = math.sqrt((p1[0]-p2[0])**2+ (p1[1]-p2[1])**2)
    if NORM ==0:
        return 0
    return 1/(NORM)

def seq_to_graph(seq, seq_rel, frames, ped_ids, norm_lap_matr=True):

    num_nodes = seq.shape[0]
    seq_len = seq.shape[2]
    
    nodes_dist = np.zeros((seq_len, num_nodes, num_nodes))#[num_peds, 2
    for t in range(seq_len):
        nodes_coord = seq[:, :, t]
        # Compute distance matrix
        for h in range(len(nodes_coord)): 
            nodes_dist[t, h, h] = 1
            
            for k in range(h+1, len(nodes_coord)):
                
                l2_norm = anorm(nodes_coord[h], nodes_coord[k])
                
                nodes_dist[t, h, k] = l2_norm
                nodes_dist[t, k, h] = l2_norm
                 
        if norm_lap_matr:
            G = nx.from_numpy_matrix(nodes_dist[t, :, :])
            nodes_dist[t, :, :] = nx.normalized_laplacian_matrix(G).toarray()
    
    # Construct the DGL graph
    g = dgl.DGLGraph()
    g.add_nodes(num_nodes)        
    
    edge_feats = []
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i!=j:
                g.add_edge(i, j)
                edge_feats.append(nodes_dist[:, i, j])
    
    assert len(edge_feats) == g.number_of_edges()
    
    # Add edge features
    g.edata['dist'] = torch.DoubleTensor(edge_feats)
    g.ndata['pos'] = torch.DoubleTensor(seq)
    g.ndata['vel'] = torch.DoubleTensor(seq_rel)
    g.ndata['frames'] = torch.DoubleTensor(frames)
    g.ndata['pid'] = torch.DoubleTensor(ped_ids)
    
    return g

def positional_encoding(g, pos_enc_dim):
    """
        Graph positional encoding v/ Laplacian eigenvectors
    """

    # Laplacian
    A = g.adjacency_matrix_scipy(return_edge_ids=False).astype(float)
    N = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float)
    L = sp.eye(g.number_of_nodes()) - N * A * N
    
    # Eigenvectors with numpy
    EigVal, EigVec = np.linalg.eig(L.toarray())
    idx = EigVal.argsort() # increasing order
    EigVal, EigVec = EigVal[idx], np.real(EigVec[:,idx])
    
    pos_enc = np.zeros((EigVec.shape[0], pos_enc_dim))
    pad_len = EigVec.shape[1]-1 if EigVec.shape[1]<=pos_enc_dim else pos_enc_dim
    pos_enc[:, :pad_len] = np.abs(EigVec[:, 1:pos_enc_dim+1])
    
    g.ndata['pos_enc'] = torch.from_numpy(pos_enc).float() 

    # # Eigenvectors with scipy
    # EigVal, EigVec = sp.linum_pedsnalg.eigs(L, k=pos_enc_dim+1, which='SR', tol=1e-2) # for 40 PEs
    # EigVec = EigVec[:, EigVal.argsort()] # increasing order
    # g.ndata['pos_enc'] = torch.from_numpy(EigVec[:,1:pos_enc_dim+1]).float() 

    return g
   
class TrajectoryDataset(Dataset):
    """Dataloder for the Trajectory datasets"""
    def __init__(self, data_dir, phase, obs_len=8, pred_len=12, skip=1,min_ped=1, delim='\t', norm_lap_matr=True,
                 preprocess=False):
        """
        Args:
        - data_dir: Directory containing dataset files in the format
        <frame_id> <ped_id> <x> <y>
        - obs_len: Number of time-steps in input trajectories
        - pred_len: Number of time-steps in output trajectories
        - skip: Number of frames to skip while making the dataset
        - threshold: Minimum error to be considered for non linear traj
        when using a linear predictor
        - min_ped: Minimum number of pedestrians that should be in a seqeunce
        - delim: Delimiter in the dataset files
        -data_format: either "channels_first" or "channels_last", default "channel_first"
        NOTE! After flatenning, the two data_format will be structured in [x_0,...x_t, y_0,...,y_t] and [x_0, y_0,...x_t, y_t] respectively.
        """
        super(TrajectoryDataset, self).__init__()

        self.data_dir = data_dir
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.seq_len = self.obs_len + self.pred_len
        self.skip = skip
        self.min_ped = min_ped
        self.delim = delim
        self.norm_lap_matr = norm_lap_matr
        self.preprocess = preprocess
        self.phase = phase
        self.node_attrs = ['frames', 'pos', 'pid', 'vel']
        self.edge_attrs = ['dist']
        if not self.preprocess:
            try:
                with open(self.data_dir+'{}_graphs.pkl'.format(self.phase), 'rb') as f:
                    data = pickle.load(f)
                    self.obs_graphs_lists = data[0]
                    self.trgt_graphs_lists = data[1]
                    self.n_samples = data[2]
                    self.id_list = data[3]
            except:
                self.preprocess_sequence()
        else:
            self.preprocess_sequence()
            
    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        return self.obs_graphs_lists[index], self.trgt_graphs_lists[index]
    
    def collate(self, samples):
        # The input samples is a list of pairs (graph, label).
        obsv_graphs, trgt_graphs = map(list, zip(*samples))
        
        return dgl.batch(obsv_graphs, self.node_attrs, self.edge_attrs), dgl.batch(trgt_graphs, self.node_attrs, self.edge_attrs)
        # return dgl.batch(obsv_graphs), dgl.batch(trgt_graphs)
    
    def preprocess_sequence(self):
        print('Preprocessing {} sequences from {}'.format(self.phase, self.data_dir))
        start_time=time.time()
        all_files = os.listdir(os.path.join(self.data_dir, self.phase))
        all_files = [os.path.join(self.data_dir+self.phase, file_path) for file_path in all_files
                     if '.txt' in file_path]
        
        num_peds_in_seq = []
        frames_list = []
        id_list = [] 
        seq_list = []
        seq_rel_list = []
        all_unique_peds = 0
        
        pbar = tqdm(total=len(all_files), position=0) 
        for path in all_files:
            pbar.update(1)
            # print('Preparing', path)
            data = read_file(path, self.delim)
            
            #NOTE: Different pedestrian at different scene can have same ID. Make them unique across all scene.
            #first map the scene id to a new global id
            unique_ids = np.unique(data[:, 1])
            unique_ids_new = all_unique_peds + np.arange(len(unique_ids))
            id_map = {pid:pid_new for pid, pid_new in zip(unique_ids, unique_ids_new)}
            #IMPORTANT! Count all unique pedes in each scene
            all_unique_peds += len(unique_ids)
            #update id for each row in the scene
            data[:, 1] = np.array([id_map[pid] for pid in data[:, 1]])
            
            frames = np.unique(data[:, 0]).tolist()
            frame_data = []
            for frame in frames:
                frame_data.append(data[frame == data[:, 0], :])

            num_sequences = int(math.ceil((len(frames) - self.seq_len + 1) / self.skip))

            for idx in range(0, num_sequences * self.skip + 1, self.skip):
                
                curr_seq_data = np.concatenate( frame_data[idx:idx + self.seq_len], axis=0)

                peds_in_curr_seq = np.unique(curr_seq_data[:, 1])
                
                num_peds_considered = 0
                curr_frames_list = []
                curr_id_list = []
                curr_seq_list = []
                curr_seq_rel_list = []
                for _, ped_id in enumerate(peds_in_curr_seq):
                    
                    curr_ped_seq = curr_seq_data[curr_seq_data[:, 1] ==ped_id, :]
                    curr_ped_seq = np.around(curr_ped_seq, decimals=4)
                    
                    pad_front = frames.index(curr_ped_seq[0, 0]) - idx
                    pad_end = frames.index(curr_ped_seq[-1, 0]) - idx + 1
                    
                    if pad_end - pad_front != self.seq_len:
                        continue
                    
                    curr_ped_frames = curr_ped_seq[:, 0]
                    curr_ped_seq = np.transpose(curr_ped_seq[:, 2:]) #(2, 20)
                    
                    # Make coordinates relative
                    curr_ped_seq_rel = np.zeros(curr_ped_seq.shape)
                    curr_ped_seq_rel[:, 1:] = curr_ped_seq[:, 1:] - curr_ped_seq[:, :-1]
            
                    curr_frames_list.append(curr_ped_frames)
                    curr_id_list.append(ped_id)
                    curr_seq_list.append(curr_ped_seq)
                    curr_seq_rel_list.append(curr_ped_seq_rel)
                    
                    num_peds_considered += 1

                if num_peds_considered > self.min_ped:
                                       
                    num_peds_in_seq.append(num_peds_considered)
                    
                    frames_list.append(curr_frames_list)
                    id_list.append(curr_id_list)
                    seq_list.append(curr_seq_list)
                    seq_rel_list.append(curr_seq_rel_list)
        
        pbar.close()
        frames_list = np.concatenate(frames_list, axis=0) #[num_peds, 20]
        id_list = np.concatenate(id_list, axis=0)
        seq_list = np.concatenate(seq_list, axis=0) #[num_peds, 2, 20]
        seq_rel_list = np.concatenate(seq_rel_list, axis=0)
        
        print('Finished preprocessing {} sequences in {:.1f}s'.format(self.phase, time.time()-start_time))
        
        self.num_seq = len(seq_list)
        
        self.id_list = id_list #NOTE id is same for obsv and target
        self.obsv_frames = frames_list[:, :self.obs_len]
        self.trgt_frames = frames_list[:, self.obs_len:]
        
        self.obs_traj = seq_list[:, :, :self.obs_len]
        self.trgt_traj = seq_list[:, :, self.obs_len:]
        
        self.obs_traj_rel = seq_rel_list[:, :, :self.obs_len]
        self.trgt_traj_rel = seq_rel_list[:, :, self.obs_len:]
        
        cum_start_idx = [0] + np.cumsum(num_peds_in_seq).tolist()
        self.seq_start_end = [(start, end) for start, end in zip(cum_start_idx, cum_start_idx[1:])]
        
        #Convert to Graphs
        self.obs_graphs_lists = []
        self.trgt_graphs_lists = []
        
        print("\nPreparing {} {} graphs ...".format(os.path.basename(os.path.normpath(self.data_dir)), self.phase))
        start_time = time.time()
        pbar = tqdm(total=len(self.seq_start_end), position=0) 
        for start, end in self.seq_start_end:
            pbar.update(1)
            obsv_g = seq_to_graph(self.obs_traj[start:end, :], self.obs_traj_rel[start:end, :],
                                  self.obsv_frames[start:end, :], self.id_list[start:end, np.newaxis])
            
            trgt_g = seq_to_graph(self.trgt_traj[start:end, :], self.trgt_traj_rel[start:end, :],
                                    self.trgt_frames[start:end, :], self.id_list[start:end, np.newaxis])
            
            self.obs_graphs_lists.append(obsv_g)
            self.trgt_graphs_lists.append(trgt_g)
        
        pbar.close()
        
        self.n_samples = len(self.obs_graphs_lists)
        
        with open(self.data_dir+ '{}_graphs.pkl'.format(self.phase), 'wb') as f: 
            pickle.dump([self.obs_graphs_lists, self.trgt_graphs_lists, self.n_samples, self.id_list], f)
            
        print('Finished preparing {} graphs in {:.1f}s'.format(self.phase, time.time()-start_time))
        
    def _add_positional_encodings(self, pos_enc_dim):
        # Graph positional encoding v/ Laplacian eigenvectors
        self.obs_graphs_lists = [positional_encoding(g, pos_enc_dim) for g in self.obs_graphs_lists]
        self.trgt_graphs_lists = [positional_encoding(g, pos_enc_dim) for g in self.trgt_graphs_lists]
        
    def _standardize_inputs(self, center, scale, data_format):
        
        for obsv_graph, trgt_graph in zip(self.obs_graphs_lists, self.trgt_graphs_lists):
            
            if center:
                obsv_graph.ndata['vel'][:, 0, :] -= MEAN_[0]
                obsv_graph.ndata['vel'][:, 1, :] -= MEAN_[1]
                
                trgt_graph.ndata['vel'][:, 0, :] -= MEAN_[0]
                trgt_graph.ndata['vel'][:, 1, :] -= MEAN_[1]
                
            if scale:
                obsv_graph.ndata['vel'][:, 0, :] /= STD_[0]
                obsv_graph.ndata['vel'][:, 1, :] /= STD_[1]
                
                trgt_graph.ndata['vel'][:, 0, :] /= STD_[0]
                trgt_graph.ndata['vel'][:, 1, :] /= STD_[1]                
                
            # start_pos = obsv_graph.ndata['pos'][:, :1, :]
            # obsv_graph.ndata['vel'] =  obsv_graph.ndata['pos']  - start_pos #[K, obsv_len, 2]
            # trgt_graph.ndata['vel'] = trgt_graph.ndata['pos'] - start_pos #[K, pred_len, 2]  
            
            #channel shift
            if data_format=='channel_last':
                assert obsv_graph.ndata['pos'].shape[1]==2, "data_format {} is specificed, \
                    but the channel dimension doesn't match".format(self.data_format)
                
                obsv_graph.ndata['pos'] = obsv_graph.ndata['pos'].permute(0, 2, 1)
                obsv_graph.ndata['vel'] = obsv_graph.ndata['vel'].permute(0, 2, 1)
                
                trgt_graph.ndata['pos'] = trgt_graph.ndata['pos'].permute(0, 2, 1)
                trgt_graph.ndata['vel'] = trgt_graph.ndata['vel'].permute(0, 2, 1)
                
                
        
if __name__=='__main__':
    from config import parse_argument
    from misc import setup_gpu
    from trajectory_visualization import plot_path
    args = parse_argument()
    device = setup_gpu(args.gpu_id, memory=args.gpu_memory)
    
    data_set = 'eth'
    data_dir = './datasets/' + data_set + '/'
    
    # datasets = {TrajectoryDataset(data_dir, phase=phase, preprocess=True) for phase in ['train', 'val', 'test']}

    phase = 'test'
    shuffle = True if phase=='train' else False
    dataset = TrajectoryDataset(data_dir, obs_len=8, pred_len=12, phase=phase, preprocess=False)
    
    dataset._standardize_inputs(args.center, args.scale, args.data_format)
    if args.pos_enc:
        dataset._add_positional_encodings(pos_enc_dim=20)

    dataloader = DataLoader(dataset, batch_size=1, shuffle=shuffle, collate_fn=dataset.collate)
    
    max_num_peds = 0
    num_peds = []
    
    for iter, (obsv_graph, target_graph) in enumerate(dataloader): 
        if obsv_graph.number_of_nodes()>max_num_peds:
            max_num_peds = obsv_graph.number_of_nodes()
            
        ped_id = obsv_graph.ndata['pid'].to(device)            
        obsv_p = obsv_graph.ndata['pos'].to(device)  
        obsv_v = obsv_graph.ndata['vel'].to(device)    # num x feat
        obsv_f = obsv_graph.ndata['frames'].to(device)
        obsv_e = obsv_graph.edata['dist'].to(device)
        if args.pos_enc:
            obsv_pos_enc = obsv_graph.ndata['pos_enc'].to(device)
        
        target_p = target_graph.ndata['pos'].to(device)  
        target_v = target_graph.ndata['vel'].to(device)  
        target_f = target_graph.ndata['frames'].to(device)
        target_e = target_graph.edata['dist'].to(device)
        if args.pos_enc:
            target_pos_enc = target_graph.ndata['pos_enc'].to(device)

        num_peds.append(obsv_graph.number_of_nodes())
        # if max_num_peds>900:
        #     break
        # print(obsv_p.shape, target_v.shape)
        # plot_path(obsv_p.permute(2, 0, 1).cpu().numpy(), target_p.permute(2, 0, 1).cpu().numpy(), 
        #           counter=iter, dset_name=data_set, save_dir='./vis_traj/' + data_set + '/')

        # break
