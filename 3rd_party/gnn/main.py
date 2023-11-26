"""
PyTorch DDP integrated with PyGeom for multi-node training
"""
from __future__ import absolute_import, division, print_function, annotations
import os
import sys
import socket
import logging

from typing import Optional, Union, Callable

import numpy as np

import hydra
import time
import torch
import torch.utils.data
import torch.utils.data.distributed
from torch.cuda.amp.grad_scaler import GradScaler
import torch.multiprocessing as mp
import torch.distributions as tdist 

import torch.distributed as dist
import torch.distributed.nn as distnn
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from omegaconf import DictConfig
from torch.nn.parallel import DistributedDataParallel as DDP
Tensor = torch.Tensor

# PyTorch Geometric
import torch_geometric
from torch_geometric.data import Data
import torch_geometric.utils as utils
import torch_geometric.nn as tgnn

# Models
import models.gnn as gnn

# Graph connectivity/plotting 
import graph_connectivity as gcon
import graph_plotting as gplot

# Clean printing
from prettytable import PrettyTable 

log = logging.getLogger(__name__)

TORCH_FLOAT_DTYPE = torch.float32

# Get MPI:
try:
    from mpi4py import MPI
    WITH_DDP = True
    LOCAL_RANK = os.environ.get('OMPI_COMM_WORLD_LOCAL_RANK', '0')
    # LOCAL_RANK = os.environ['OMPI_COMM_WORLD_LOCAL_RANK']
    SIZE = MPI.COMM_WORLD.Get_size()
    RANK = MPI.COMM_WORLD.Get_rank()
    COMM = MPI.COMM_WORLD

    WITH_CUDA = torch.cuda.is_available()
    DEVICE = 'gpu' if WITH_CUDA else 'cpu'
    if DEVICE == 'gpu':
        DEVICE_ID = 'cuda:0' 
    else:
        DEVICE_ID = 'cpu'

    # pytorch will look for these
    os.environ['RANK'] = str(RANK)
    os.environ['WORLD_SIZE'] = str(SIZE)
    # -----------------------------------------------------------
    # NOTE: Get the hostname of the master node, and broadcast
    # it to all other nodes It will want the master address too,
    # which we'll broadcast:
    # -----------------------------------------------------------
    MASTER_ADDR = socket.gethostname() if RANK == 0 else None
    MASTER_ADDR = MPI.COMM_WORLD.bcast(MASTER_ADDR, root=0)
    os.environ['MASTER_ADDR'] = MASTER_ADDR
    os.environ['MASTER_PORT'] = str(2345)

except (ImportError, ModuleNotFoundError) as e:
    WITH_DDP = False
    SIZE = 1
    RANK = 0
    LOCAL_RANK = 0
    MASTER_ADDR = 'localhost'
    log.warning('MPI Initialization failed!')
    log.warning(e)

def init_process_group(
    rank: Union[int, str],
    world_size: Union[int, str],
    backend: Optional[str] = None,
) -> None:
    if WITH_CUDA:
        backend = 'nccl' if backend is None else str(backend)

    else:
        backend = 'gloo' if backend is None else str(backend)

    dist.init_process_group(
        backend,
        rank=int(rank),
        world_size=int(world_size),
        init_method='env://',
    )


def cleanup():
    dist.destroy_process_group()

def force_abort():
    time.sleep(2)
    if WITH_DDP:
        COMM.Abort()
    else:
        sys.exit("Exiting...")

def metric_average(val: Tensor):
    if (WITH_DDP):
        dist.all_reduce(val, op=dist.ReduceOp.SUM)
        return val / SIZE
    return val

class Trainer:
    def __init__(self, cfg: DictConfig, scaler: Optional[GradScaler] = None):
        self.cfg = cfg
        self.rank = RANK
        if scaler is None:
            self.scaler = None
        self.device = 'gpu' if torch.cuda.is_available() else 'cpu'
        self.backend = self.cfg.backend
        if WITH_DDP:
            init_process_group(RANK, SIZE, backend=self.backend)
        
        # ~~~~ Init torch stuff 
        self.setup_torch()

        # ~~~~ Setup local graph 
        self.data_reduced, self.data_full, self.idx_full2reduced, self.idx_reduced2full = self.setup_local_graph()

        # ~~~~ Setup halo nodes 
        self.neighboring_procs = []
        self.setup_halo()

        # ~~~~ Setup data 
        self.data = self.setup_data()
        if RANK == 0: log.info('Done with setup_data')


        # ~~~~ # self.n_nodes_internal_procs = list(torch.empty([1], dtype=torch.int64, device=DEVICE_ID)) * SIZE
        # ~~~~ # if WITH_CUDA:
        # ~~~~ #     self.data.n_nodes_internal = self.data.n_nodes_internal.cuda()
        # ~~~~ # dist.all_gather(self.n_nodes_internal_procs, self.data.n_nodes_internal)
        # ~~~~ # print('[RANK %d] -- data: ' %(RANK), self.data)

        # ~~~~ Setup halo exchange masks
        self.mask_send, self.mask_recv = self.build_masks()
        if RANK == 0: log.info('Done with build_masks')

        # ~~~~ Initialize send/recv buffers on device (if applicable)
        self.hidden_channels = 32
        self.buffer_send, self.buffer_recv = self.build_buffers(self.hidden_channels)
        if RANK == 0: log.info('Done with build_buffers')

        # ~~~~ # # ~~~~ Do a halo swap on position matrices 
        # ~~~~ # #if RANK == 1: 
        # ~~~~ # #    print('[RANK %d] -- pos before: ' %(RANK), self.data.pos)
        # ~~~~ # if WITH_CUDA:
        # ~~~~ #     self.data.pos = self.data.pos.cuda()
        # ~~~~ # self.data.pos = self.halo_swap(self.data.pos, self.pos_buffer_send, self.pos_buffer_recv)
        # ~~~~ # #if RANK == 0: 
        # ~~~~ # #    print('[RANK %d] -- pos after: ' %(RANK), self.data.pos)

        # ~~~~ Build model and move to gpu 
        self.model = self.build_model()
        if self.device == 'gpu':
            self.model.cuda()
        if RANK == 0: log.info('Done with build_model')

        # ~~~~ Init training and testing loss history 
        self.loss_hist_train = np.zeros(self.cfg.epochs)
        self.loss_hist_test = np.zeros(self.cfg.epochs)

        # ~~~~ Set model and checkpoint savepaths 
        try:
            self.ckpt_path = cfg.ckpt_dir + self.model.get_save_header() + '.tar'
            self.model_path = cfg.model_dir + self.model.get_save_header() + '.tar'
        except (AttributeError) as e:
            self.ckpt_path = cfg.ckpt_dir + 'checkpoint.tar'
            self.model_path = cfg.model_dir + 'model.tar'

        # ~~~~ Load model parameters if we are restarting from checkpoint
        self.epoch = 0
        self.epoch_start = 1
        self.training_iter = 0
        if self.cfg.restart:
            ckpt = torch.load(self.ckpt_path)
            self.model.load_state_dict(ckpt['model_state_dict'])
            self.epoch_start = ckpt['epoch'] + 1
            self.epoch = self.epoch_start
            self.training_iter = ckpt['training_iter']

            self.loss_hist_train = ckpt['loss_hist_train']
            self.loss_hist_test = ckpt['loss_hist_test']

            if len(self.loss_hist_train) < self.cfg.epochs:
                loss_hist_train_new = np.zeros(self.cfg.epochs)
                loss_hist_test_new = np.zeros(self.cfg.epochs)

                loss_hist_train_new[:len(self.loss_hist_train)] = self.loss_hist_train
                loss_hist_test_new[:len(self.loss_hist_test)] = self.loss_hist_test

                self.loss_hist_train = loss_hist_train_new
                self.loss_hist_test = loss_hist_test_new

        # ~~~~ Wrap model in DDP
        if WITH_DDP and SIZE > 1:
            self.model = DDP(self.model)

        # ~~~~ Set loss function
        self.loss_fn = nn.MSELoss()

        # ~~~~ Set optimizer 
        self.optimizer = self.build_optimizer(self.model)

        # ~~~~ Set scheduler 
        self.scheduler = self.build_scheduler(self.optimizer)

        # ~~~~ Load optimizer+scheduler parameters if we are restarting from checkpoint
        if self.cfg.restart:
            self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            self.scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            if RANK == 0:
                astr = 'RESTARTING FROM CHECKPOINT -- STATE AT EPOCH %d/%d' %(self.epoch_start-1, self.cfg.epochs)
                sepstr = '-' * len(astr)
                log.info(sepstr)
                log.info(astr)
                log.info(sepstr)

    def print_node_attribute(self, values_tensor: Tensor, name_tensor: str) -> None:

        global_ids = self.data['train']['example'].global_ids

        # Sort by global ids 
        _, idx_sort = torch.sort(global_ids)

        
        values_list = values_tensor[idx_sort].tolist()
        integers_list = global_ids[idx_sort].tolist()

        if values_tensor.shape[0] != len(global_ids):
            raise ValueError("The tensor being printed is not the same size as global_ids. Input only the local nodes.")

        table = PrettyTable(['[RANK %d, SIZE %d] %s' %(RANK,SIZE,name_tensor), 'Global IDs'])
        for value, integer in zip(values_list, integers_list):
            table.add_row([value, integer])
        
        print(table)
        return

    def build_model(self) -> nn.Module:
        # ~~~~ Toy model 
        #model = gnn.toy_gnn()
        #model = gnn.toy_gnn_distributed()

        # ~~~~ Actual model 
        if RANK == 0:
            log.info('In build_model...')

        sample = self.data['train']['example'] 
        model = gnn.mp_gnn_distributed(input_channels = sample.x.shape[1], 
                           hidden_channels = self.hidden_channels,
                           output_channels = sample.y.shape[1],
                           n_mlp_layers = [3,3,3], 
                           n_messagePassing_layers = 5,
                           activation = F.elu,
                           halo_swap_mode = 'all_to_all', 
                           #halo_swap_mode = 'sendrecv', 
                           #halo_swap_mode = 'none', 
                           name = 'RANK_%d_SIZE_%d' %(RANK,SIZE))
        
        return model

    def build_optimizer(self, model: nn.Module) -> torch.optim.Optimizer:
        """
        DDP: scale learning rate by the number of GPUs
        """
        optimizer = optim.Adam(model.parameters(),
                               lr=SIZE * self.cfg.lr_init)
        return optimizer

    def build_scheduler(self, optimizer: torch.optim.Optimizer) -> torch.optim.lr_scheduler:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5,
                                patience=5, threshold=0.0001, threshold_mode='rel',
                                cooldown=0, min_lr=1e-8, eps=1e-08, verbose=True)
        return scheduler

    def setup_torch(self):
        torch.manual_seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)

    def halo_swap(self, input_tensor, buff_send, buff_recv):
        """
        Performs halo swap using send/receive buffers
        """
        if SIZE > 1:
            # Fill send buffer
            for i in self.neighboring_procs:
                buff_send[i] = input_tensor[self.mask_send[i]]

            # Perform swap
            req_send_list = []
            for i in self.neighboring_procs:
                req_send = dist.isend(tensor=buff_send[i], dst=i)
                req_send_list.append(req_send)
            
            req_recv_list = []
            for i in self.neighboring_procs:
                req_recv = dist.irecv(tensor=buff_recv[i], src=i)
                req_recv_list.append(req_recv)

            for req_send in req_send_list:
                req_send.wait()

            for req_recv in req_recv_list:
                req_recv.wait()

            dist.barrier()

            # Fill halo nodes 
            for i in self.neighboring_procs:
                input_tensor[self.mask_recv[i]] = buff_recv[i]
        return input_tensor 

    def build_masks(self):
        """
        Builds index masks for facilitating halo swap of nodes 
        """
        mask_send = [torch.tensor([])] * SIZE
        mask_recv = [torch.tensor([])] * SIZE

        #mask_send = [None] * SIZE
        #mask_recv = [None] * SIZE

        if SIZE > 1: 
            #n_nodes_local = self.data.n_nodes_internal + self.data.n_nodes_halo
            halo_info = self.data['train']['example'].halo_info

            for i in self.neighboring_procs:
                idx_i = halo_info[:,3] == i
                # index of nodes to send to proc i 
                mask_send[i] = halo_info[:,0][idx_i] 
                
                # index of nodes to receive from proc i  
                mask_recv[i] = halo_info[:,1][idx_i]

                if len(mask_send[i]) != len(mask_recv[i]): 
                    log.info('For neighbor rank %d, the number of send nodes and the number of receive nodes do not match. Check to make sure graph is partitioned correctly.' %(i))
                    force_abort()
        return mask_send, mask_recv 

    def build_buffers(self, n_features):
        buff_send = [torch.tensor([])] * SIZE
        buff_recv = [torch.tensor([])] * SIZE
        
        if SIZE > 1: 

            # Get the maximum number of nodes that will be exchanged (required for all_to_all based halo swap)
            n_nodes_to_exchange = torch.zeros(SIZE)
            for i in self.neighboring_procs:
                n_nodes_to_exchange[i] = len(self.mask_send[i])
            n_max = n_nodes_to_exchange.max()
            if WITH_CUDA: 
                n_max = n_max.cuda()
            dist.all_reduce(n_max, op=dist.ReduceOp.MAX)
            n_max = int(n_max)

            # fill the buffers 
            for i in range(SIZE): 
                buff_send[i] = torch.empty([n_max, n_features], dtype=torch.float32, device=DEVICE_ID) 
                buff_recv[i] = torch.empty([n_max, n_features], dtype=torch.float32, device=DEVICE_ID)

            #for i in self.neighboring_procs:
            #    buff_send[i] = torch.empty([len(self.mask_send[i]), n_features], dtype=torch.float32, device=DEVICE_ID) 
            #    buff_recv[i] = torch.empty([len(self.mask_recv[i]), n_features], dtype=torch.float32, device=DEVICE_ID)

        return buff_send, buff_recv 

    def gather_node_tensor(self, input_tensor, dst=0, dtype=torch.float32):
        """
        Gathers node-based tensor into root proc. Shape is [n_internal_nodes, n_features] 
        NOTE: input tensor on all ranks should correspond to INTERNAL nodes (exclude halo nodes) 
        n_internal_nodes can vary for each proc, but n_features must be the same 
        """
        # torch.distributed.gather(tensor, gather_list=None, dst=0, group=None, async_op=False)
        n_nodes = torch.tensor(input_tensor.shape[0])
        n_features = torch.tensor(input_tensor.shape[1])

        n_nodes_procs = list(torch.empty([1], dtype=torch.int64, device=DEVICE_ID)) * SIZE
        if WITH_CUDA:
            n_nodes = n_nodes.cuda()
        dist.all_gather(n_nodes_procs, n_nodes)

        gather_list = None
        if RANK == 0:
            gather_list = [None] * SIZE
            for i in range(SIZE):
                gather_list[i] = torch.empty([n_nodes_procs[i], n_features], 
                                             dtype=dtype,
                                             device=DEVICE_ID)
        dist.gather(input_tensor, gather_list, dst=0)
        return gather_list

    # def gather_node_tensor(self, input_tensor, dst=0, dtype=torch.float32):
    #     """
    #     Gathers node-based tensor into root proc. Shape is [n_internal_nodes, n_features] 
    #     NOTE: input tensor on all ranks should correspond to INTERNAL nodes (exclude halo nodes) 
    #     n_internal_nodes can vary for each proc, but n_features must be the same 
    #     """
    #     # torch.distributed.gather(tensor, gather_list=None, dst=0, group=None, async_op=False)
    #     n_features = input_tensor.shape[1]
    #     gather_list = None
    #     if RANK == 0:
    #         gather_list = [None] * SIZE
    #         for i in range(SIZE):
    #             gather_list[i] = torch.empty([self.n_nodes_internal_procs[i], n_features],
    #                                          dtype=dtype,
    #                                          device=DEVICE_ID)
    #     dist.gather(input_tensor, gather_list, dst=0)
    #     return gather_list 
        
    def setup_local_graph(self):
        """
        Load in the local graph
        """
        main_path = self.cfg.gnn_outputs_path 

        path_to_pos_full = main_path + 'pos_node_rank_%d_size_%d' %(RANK,SIZE)
        path_to_ei = main_path + 'edge_index_rank_%d_size_%d' %(RANK,SIZE)
        path_to_overlap = main_path + 'overlap_ids_rank_%d_size_%d' %(RANK,SIZE)
        path_to_glob_ids = main_path + 'global_ids_rank_%d_size_%d' %(RANK,SIZE)
        path_to_unique_local = main_path + 'local_unique_mask_rank_%d_size_%d' %(RANK,SIZE)
        path_to_unique_halo = main_path + 'halo_unique_mask_rank_%d_size_%d' %(RANK,SIZE)
        
        # ~~~~ Get positions and global node index
        if self.cfg.verbose: log.info('[RANK %d]: Loading positions and global node index' %(RANK))
        pos = np.loadtxt(path_to_pos_full, dtype=np.float32)
        gli = np.loadtxt(path_to_glob_ids, dtype=np.int64).reshape((-1,1))

        # ~~~~ Get edge index
        if self.cfg.verbose: log.info('[RANK %d]: Loading edge index' %(RANK))
        ei = np.loadtxt(path_to_ei, dtype=np.int64).T
        
        # ~~~~ Get local unique mask
        if self.cfg.verbose: log.info('[RANK %d]: Loading local unique mask' %(RANK))
        local_unique_mask = np.loadtxt(path_to_unique_local, dtype=np.int64)

        # ~~~~ Get halo unique mask
        halo_unique_mask = np.array([])
        if SIZE > 1:
            halo_unique_mask = np.loadtxt(path_to_unique_halo, dtype=np.int64)

        # ~~~~ Make the full graph: 
        if self.cfg.verbose: log.info('[RANK %d]: Making the FULL GLL-based graph with overlapping nodes' %(RANK))
        data_full = Data(x = None, edge_index = torch.tensor(ei), pos = torch.tensor(pos), global_ids = torch.tensor(gli.squeeze()), local_unique_mask = torch.tensor(local_unique_mask), halo_unique_mask = torch.tensor(halo_unique_mask))
        data_full.edge_index = utils.remove_self_loops(data_full.edge_index)[0]
        data_full.edge_index = utils.coalesce(data_full.edge_index)
        data_full.edge_index = utils.to_undirected(data_full.edge_index)
        data_full.local_ids = torch.tensor(range(data_full.pos.shape[0]))

        # ~~~~ Get reduced (non-overlapping) graph and indices to go from full to reduced  
        if self.cfg.verbose: log.info('[RANK %d]: Making the REDUCED GLL-based graph with non-overlapping nodes' %(RANK))
        data_reduced, idx_full2reduced = gcon.get_reduced_graph(data_full)

        # ~~~~ Get the indices to go from reduced back to full graph  
        # idx_reduced2full = None
        if self.cfg.verbose: log.info('[RANK %d]: Getting idx_reduced2full' %(RANK))
        idx_reduced2full = gcon.get_upsample_indices(data_full, data_reduced, idx_full2reduced)

        return data_reduced, data_full, idx_full2reduced, idx_reduced2full


    def setup_halo(self):
        if self.cfg.verbose: log.info('[RANK %d]: Assembling halo_ids_list using reduced graph' %(RANK))
        main_path = self.cfg.gnn_outputs_path

        halo_info = None
        if SIZE > 1:
            halo_info = torch.tensor(np.load(main_path + '/halo_info_rank_%d_size_%d.npy' %(RANK,SIZE)))
            # Get list of neighboring processors for each processor
            self.neighboring_procs = np.unique(halo_info[:,3])
            n_nodes_local = self.data_reduced.pos.shape[0]
            n_nodes_halo = halo_info.shape[0]
        else:
            #print('[RANK %d] neighboring procs: ' %(RANK), self.neighboring_procs)
            halo_info = torch.Tensor([])
            n_nodes_local = self.data_reduced.pos.shape[0]
            n_nodes_halo = 0

        if self.cfg.verbose: log.info('[RANK %d] neighboring procs: ' %(RANK), self.neighboring_procs)

        self.data_reduced.n_nodes_local = torch.tensor(n_nodes_local, dtype=torch.int64)
        self.data_reduced.n_nodes_halo = torch.tensor(n_nodes_halo, dtype=torch.int64)
        self.data_reduced.halo_info = halo_info

        return 

    def setup_data(self):
        """
        Generate the PyTorch Geometric Dataset 
        """
        if RANK == 0:
            log.info('In setup_data...')

        # Load data 
        main_path = self.cfg.gnn_outputs_path
        path_to_x = main_path + 'fld_u_rank_%d_size_%d' %(RANK,SIZE)
        #path_to_y = main_path + 'fld_p_rank_%d_size_%d' %(RANK,SIZE)
        path_to_y = main_path + 'fld_u_rank_%d_size_%d' %(RANK,SIZE)
        data_x = np.loadtxt(path_to_x, ndmin=2, dtype=np.float32) #[:,0:1]
        data_y = np.loadtxt(path_to_y, ndmin=2, dtype=np.float32) 

        # Get data in reduced format (non-overlapping)
        data_x_reduced = data_x[self.idx_full2reduced, :]
        data_y_reduced = data_y[self.idx_full2reduced, :]
        pos_reduced = self.data_reduced.pos

        # Read in edge weights 
        path_to_ew = main_path + 'edge_weights_rank_%d_size_%d.npy' %(RANK,SIZE)
        edge_freq = torch.tensor(np.load(path_to_ew))
        self.data_reduced.edge_weight = 1.0/edge_freq

        # Read in node degree
        path_to_node_degree = main_path + 'node_degree_rank_%d_size_%d.npy' %(RANK,SIZE)
        node_degree = torch.tensor(np.load(path_to_node_degree))
        self.data_reduced.node_degree = node_degree

        # Add halo nodes by appending the end of the node arrays  
        n_nodes_halo = self.data_reduced.n_nodes_halo 
        n_features_x = data_x_reduced.shape[1]
        data_x_halo = torch.zeros((n_nodes_halo, n_features_x), dtype=TORCH_FLOAT_DTYPE) 

        n_features_y = data_y_reduced.shape[1]
        data_y_halo = torch.zeros((n_nodes_halo, n_features_y), dtype=TORCH_FLOAT_DTYPE)

        n_features_pos = pos_reduced.shape[1]
        pos_halo = torch.zeros((n_nodes_halo, n_features_pos), dtype=TORCH_FLOAT_DTYPE)

        node_degree_halo = torch.zeros((n_nodes_halo), dtype=TORCH_FLOAT_DTYPE)

        # Add self-edges for halo nodes (unused) 
        n_nodes_local = self.data_reduced.n_nodes_local
        edge_index_halo = torch.arange(n_nodes_local, n_nodes_local + n_nodes_halo, dtype=torch.int64)
        edge_index_halo = torch.stack((edge_index_halo,edge_index_halo))

        # Add filler edge weights for these self-edges 
        edge_weight_halo = torch.zeros(n_nodes_halo)

        # Populate data object 
        n_features_in = data_x_reduced.shape[1]
        n_features_out = data_y_reduced.shape[1]
        n_nodes = self.data_reduced.pos.shape[0]
        device_for_loading = 'cpu'

        # Populate edge_attrs
        data_ref = self.data_reduced
        cart = torch_geometric.transforms.Cartesian(norm=False, max_value = None, cat = False)
        dist = torch_geometric.transforms.Distance(norm = False, max_value = None, cat = True)
        cart(data_ref) # adds cartesian/component-wise distance
        dist(data_ref) # adds euclidean distance

        # Get dictionary 
        reduced_graph_dict = self.data_reduced.to_dict()

        # Create training dataset -- only 1 snapshot for demo
        data_train_list = []
        data_temp = Data(   
                            x = torch.tensor(data_x_reduced), 
                            y = torch.tensor(data_y_reduced)
                        )
        for key in reduced_graph_dict.keys():
            data_temp[key] = reduced_graph_dict[key]
        data_temp.x = torch.cat((data_temp.x, data_x_halo), dim=0)
        data_temp.y = torch.cat((data_temp.y, data_y_halo), dim=0)
        data_temp.pos = torch.cat((data_temp.pos, pos_halo), dim=0)
        data_temp.node_degree = torch.cat((data_temp.node_degree, node_degree_halo), dim=0)
        data_temp.edge_index = torch.cat((data_temp.edge_index, edge_index_halo), dim=1)
        data_temp.edge_weight = torch.cat((data_temp.edge_weight, edge_weight_halo), dim=0)

        data_temp = data_temp.to(device_for_loading)
        data_train_list.append(data_temp)
        n_train = len(data_train_list) # should be 1
       
        # Create validation dataset -- same data for demo  
        data_valid_list = []
        data_temp = Data(   
                            x = torch.tensor(data_x_reduced), 
                            y = torch.tensor(data_y_reduced)
                        )
        for key in reduced_graph_dict.keys():
            data_temp[key] = reduced_graph_dict[key]
        data_temp.x = torch.cat((data_temp.x, data_x_halo), dim=0)
        data_temp.y = torch.cat((data_temp.y, data_y_halo), dim=0)
        data_temp.pos = torch.cat((data_temp.pos, pos_halo), dim=0)
        data_temp.node_degree = torch.cat((data_temp.node_degree, node_degree_halo), dim=0)
        data_temp.edge_index = torch.cat((data_temp.edge_index, edge_index_halo), dim=1)
        data_temp.edge_weight = torch.cat((data_temp.edge_weight, edge_weight_halo), dim=0)

        data_temp = data_temp.to(device_for_loading)
        data_valid_list.append(data_temp)
        n_valid = len(data_valid_list) # should be 1

        train_dataset = data_train_list
        test_dataset = data_valid_list 

        # No need for distributed sampler -- create standard dataset loader  
        train_loader = torch_geometric.loader.DataLoader(train_dataset, batch_size=self.cfg.batch_size, shuffle=False)
        test_loader = torch_geometric.loader.DataLoader(test_dataset, batch_size=self.cfg.test_batch_size, shuffle=False)  

        if (RANK == 0):
            print(data_train_list[0])

        return {
            'train': {
                'loader': train_loader,
                'example': train_dataset[0],
            },
            'test': {
                'loader': test_loader,
                'example': test_dataset[0],
            }
        }

    def setup_data_random(self):
        """
        Generate the PyTorch Geometric Dataset using dummy random data. 
        """
        kwargs = {}
        n_features_in = 3 
        n_features_out = 1 
        n_nodes = self.data_reduced.pos.shape[0]
        device_for_loading = 'cpu'

        reduced_graph_dict = self.data_reduced.to_dict()

        n_train = 16
        data_train_list = []
        for i in range(n_train): 
            data_temp = Data(   
                                x = torch.rand((n_nodes, n_features_in)), 
                                y = torch.rand((n_nodes, n_features_out))
                            )
            for key in reduced_graph_dict.keys():
                data_temp[key] = reduced_graph_dict[key]

            data_temp = data_temp.to(device_for_loading)
            data_train_list.append(data_temp)

        n_valid = 16
        data_valid_list = [] 
        for i in range(n_valid):
            data_temp = Data(   
                                x = torch.rand((n_nodes, n_features_in)), 
                                y = torch.rand((n_nodes, n_features_out))
                            )
            for key in reduced_graph_dict.keys():
                data_temp[key] = reduced_graph_dict[key]

            data_temp = data_temp.to(device_for_loading)
            data_valid_list.append(data_temp)

        train_dataset = data_train_list
        test_dataset = data_valid_list 

        # DDP: use DistributedSampler to partition training data
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, num_replicas=SIZE, rank=RANK,
        )
        train_loader = torch_geometric.loader.DataLoader(
            train_dataset,
            batch_size=self.cfg.batch_size,
            sampler=train_sampler,
            **kwargs
        )

        # DDP: use DistributedSampler to partition the test data
        test_sampler = torch.utils.data.distributed.DistributedSampler(
            test_dataset, num_replicas=SIZE, rank=RANK
        )
        test_loader = torch_geometric.loader.DataLoader(
            test_dataset, batch_size=self.cfg.test_batch_size
        )

        return {
            'train': {
                'sampler': train_sampler,
                'loader': train_loader,
                'example': train_dataset[0],
            },
            'test': {
                'sampler': test_sampler,
                'loader': test_loader,
                'example': test_dataset[0],
            }
        }

    def train_step_toy_tgv(self, data: DataBatch) -> Tensor:
        loss = torch.tensor([0.0])
        
        if WITH_CUDA:
            data.x = data.x.cuda() 
            data.y = data.y.cuda()
            data.edge_index = data.edge_index.cuda()
            data.edge_weight = data.edge_weight.cuda()
            data.edge_attr = data.edge_attr.cuda()
            data.pos = data.pos.cuda()
            data.batch = data.batch.cuda()
            loss = loss.cuda()
        
        self.optimizer.zero_grad()
        
        print('[RANK %d] input: data.x.shape: ' %(RANK), data.x.shape)
        print('[RANK %d] target: data.y.shape: ' %(RANK), data.y.shape)
        
        # Prediction 
        out_gnn = self.model(x = data.x, 
                             edge_index = data.edge_index,
                             edge_weight = data.edge_weight,
                             halo_info = data.halo_info,
                             mask_send = self.mask_send,
                             mask_recv = self.mask_recv,
                             buffer_send = self.buffer_send,
                             buffer_recv = self.buffer_recv,
                             neighboring_procs = self.neighboring_procs, 
                             SIZE = SIZE
                             )
        

        #self.print_node_attribute(out_gnn[:data.n_nodes_local], 'out_gnn')

        # Print 
        try:
            w_mp = self.model.module.w_mp
        except: 
            w_mp = self.model.w_mp
        
        # if SIZE > 1: 
        #     w_mp = self.model.module.w_mp
        # else:
        #     w_mp = self.model.w_mp

        # Accumulate loss
        target = data.y

        # # Toy loss: evaluate only at the central node  
        # # Get the index of global id = 2 
        # gid_loss = 14 # degree = 8 (for size = 8) 
        # #gid_loss = 11 # degree = 4 (for size = 8)
        # #gid_loss = 19 # degree = 1 (for size = 8)
        # idx_loss = data.global_ids == gid_loss
        # n_nodes_local = data.n_nodes_local
        # loss = 0.5*self.loss_fn(out_gnn[:n_nodes_local][idx_loss], target[:n_nodes_local][idx_loss]) # Loss = (x_B - y_B)^2
        # #loss = self.loss_fn(out_gnn[:n_nodes_local], target[:n_nodes_local]) # Loss = (x_B - y_B)^2

        # Toy loss: evaluate at all of the nodes 
        n_nodes_local = data.n_nodes_local
        #print('[RANK %d] Output_local:' %(RANK), out_gnn[:n_nodes_local])
        #print('[RANK %d] Node degree: ' %(RANK), data.node_degree[:n_nodes_local])
        #print('[RANK %d] global id: ' %(RANK), data.global_ids[:n_nodes_local])

        if SIZE == 1:
            loss = self.loss_fn(out_gnn[:n_nodes_local], target[:n_nodes_local])
        else: # custom 
            squared_errors_local = torch.pow(out_gnn[:n_nodes_local] - target[:n_nodes_local], 2)
            squared_errors_local = squared_errors_local/data.node_degree[:n_nodes_local].unsqueeze(-1)
            print('[RANK %d] squared_errors_local shape: ' %(RANK), squared_errors_local.shape)

            sum_squared_errors_local = squared_errors_local.sum()
            effective_nodes_local = torch.sum(1.0/data.node_degree[:n_nodes_local])

            print('[RANK %d] sum_squared_errors_local: ' %(RANK), sum_squared_errors_local)
            print('[RANK %d] effective_nodes_local: ' %(RANK), effective_nodes_local)

            effective_nodes = distnn.all_reduce(effective_nodes_local)
            sum_squared_errors = distnn.all_reduce(sum_squared_errors_local)
            loss = (1.0/effective_nodes) * sum_squared_errors

        print('[RANK %d] Loss: ' %(RANK), format(loss.item(), ".6g"))

        loss.backward()
        #self.optimizer.step()

        # Print the backward gradient   
        print('[RANK %d] w_grad_torch: ' %(RANK), w_mp.grad)

        force_abort()
        
        return loss 


    def train_step_toy_1d(self, data: DataBatch) -> Tensor:
        loss = torch.tensor([0.0])
        
        if WITH_CUDA:
            data.x = data.x.cuda() 
            data.edge_index = data.edge_index.cuda()
            data.edge_weight = data.edge_weight.cuda()
            data.halo_info = data.halo_info.cuda()
            data.edge_attr = data.edge_attr.cuda()
            data.pos = data.pos.cuda()
            data.node_degree = data.node_degree.cuda()
            data.batch = data.batch.cuda()
            loss = loss.cuda()
        
        self.optimizer.zero_grad()

        # Prediction 
        #out_gnn = self.model(data.x, data.edge_index, data.edge_attr, data.pos, data.batch)

        self.neighboring_procs = torch.tensor(self.neighboring_procs)

        out_gnn = self.model(x = data.x, 
                             edge_index = data.edge_index,
                             edge_weight = data.edge_weight,
                             halo_info = data.halo_info,
                             mask_send = self.mask_send,
                             mask_recv = self.mask_recv,
                             buffer_send = self.buffer_send,
                             buffer_recv = self.buffer_recv,
                             neighboring_procs = self.neighboring_procs, 
                             SIZE = torch.tensor(SIZE)
                             )
        
        try:
            w_mp = self.model.module.w_mp
        except: 
            w_mp = self.model.w_mp

        #print('\n\n[RANK %d] Input:' %(RANK), data.x)
        #print('[RANK %d] Weight:' %(RANK), w_mp)
        #print('[RANK %d] Output:' %(RANK), out_gnn)

        # Accumulate loss
        target = data.y
        if WITH_CUDA:
            target = target.cuda()

        # # Toy loss: evaluate only at the central node  
        # # Get the index of global id = 2 
        # gid_loss = 2 
        # idx_loss = data.global_ids == gid_loss
        # n_nodes_local = data.n_nodes_local
        # loss = 0.5*self.loss_fn(out_gnn[:n_nodes_local][idx_loss], target[:n_nodes_local][idx_loss]) # Loss = (x_B - y_B)^2

        # Toy loss: evaluate at all of the nodes 
        n_nodes_local = data.n_nodes_local
        #print('[RANK %d] Output_local:' %(RANK), out_gnn[:n_nodes_local])
        #print('[RANK %d] Node degree: ' %(RANK), data.node_degree[:n_nodes_local])
        #print('[RANK %d] global id: ' %(RANK), data.global_ids[:n_nodes_local])

        if SIZE == 1:
            loss = self.loss_fn(out_gnn[:n_nodes_local], target[:n_nodes_local])
        else: # custom 
            squared_errors_local = torch.pow(out_gnn[:n_nodes_local] - target[:n_nodes_local], 2)
            squared_errors_local = squared_errors_local/data.node_degree[:n_nodes_local].unsqueeze(-1)
            #print('[RANK %d] squared_errors_local shape: ' %(RANK), squared_errors_local.shape)

            sum_squared_errors_local = squared_errors_local.sum()
            effective_nodes_local = torch.sum(1.0/data.node_degree[:n_nodes_local])

            #print('[RANK %d] sum_squared_errors_local: ' %(RANK), sum_squared_errors_local)
            #print('[RANK %d] effective_nodes_local: ' %(RANK), effective_nodes_local)

            effective_nodes = distnn.all_reduce(effective_nodes_local)
            sum_squared_errors = distnn.all_reduce(sum_squared_errors_local)
            loss = (1.0/effective_nodes) * sum_squared_errors


        print('[RANK %d] Loss: ' %(RANK), format(loss.item(), ".6g"))
        
        loss.backward()
        #self.optimizer.step()

        # Print the backward gradient   
        print('[RANK %d] w_grad_torch: ' %(RANK), w_mp.grad)

        # # Compute the manual gradient, using contribution only from node 2  
        # w_manual = torch.tensor(w_mp.item())
        # w_grad_manual = None
        # if SIZE == 1:
        #     w_grad_manual = -(data.y[1] - (w_manual * data.x[0] + w_manual * data.x[2])) * (data.x[0] + data.x[2])
        # else:
        #     x_in = [14.3, 10.1, 13.2]
        #     y_target = [3.43, 12.4, 2.23]
        #     w_manual = 0.5 
        #     y_tilde = w_manual * x_in[0] + w_manual * x_in[2]
        #     w_grad_manual = -(y_target[1] - y_tilde) * ( (x_in[0] + x_in[2]) )
        #     
        #     #w_grad_manual = -(y_target[1] - y_tilde) * ( (x_in[0]) ) # half of what rank 0 sees 
        #     #w_grad_manual = -(y_target[1] - y_tilde) * ( (x_in[2]) ) # half of what rank 1 sees 


        # Compute the manual gradient, taking into account all nodes 
        #if SIZE == 1: 
        x_in = [14.3, 10.1, 13.2]
        y_target = [3.43, 12.4, 2.23]
        w_manual = 0.5

        # node 1 
        y_tilde = w_manual * x_in[1] 
        w_grad_1 = -2.0*(y_target[0] - y_tilde) * (x_in[1])

        # ndoe 2 
        y_tilde = w_manual * x_in[0] + w_manual * x_in[2] 
        w_grad_2 = -2.0 * (y_target[1] - y_tilde) * (x_in[0] + x_in[2])

        # node 3 
        y_tilde = w_manual * x_in[1]
        w_grad_3 =  -2.0*(y_target[2] - y_tilde) * (x_in[1])

        w_grad_manual = (1.0/3.0) * (w_grad_1 + w_grad_2 + w_grad_3)

        # elif SIZE == 2: 
        #     x_in = [14.3, 10.1, 13.2]
        #     y_target = [3.43, 12.4, 2.23]
        #     w_manual = 0.5
        #     gid = data.global_ids.tolist()
        #     #print('[RANK %d] gid: ' %(RANK), gid)
        #     if RANK == 0:  
        #         
        #         # node 1 
        #         y_tilde = w_manual * x_in[1]
        #         w_grad_1 = -2.0*(y_target[0] - y_tilde) * (x_in[1]) 

        #         # node 2 
        #         y_tilde = w_manual * x_in[0] + w_manual * x_in[2]
        #         # w_grad_2 = -2.0 * (y_target[1] - y_tilde) * (x_in[0] + x_in[2])
        #         w_grad_2 = -2.0 * (y_target[1] - y_tilde) * (x_in[0])

        #         w_grad_manual = (1.0/2.0)*(w_grad_1 + w_grad_2)
        #         
        #     if RANK == 1:
        #         # node 2
        #         y_tilde = w_manual * x_in[0] + w_manual * x_in[2]
        #         w_grad_2 = -2.0 * (y_target[1] - y_tilde) * (x_in[0] + x_in[2])
        #         w_grad_2 = -2.0 * (y_target[1] - y_tilde) * (x_in[2])

        #         # node 3
        #         y_tilde = w_manual * x_in[1]
        #         w_grad_3 =  -2.0*(y_target[2] - y_tilde) * (x_in[1])

        #         w_grad_manual = (1.0/2.0)*(w_grad_2 + w_grad_3)


        # print('[RANK %d] w_grad_manual: ' %(RANK), w_grad_manual)


        force_abort()

        return loss 

    def train_step(self, data: DataBatch) -> Tensor:
        loss = torch.tensor([0.0])
        
        if WITH_CUDA:
            data.x = data.x.cuda() 
            data.y = data.y.cuda()
            data.edge_index = data.edge_index.cuda()
            data.edge_weight = data.edge_weight.cuda()
            data.edge_attr = data.edge_attr.cuda()
            data.pos = data.pos.cuda()
            data.batch = data.batch.cuda()
            loss = loss.cuda()
        
        self.optimizer.zero_grad()
        
        # Prediction 
        out_gnn = self.model(x = data.x,
                             edge_index = data.edge_index,
                             edge_weight = data.edge_weight,
                             pos = data.pos, 
                             halo_info = data.halo_info,
                             mask_send = self.mask_send,
                             mask_recv = self.mask_recv,
                             buffer_send = self.buffer_send,
                             buffer_recv = self.buffer_recv,
                             neighboring_procs = self.neighboring_procs,
                             SIZE = SIZE,
                             batch = data.batch)


        # Accumulate loss
        target = data.y

        # Toy loss: evaluate at all of the nodes 
        n_nodes_local = data.n_nodes_local

        if SIZE == 1:
            loss = self.loss_fn(out_gnn[:n_nodes_local], target[:n_nodes_local])
        else: # custom 
            n_output_features = out_gnn.shape[1]
            squared_errors_local = torch.pow(out_gnn[:n_nodes_local] - target[:n_nodes_local], 2)
            squared_errors_local = squared_errors_local/data.node_degree[:n_nodes_local].unsqueeze(-1)

            sum_squared_errors_local = squared_errors_local.sum()
            effective_nodes_local = torch.sum(1.0/data.node_degree[:n_nodes_local])


            effective_nodes = distnn.all_reduce(effective_nodes_local)
            sum_squared_errors = distnn.all_reduce(sum_squared_errors_local)
            loss = (1.0/(effective_nodes*n_output_features)) * sum_squared_errors

        loss.backward()
        self.optimizer.step()

        return loss 


    def train_step_verification(self, data: DataBatch) -> Tensor:
        loss = torch.tensor([0.0])
        
        if WITH_CUDA:
            data.x = data.x.cuda() 
            data.y = data.y.cuda()
            data.edge_index = data.edge_index.cuda()
            data.edge_weight = data.edge_weight.cuda()
            data.edge_attr = data.edge_attr.cuda()
            data.pos = data.pos.cuda()
            data.batch = data.batch.cuda()
            data.halo_info = data.halo_info.cuda()
            data.node_degree = data.node_degree.cuda()
            loss = loss.cuda()
        
        self.optimizer.zero_grad()
        
        # # Prediction 
        # log.info(f'[RANK {RANK}] : halo_info : {data.halo_info.device}')
        # log.info(f'[RANK {RANK}] : buffer_send[0] : {self.buffer_send[0].device}')
        # log.info(f'[RANK {RANK}] : buffer_send[1] : {self.buffer_send[1].device}')
        # log.info(f'[RANK {RANK}] : buffer_recv[0] : {self.buffer_recv[0].device}')
        # log.info(f'[RANK {RANK}] : buffer_recv[1] : {self.buffer_recv[1].device}')

        out_gnn = self.model(x = data.x,
                             edge_index = data.edge_index,
                             edge_weight = data.edge_weight,
                             pos = data.pos, 
                             halo_info = data.halo_info,
                             mask_send = self.mask_send,
                             mask_recv = self.mask_recv,
                             buffer_send = self.buffer_send,
                             buffer_recv = self.buffer_recv,
                             neighboring_procs = self.neighboring_procs,
                             SIZE = SIZE,
                             batch = data.batch)

        # Accumulate loss
        target = data.y

        # Toy loss: evaluate at all of the nodes 
        n_nodes_local = data.n_nodes_local

        if SIZE == 1:
            loss = self.loss_fn(out_gnn[:n_nodes_local], target[:n_nodes_local])
        else: # custom 
            n_output_features = out_gnn.shape[1]
            squared_errors_local = torch.pow(out_gnn[:n_nodes_local] - target[:n_nodes_local], 2)
            squared_errors_local = squared_errors_local/data.node_degree[:n_nodes_local].unsqueeze(-1)

            sum_squared_errors_local = squared_errors_local.sum()
            effective_nodes_local = torch.sum(1.0/data.node_degree[:n_nodes_local])


            effective_nodes = distnn.all_reduce(effective_nodes_local)
            sum_squared_errors = distnn.all_reduce(sum_squared_errors_local)
            loss = (1.0/(effective_nodes*n_output_features)) * sum_squared_errors

        log.info('[RANK %d] Loss: %g' %(RANK, loss.item()))

        loss.backward()
        #self.optimizer.step()

        # Print the backward gradient   
        if SIZE == 1:
            model = self.model 

        else:
            model = self.model.module 

        # loop through model parameters 
        grad_dict = {name: param.grad for name, param in model.named_parameters()}
        grad_dict["loss"] = loss.item()
        savepath = self.cfg.work_dir + '/outputs/postproc/gradient_data/tgv_poly_1'
        #savepath = self.cfg.work_dir + '/outputs/postproc/gradient_data/tgv_poly_7'
        torch.save(grad_dict, savepath + '/%s.tar' %(model.get_save_header()))
         
        force_abort()

        return loss 

    def train_epoch(self, epoch: int) -> dict:
        self.model.train()
        start = time.time()
        running_loss = torch.tensor(0.)
        count = torch.tensor(0.)

        if WITH_CUDA:
            running_loss = running_loss.cuda()
            count = count.cuda()

        #train_sampler = self.data['train']['sampler']
        #train_sampler.set_epoch(epoch)

        train_loader = self.data['train']['loader']

        for bidx, data in enumerate(train_loader):
            batch_size = len(data)
            loss = self.train_step_verification(data)
            #loss = self.train_step(data)
            #loss = self.train_step_toy_1d(data)
            #loss = self.train_step_toy_tgv(data)
            running_loss += loss.item()
            count += 1 # accumulate current batch count
            self.training_iter += 1 # accumulate total training iteration

            # Log on Rank 0:
            if bidx % self.cfg.logfreq == 0 and RANK == 0:
                metrics = {
                    'epoch': epoch,
                    'dt': time.time() - start,
                    'batch_loss': loss.item(),
                    'running_loss': running_loss,
                }
                pre = [
                    f'[{RANK}]',
                    (   # looks like: [num_processed/total (% complete)]
                        f'[{epoch}/{self.cfg.epochs}:'
                        f' Batch {bidx+1}'
                        f' ({100. * (bidx+1) / len(train_loader):.0f}%)]'
                    ),
                ]
                log.info(' '.join([
                    *pre, *[f'{k}={v:.4f}' for k, v in metrics.items()]
                ]))

        # divide running loss by number of batches
        running_loss = running_loss / count
        loss_avg = metric_average(running_loss)

        return {'loss': loss_avg}

    def test(self) -> dict:
        running_loss = torch.tensor(0.)
        count = torch.tensor(0.)
        if WITH_CUDA:
            running_loss = running_loss.cuda()
            count = count.cuda()
        self.model.eval()
        test_loader = self.data['test']['loader']

        with torch.no_grad():
            for data in test_loader:
                loss = torch.tensor([0.0])
                
                if WITH_CUDA:
                    data.x = data.x.cuda()
                    data.edge_index = data.edge_index.cuda()
                    data.edge_attr = data.edge_attr.cuda()
                    data.pos = data.pos.cuda()
                    data.batch = data.batch.cuda()
                    loss = loss.cuda()
                
                # Prediction 
                out_gnn = self.model(data.x, data.edge_index, data.edge_attr, data.pos, data.batch)

                # Accumulate loss
                target = data.y
                if WITH_CUDA:
                    target = target.cuda()

                # Standard loss 
                loss = self.loss_fn(out_gnn, target)
                
                running_loss += loss.item()
                count += 1

            running_loss = running_loss / count
            loss_avg = metric_average(running_loss)

        return {'loss': loss_avg}


def train(cfg: DictConfig) -> None:
    start = time.time()
    trainer = Trainer(cfg)
    epoch_times = []

    # # Get the sample 
    # sample = trainer.data['train']['example']

    # print('[RANK = %d] x\n' %(RANK), sample.x)
    # print('[RANK = %d] y\n' %(RANK), sample.y)
    # print('[RANK = %d] halo_info' %(RANK), sample.halo_info)
    # print('[RANK = %d] local_unique_mask' %(RANK), sample.local_unique_mask)
    # print('[RANK = %d] halo_unique_mask' %(RANK), sample.halo_unique_mask)

    # # Test the halo swap 
    # # get masks  
    # mask_send, mask_recv = trainer.build_masks()

    # # build buffers 
    # n_features_x = sample.x.shape[1]
    # buffer_send, buffer_recv = trainer.build_buffers(n_features_x)

    # # do the swap 
    # print('[RANK = %d] x before: ' %(RANK), sample.x)

    # sample.x = trainer.halo_swap(sample.x, buffer_send, buffer_recv)

    # print('[RANK = %d] x after swap: ' %(RANK), sample.x)

    # # do the scatter 
    # print('[RANK = %d] halo info: ' %(RANK), sample.halo_info)

    # # 1) get the local nodes which will receive the scatter 
    # idx_recv = sample.halo_info[:,0]
    # idx_send = sample.halo_info[:,1]

    # # 2) perform the accumulation 
    # # Perform the scatter operation using index_add_
    # sample.x.index_add_(0, idx_recv, sample.x.index_select(0, idx_send))
    # 
    # print('[RANK = %d] x after scatter: ' %(RANK), sample.x)


    for epoch in range(trainer.epoch_start, cfg.epochs+1):
        # ~~~~ Training step 
        t0 = time.time()
        trainer.epoch = epoch
        train_metrics = trainer.train_epoch(epoch)
        trainer.loss_hist_train[epoch-1] = train_metrics["loss"]

        epoch_time = time.time() - t0
        epoch_times.append(epoch_time)

        # ~~~~ Validation step
        test_metrics = trainer.test()
        trainer.loss_hist_test[epoch-1] = test_metrics["loss"]
        
        # ~~~~ Printing
        if RANK == 0:
            astr = f'[TEST] loss={test_metrics["loss"]:.4e}'
            sepstr = '-' * len(astr)
            log.info(sepstr)
            log.info(astr)
            log.info(sepstr)
            summary = '  '.join([
                '[TRAIN]',
                f'loss={train_metrics["loss"]:.4e}',
                f'epoch_time={epoch_time:.4g} sec'
            ])
            log.info((sep := '-' * len(summary)))
            log.info(summary)
            log.info(sep)

        # ~~~~ Step scheduler based on validation loss
        trainer.scheduler.step(test_metrics["loss"])

        # ~~~~ Checkpointing step 
        if epoch % cfg.ckptfreq == 0 and RANK == 0:
            astr = 'Checkpointing on root processor, epoch = %d' %(epoch)
            sepstr = '-' * len(astr)
            log.info(sepstr)
            log.info(astr)
            log.info(sepstr)

            if not os.path.exists(cfg.ckpt_dir):
                os.makedirs(cfg.ckpt_dir)

            if WITH_DDP and SIZE > 1:
                sd = trainer.model.module.state_dict()
            else:
                sd = trainer.model.state_dict()

            ckpt = {'epoch' : epoch,
                    'training_iter' : trainer.training_iter,
                    'model_state_dict' : sd,
                    'optimizer_state_dict' : trainer.optimizer.state_dict(),
                    'scheduler_state_dict' : trainer.scheduler.state_dict(),
                    'loss_hist_train' : trainer.loss_hist_train,
                    'loss_hist_test' : trainer.loss_hist_test}
            
            torch.save(ckpt, trainer.ckpt_path)
        dist.barrier()

    rstr = f'[{RANK}] ::'
    log.info(' '.join([
        rstr,
        f'Total training time: {time.time() - start} seconds'
    ]))
    
    if RANK == 0:
        if WITH_CUDA:
            trainer.model.to('cpu')
        if not os.path.exists(cfg.model_dir):
            os.makedirs(cfg.model_dir)

        if WITH_DDP and SIZE > 1:
            sd = trainer.model.module.state_dict()
            ind = trainer.model.module.input_dict()
        else:
            sd = trainer.model.state_dict()
            ind = trainer.model.input_dict()

        save_dict = {
                    'state_dict' : sd,
                    'input_dict' : ind,
                    'loss_hist_train' : trainer.loss_hist_train,
                    'loss_hist_test' : trainer.loss_hist_test,
                    'training_iter' : trainer.training_iter
                    }
        
        torch.save(save_dict, trainer.model_path)

    # Plot connectivity
    if (cfg.plot_connectivity):
        gplot.plot_graph(trainer.data['train']['example'], RANK, cfg.work_dir)

def halo_test(cfg: DictConfig) -> None:
    trainer = Trainer(cfg)

    # Get the sample 
    sample = trainer.data['train']['example']

    # print('[RANK = %d] x\n' %(RANK), sample.x)
    # print('[RANK = %d] y\n' %(RANK), sample.y)
    # print('[RANK = %d] halo_info' %(RANK), sample.halo_info)
    # print('[RANK = %d] local_unique_mask' %(RANK), sample.local_unique_mask)
    # print('[RANK = %d] halo_unique_mask' %(RANK), sample.halo_unique_mask)

    n_nodes_local = sample.n_nodes_local
    #trainer.print_node_attribute(sample.x[:n_nodes_local,:], 'x')


    # Test the halo swap 
    # get masks  
    mask_send, mask_recv = trainer.build_masks()

    # build buffers 
    n_features_x = sample.x.shape[1]
    buffer_send, buffer_recv = trainer.build_buffers(n_features_x)
    
    # fill the buffers 
    input_tensor = sample.x 
    for i in trainer.neighboring_procs:
        n_send = len(mask_send[i])
        buffer_send[i][:n_send,:] = input_tensor[mask_send[i]]
        print('[RANK = %d] \t i = %d \t n_send = %d \t buffer_send[i].shape: ' %(RANK,i,n_send), buffer_send[i].shape)

    # # do the swap 
    # #trainer.print_node_attribute(sample.x, 'x before')
    # print('[RANK = %d] x before swap: ' %(RANK), sample.x)
    # sample.x = trainer.halo_swap(sample.x, buffer_send, buffer_recv)
    # print('[RANK = %d] x after swap: ' %(RANK), sample.x)

    # ~~~~ ALL to ALL based swap
    # First step: fill the buffers 
    if SIZE > 1:

        # Perform all to all
        distnn.all_to_all(buffer_recv, buffer_send)

        # Fill halo nodes 
        for i in trainer.neighboring_procs:
            n_recv = len(mask_recv[i])
            sample.x[mask_recv[i]] = buffer_recv[i][:n_recv,:]

    #trainer.print_node_attribute(sample.x[:n_nodes_local,:], 'x')


@hydra.main(version_base=None, config_path='./conf', config_name='config')
def main(cfg: DictConfig) -> None:
    print('Rank %d, local rank %d, which has device %s. Sees %d devices.' %(RANK,int(LOCAL_RANK),DEVICE,torch.cuda.device_count()))

    train(cfg)
    #halo_test(cfg)
    
    cleanup()

if __name__ == '__main__':
    main()
