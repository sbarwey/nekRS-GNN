from __future__ import absolute_import, division, print_function, annotations
from typing import Optional, Union, Callable, List
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as tgnn
from torch_scatter import scatter_mean
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj, OptTensor, PairTensor
from pooling import TopKPooling_Mod, avg_pool_mod, avg_pool_mod_no_x
import torch.distributed as dist
import torch.distributed.nn as distnn


class toy_gnn_distributed(torch.nn.Module):
    """
    Toy GNN for testing distributed backpropagation.
    There is only one parameter: w_mp, a scalar edge weight used in edge aggregation.
    """
    def __init__(self):
        super().__init__()
        self.edge_aggregator = EdgeAggregation(aggr='add') # for edge aggregation 
        #self.w_mp = nn.Parameter(torch.randn(1)) # GNN parameter 
        self.w_mp = nn.Parameter(torch.tensor([0.5])) # GNN parameter 

    def forward(
            self,
            x: Tensor,
            edge_index: LongTensor,
            edge_weight: Tensor,
            halo_info: Tensor,
            mask_send: list,
            mask_recv: list,
            buffer_send: Tensor,
            buffer_recv: Tensor,
            neighboring_procs: Tensor,
            SIZE: Tensor,
            batch: Optional[LongTensor] = None) -> Tensor:

        if batch is None:
            batch = edge_index.new_zeros(x.size(0))
        
        # ~~~~ Edge update:  
        x_own = x[edge_index[1,:], :]
        x_nei = x[edge_index[0,:], :] 
        ea = self.w_mp * x_nei 

        # Scale by edge weights 
        edge_weight = edge_weight.unsqueeze(1)
        ea = ea * edge_weight

        # ~~~~ Edge aggregation: sums ea using local neighborhoods 
        edge_agg = self.edge_aggregator(x, edge_index, ea)
        
        if SIZE > 1:
            # ~~~~ Halo exchange: swap the edge aggregates. This populates the halo nodes  
            edge_agg = self.halo_swap(edge_agg, 
                                      mask_send,
                                      mask_recv,
                                      buffer_send, 
                                      buffer_recv, 
                                      neighboring_procs, 
                                      SIZE)

            # ~~~~ Local scatter using halo nodes (use halo_info) 
            idx_recv = halo_info[:,0]
            idx_send = halo_info[:,1]
            edge_agg.index_add_(0, idx_recv, edge_agg.index_select(0, idx_send))

        # ~~~~ Node update (node becomes result of edge aggregation in this simple model) 
        x = edge_agg
        
        return x 


    def halo_swap(self, 
                  input_tensor, 
                  mask_send, 
                  mask_recv, 
                  buff_send, 
                  buff_recv, 
                  neighboring_procs, 
                  SIZE):
        """
        Performs halo swap using send/receive buffers
        """
        if SIZE > 1:

            # all_to_all:
            # Fill send buffer
            for i in neighboring_procs:
                n_send = len(mask_send[i])
                buff_send[i][:n_send,:] = input_tensor[mask_send[i]]

            # Perform all_to_all
            distnn.all_to_all(buff_recv, buff_send)
        
            # Fill halo nodes
            for i in neighboring_procs:
                n_recv = len(mask_recv[i])
                input_tensor[mask_recv[i]] = buff_recv[i][:n_recv,:]

            # # all_to_all: old 
            # # Fill send buffer
            # for i in neighboring_procs:
            #     buff_send[i] = input_tensor[mask_send[i]]

            # # Update send buffer 
            # for i in range(len(buff_send)):
            #     #if buff_send[i] is None:
            #     if len(buff_send[i]) == 0:
            #         buff_send[i] = torch.tensor([[0.0]], device=input_tensor.device)

            # for i in range(len(buff_recv)):
            #     #if buff_recv[i] is None:
            #     if len(buff_recv[i]) == 0:
            #         buff_recv[i] = torch.tensor([[0.0]], device=input_tensor.device)

            # # Perform all_to_all
            # distnn.all_to_all(buff_recv, buff_send)
        
            # # Fill halo nodes
            # for i in neighboring_procs:
            #     input_tensor[mask_recv[i]] = buff_recv[i]


            # # OLD : using isend irecv, faster but not differentiable  
            # # Fill send buffer
            # for i in neighboring_procs:
            #     buff_send[i] = input_tensor[mask_send[i]]

            # # Perform swap
            # req_send_list = []
            # for i in neighboring_procs:
            #     req_send = dist.isend(tensor=buff_send[i], dst=i)
            #     req_send_list.append(req_send)

            # req_recv_list = []
            # for i in neighboring_procs:
            #     req_recv = dist.irecv(tensor=buff_recv[i], src=i)
            #     req_recv_list.append(req_recv)

            # for req_send in req_send_list:
            #     req_send.wait()

            # for req_recv in req_recv_list:
            #     req_recv.wait()

            # dist.barrier()

            # # Fill halo nodes
            # for i in neighboring_procs:
            #     input_tensor[mask_recv[i]] = buff_recv[i]


        return input_tensor

    def input_dict(self) -> dict:
        a = {}
        return a

class toy_gnn(torch.nn.Module):
    """
    Toy GNN for testing distributed backpropagation.
    There is only one parameter: w_mp, a scalar edge weight used in edge aggregation.
    """
    def __init__(self):
        super().__init__()
        self.edge_aggregator = EdgeAggregation(aggr='add') # for edge aggregation 
        #self.w_mp = nn.Parameter(torch.randn(1)) # GNN parameter 
        self.w_mp = nn.Parameter(torch.tensor([0.5])) # GNN parameter 

    def forward(
            self,
            x: Tensor,
            edge_index: LongTensor,
            edge_attr: Tensor,
            pos: Tensor,
            batch: Optional[LongTensor] = None) -> Tensor:

        if batch is None:
            batch = edge_index.new_zeros(x.size(0))
        
        # ~~~~ Edge update:  
        x_nei = x[edge_index[0,:], :] 
        ea = self.w_mp * x_nei 

        # ~~~~ Edge aggregation: sums ea 
        edge_agg = self.edge_aggregator(x, edge_index, ea)

        # ~~~~ Node update (node becomes result of edge aggregation in this simple model) 
        x = edge_agg
        
        return x 

    def input_dict(self) -> dict:
        a = {}
        return a

class mp_gnn_distributed(torch.nn.Module):
    def __init__(self, 
                 input_channels: int, 
                 hidden_channels: int, 
                 output_channels: int, 
                 n_mlp_layers: List[int], 
                 n_messagePassing_layers: int,
                 activation: Callable,
                 halo_swap_mode: Optional[str] = 'all_to_all',
                 name: Optional[str] = 'model'):
        super().__init__()
        
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.output_channels = output_channels 
        self.n_mlp_layers = n_mlp_layers
        self.n_messagePassing_layers = n_messagePassing_layers
        self.act = activation
        self.halo_swap_mode = halo_swap_mode
        self.name = name 

        # ~~~~ node encoder 
        self.node_encoder = torch.nn.ModuleList()
        for j in range(self.n_mlp_layers[0]):
            if j == 0:
                input_features = self.input_channels
                output_features = self.hidden_channels 
            else:
                input_features = self.hidden_channels
                output_features = self.hidden_channels
            self.node_encoder.append( nn.Linear(input_features, output_features, bias=True) )

        # ~~~~ node decoder 
        self.node_decoder = torch.nn.ModuleList()
        for j in range(self.n_mlp_layers[0]):
            if j == self.n_mlp_layers[0] - 1:
                input_features = self.hidden_channels
                output_features = self.output_channels
            else:
                input_features = self.hidden_channels
                output_features = self.hidden_channels
            self.node_decoder.append( nn.Linear(input_features, output_features, bias=True) )

        # ~~~~ message passing layer 
        self.mp_layers = torch.nn.ModuleList()
        for j in range(self.n_messagePassing_layers):
            self.mp_layers.append( mp_layer_distributed(channels = hidden_channels,
                                             n_mlp_layers_edge = self.n_mlp_layers[1], 
                                             n_mlp_layers_node = self.n_mlp_layers[2],
                                             activation = self.act,
                                             halo_swap_mode = self.halo_swap_mode) 
                                  )
        
        self.reset_parameters()

    def forward(
            self,
            x: Tensor,
            edge_index: LongTensor,
            edge_weight: Tensor,
            pos: Tensor,
            halo_info: Tensor,
            mask_send: list,
            mask_recv: list,
            buffer_send: Tensor,
            buffer_recv: Tensor,
            neighboring_procs: Tensor,
            SIZE: Tensor,
            batch: Optional[LongTensor] = None) -> Tensor:
        
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))
        
        # ~~~~ Node Encoder: 
        n_layers = self.n_mlp_layers[0]
        for i in range(n_layers):
            x = self.node_encoder[i](x)
            if i < n_layers - 1:
                x = self.act(x)

        # ~~~~ Message passing with residual connection:  
        for i in range(self.n_messagePassing_layers):
            x = x + self.mp_layers[i](  x,
                                        edge_index,
                                        edge_weight,
                                        pos,
                                        halo_info,
                                        mask_send,
                                        mask_recv,
                                        buffer_send,
                                        buffer_recv,
                                        neighboring_procs,
                                        SIZE,
                                        batch)
        
        # ~~~~ Node decoder:
        n_layers = self.n_mlp_layers[0]
        for i in range(n_layers):
            x = self.node_decoder[i](x)
            if i < n_layers - 1:
                x = self.act(x)

        return x 

    def reset_parameters(self):
        for module in self.node_encoder:
            module.reset_parameters()

        for module in self.node_decoder:
            module.reset_parameters()

        for module in self.mp_layers:
            module.reset_parameters()
            
        return

    def input_dict(self) -> dict:
        a = {'input_channels': self.input_channels,
             'hidden_channels': self.hidden_channels,
             'output_channels': self.output_channels,
             'n_mlp_layers': self.n_mlp_layers,
             'n_messagePassing_layers': self.n_messagePassing_layers,
             'activation': self.act,
             'halo_swap_mode': self.halo_swap_mode,
             'name': self.name}
        return a

    def get_save_header(self) -> str:
        header = '%s' %(self.name)
        header += '_input_channels_%d' %(self.input_channels)
        header += '_hidden_channels_%d' %(self.hidden_channels)
        header += '_output_channels_%d' %(self.output_channels)
        header += '_nMessagePassingLayers_%d' %(self.n_messagePassing_layers)
        header += '_halo_%s' %(self.halo_swap_mode)
        return header
 
class mp_layer_distributed(torch.nn.Module):
    def __init__(self, 
                 channels: int, 
                 n_mlp_layers_edge: int, 
                 n_mlp_layers_node: int,
                 activation: Callable,
                 halo_swap_mode: str):
        super().__init__()

        self.edge_aggregator = EdgeAggregation(aggr='add')
        self.channels = channels
        self.n_mlp_layers_edge = n_mlp_layers_edge
        self.n_mlp_layers_node = n_mlp_layers_node
        self.act = activation
        self.halo_swap_mode = halo_swap_mode

        self.edge_updater = torch.nn.ModuleList()
        for j in range(self.n_mlp_layers_edge):
            if j == 0:
                input_features = self.channels*3 + 3 # additional 3 for node positions  
                output_features = self.channels 
            else:
                input_features = self.channels
                output_features = self.channels
            self.edge_updater.append( nn.Linear(input_features, output_features, bias=True) )

        self.node_updater = torch.nn.ModuleList()
        for j in range(self.n_mlp_layers_node):
            if j == 0:
                input_features = self.channels*2
                output_features = self.channels 
            else:
                input_features = self.channels
                output_features = self.channels
            self.node_updater.append( nn.Linear(input_features, output_features, bias=True) )

        self.reset_parameters()

    def forward(
            self,
            x: Tensor,
            edge_index: LongTensor,
            edge_weight: Tensor,
            pos: Tensor,
            halo_info: Tensor,
            mask_send: list,
            mask_recv: list,
            buffer_send: list,
            buffer_recv: list,
            neighboring_procs: Tensor,
            SIZE: Tensor,
            batch: Optional[LongTensor] = None) -> Tensor:
        """
        Includes call to halo swap
        """
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))
        
        # ~~~~ Edge update 
        x_nei = x[edge_index[0,:], :] 
        x_own = x[edge_index[1,:], :] 
        pos_nei = pos[edge_index[0,:], :]
        pos_own = pos[edge_index[1,:], :] 
        ea = torch.cat((x_nei, x_own, x_nei - x_own, pos_nei - pos_own), dim=1)
        n_layers = self.n_mlp_layers_edge
        for j in range(n_layers):
            ea = self.edge_updater[j](ea) 
            if j < n_layers - 1:
                ea = self.act(ea)

        # Scale by edge weights
        edge_weight = edge_weight.unsqueeze(1)
        ea = ea * edge_weight

        # ~~~~ Local edge aggregation 
        edge_agg = self.edge_aggregator(x, edge_index, ea)

        if SIZE > 1:
            # ~~~~ Halo exchange: swap the edge aggregates. This populates the halo nodes  
            edge_agg = self.halo_swap(edge_agg, 
                                      mask_send,
                                      mask_recv,
                                      buffer_send, 
                                      buffer_recv, 
                                      neighboring_procs, 
                                      SIZE)

            # ~~~~ Local scatter using halo nodes (use halo_info) 
            idx_recv = halo_info[:,0]
            idx_send = halo_info[:,1]
            edge_agg.index_add_(0, idx_recv, edge_agg.index_select(0, idx_send))

        # ~~~~ Node update 
        x = torch.cat((x, edge_agg), dim=1)
        n_layers = self.n_mlp_layers_node
        for j in range(n_layers):
            x = self.node_updater[j](x) 
            if j < n_layers - 1:
                x = self.act(x)

        return x  

    def halo_swap(self, 
                  input_tensor, 
                  mask_send, 
                  mask_recv, 
                  buff_send, 
                  buff_recv, 
                  neighboring_procs, 
                  SIZE):
        """
        Performs halo swap using send/receive buffers
        uses all_to_all implementation
        """
        if SIZE > 1:
            if self.halo_swap_mode == 'all_to_all':
                # Fill send buffer
                for i in neighboring_procs:
                    n_send = len(mask_send[i])
                    buff_send[i][:n_send,:] = input_tensor[mask_send[i]]

                # Perform all_to_all
                distnn.all_to_all(buff_recv, buff_send)
            
                # Fill halo nodes
                for i in neighboring_procs:
                    n_recv = len(mask_recv[i])
                    input_tensor[mask_recv[i]] = buff_recv[i][:n_recv,:]

            elif self.halo_swap_mode == 'sendrecv':
                # Fill send buffer
                for i in neighboring_procs:
                    n_send = len(mask_send[i])
                    buff_send[i][:n_send,:] = input_tensor[mask_send[i]]

                # Perform swap
                req_send_list = []
                for i in neighboring_procs:
                    req_send = dist.isend(tensor=buff_send[i], dst=i)
                    req_send_list.append(req_send)

                req_recv_list = []
                for i in neighboring_procs:
                    req_recv = dist.irecv(tensor=buff_recv[i], src=i)
                    req_recv_list.append(req_recv)

                for req_send in req_send_list:
                    req_send.wait()

                for req_recv in req_recv_list:
                    req_recv.wait()

                dist.barrier()

                # Fill halo nodes
                for i in neighboring_procs:
                    n_recv = len(mask_recv[i])
                    input_tensor[mask_recv[i]] = buff_recv[i][:n_recv,:]
            elif self.halo_swap_mode == 'none':
                pass
            else:
                raise ValueError("halo_swap_mode %s not valid. Valid options: all_to_all, sendrecv" %(self.halo_swap_mode))

        return input_tensor

    def reset_parameters(self):
        for module in self.edge_updater:
            module.reset_parameters()

        for module in self.node_updater:
            module.reset_parameters()
        return

class mp_gnn(torch.nn.Module):
    def __init__(self, 
                 input_channels: int, 
                 hidden_channels: int, 
                 output_channels: int, 
                 n_mlp_layers: List[int], 
                 n_messagePassing_layers: int,
                 activation: Callable,
                 name: Optional[str] = 'model'):
        super().__init__()
        
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.output_channels = output_channels 
        self.n_mlp_layers = n_mlp_layers
        self.n_messagePassing_layers = n_messagePassing_layers
        self.act = activation
        self.name = name 

        # ~~~~ node encoder 
        self.node_encoder = torch.nn.ModuleList()
        for j in range(self.n_mlp_layers[0]):
            if j == 0:
                input_features = self.input_channels
                output_features = self.hidden_channels 
            else:
                input_features = self.hidden_channels
                output_features = self.hidden_channels
            self.node_encoder.append( nn.Linear(input_features, output_features, bias=True) )

        # ~~~~ node decoder 
        self.node_decoder = torch.nn.ModuleList()
        for j in range(self.n_mlp_layers[0]):
            if j == self.n_mlp_layers[0] - 1:
                input_features = self.hidden_channels
                output_features = self.output_channels
            else:
                input_features = self.hidden_channels
                output_features = self.hidden_channels
            self.node_decoder.append( nn.Linear(input_features, output_features, bias=True) )

        # ~~~~ message passing layer 
        self.mp_layers = torch.nn.ModuleList()
        for j in range(self.n_messagePassing_layers):
            self.mp_layers.append( mp_layer(channels = hidden_channels,
                                             n_mlp_layers_edge = self.n_mlp_layers[1], 
                                             n_mlp_layers_node = self.n_mlp_layers[2],
                                             activation = self.act) 
                                  )
        
        self.reset_parameters()

    def forward(
            self,
            x: Tensor,
            edge_index: LongTensor,
            pos: Tensor,
            batch: Optional[LongTensor] = None) -> Tensor:

        if batch is None:
            batch = edge_index.new_zeros(x.size(0))
        
        # ~~~~ Node Encoder: 
        n_layers = self.n_mlp_layers[0]
        for i in range(n_layers):
            x = self.node_encoder[i](x)
            if i < n_layers - 1:
                x = self.act(x)

        # ~~~~ Message passing with residual:  
        for i in range(self.n_messagePassing_layers):
            x = x + self.mp_layers[i](x, edge_index, pos, batch)
        
        # ~~~~ Node decoder:
        n_layers = self.n_mlp_layers[0]
        for i in range(n_layers):
            x = self.node_decoder[i](x)
            if i < n_layers - 1:
                x = self.act(x)

        return x 

    def reset_parameters(self):
        for module in self.node_encoder:
            module.reset_parameters()

        for module in self.node_decoder:
            module.reset_parameters()

        for module in self.mp_layers:
            module.reset_parameters()
            
        return

    def input_dict(self) -> dict:
        a = {'input_channels': self.input_channels,
             'hidden_channels': self.hidden_channels,
             'output_channels': self.output_channels,
             'n_mlp_layers': self.n_mlp_layers,
             'n_messagePassing_layers': self.n_messagePassing_layers,
             'activation': self.act,
             'name': self.name}
        return a

    def get_save_header(self) -> str:
        header = '%s' %(self.name)
        return header

 
class mp_layer(torch.nn.Module):
    def __init__(self, 
                 channels: int, 
                 n_mlp_layers_edge: int, 
                 n_mlp_layers_node: int,
                 activation: Callable):
        super().__init__()

        self.edge_aggregator = EdgeAggregation(aggr='add')
        self.channels = channels
        self.n_mlp_layers_edge = n_mlp_layers_edge
        self.n_mlp_layers_node = n_mlp_layers_node
        self.act = activation

        self.edge_updater = torch.nn.ModuleList()
        for j in range(self.n_mlp_layers_edge):
            if j == 0:
                input_features = self.channels*3 + 3 # additional 3 for node positions  
                output_features = self.channels 
            else:
                input_features = self.channels
                output_features = self.channels
            self.edge_updater.append( nn.Linear(input_features, output_features, bias=True) )

        self.node_updater = torch.nn.ModuleList()
        for j in range(self.n_mlp_layers_node):
            if j == 0:
                input_features = self.channels*2
                output_features = self.channels 
            else:
                input_features = self.channels
                output_features = self.channels
            self.node_updater.append( nn.Linear(input_features, output_features, bias=True) )

        self.reset_parameters()

    def forward(
            self,
            x: Tensor,
            edge_index: LongTensor,
            pos: Tensor,
            batch: Optional[LongTensor] = None) -> Tensor:

        if batch is None:
            batch = edge_index.new_zeros(x.size(0))
        
        # ~~~~ Edge update 
        x_nei = x[edge_index[0,:], :] 
        x_own = x[edge_index[1,:], :] 
        pos_nei = pos[edge_index[0,:], :]
        pos_own = pos[edge_index[1,:], :] 
        ea = torch.cat((x_nei, x_own, x_nei - x_own, pos_nei - pos_own), dim=1)
        n_layers = self.n_mlp_layers_edge
        for j in range(n_layers):
            ea = self.edge_updater[j](ea) 
            if j < n_layers - 1:
                ea = self.act(ea)

        edge_agg = self.edge_aggregator(x, edge_index, ea)

        x = torch.cat((x, edge_agg), dim=1)
        n_layers = self.n_mlp_layers_node
        for j in range(n_layers):
            x = self.node_updater[j](x) 
            if j < n_layers - 1:
                x = self.act(x)

        return x  

    def reset_parameters(self):
        for module in self.edge_updater:
            module.reset_parameters()

        for module in self.node_updater:
            module.reset_parameters()
        return


class EdgeAggregation(MessagePassing):
    r"""This is a custom class that returns node quantities that represent the neighborhood-averaged edge features.
    Args:
        edge_dim (int, optional): Edge feature dimensionality. If set to
            :obj:`None`, node and edge feature dimensionality is expected to
            match. Other-wise, edge features are linearly transformed to match
            node feature dimensionality. (default: :obj:`None`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.

    Shapes: 
        - **input:**
          node features :math:`(|\mathcal{V}|, F_{in})` or 
          :math:`((|\mathcal{V_s}|, F_{s}), (|\mathcal{V_t}|, F_{t}))`
          if bipartite,
          edge indices :math:`(2, |\mathcal{E}|)`, 
          edge features :math:`(|\mathcal{E}|, D)` *(optional)*
        - **output:** node features :math:`(|\mathcal{V}|, F_{out})` or
          :math:`(|\mathcal{V}_t|, F_{out})` if bipartite
    """

    propagate_type = {'x': Tensor, 'edge_attr': Tensor}

    def __init__(self, **kwargs):
        kwargs.setdefault('aggr', 'mean')
        super().__init__(**kwargs)

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tensor:
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=None)
        return out

    def message(self, x_j: Tensor, edge_attr: Tensor) -> Tensor:
        x_j = edge_attr
        return x_j

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}'
