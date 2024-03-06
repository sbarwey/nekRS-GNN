import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LogNorm
import matplotlib.cm as cm


plt.rcParams.update({'font.size': 22})

import torch

def get_grad_data(SIZE, keys, halo_mode_list):
    DATA_FULL = []
    for RANK in range(SIZE):
        data_rank = {}
        for halo_mode in halo_mode_list:
            model_name = 'RANK_%d_SIZE_%d_input_channels_3_hidden_channels_32_output_channels_3_nMessagePassingLayers_5_halo_%s.tar' %(RANK,SIZE,halo_mode)
            a = torch.load(data_path + '/' + model_name, map_location=torch.device('cpu'))
            
            # loss 
            loss = a['loss']
            
            # grad
            grad_full = torch.tensor([])
            for key in keys:
                grad = a[key].flatten()
                grad_full = torch.cat((grad_full, grad))

            data_rank[halo_mode] = {}
            data_rank[halo_mode]['loss'] = loss 
            data_rank[halo_mode]['grad'] = grad_full
            
        DATA_FULL.append(data_rank)
    return DATA_FULL 

if __name__ == "__main__":

    if 1 == 0:
        """
        Load model and plot loss 
        """
        a = torch.load('saved_models/model.tar')
        loss_train = a['loss_hist_train']

        epochs = np.arange(1, len(loss_train)+1)


        fig, ax = plt.subplots()
        ax.plot(epochs, loss_train)
        ax.set_xlabel('Iterations')
        ax.set_ylabel('Loss')
        ax.set_title('Training Demo -- Single Snapshot (Periodic Hill)')
        plt.show(block=False)

    if 1 == 0:
        """
        Model weight comparisons -- are the models the same? 
        """
        data_path = './outputs/postproc/models/tgv_poly_1'
        SIZE_LIST = [1,2,4,8,16]
        halo_mode = 'all_to_all'
        DATA_FULL = []
        for SIZE in SIZE_LIST:
            for RANK in range(SIZE):
                model_name = 'RANK_%d_SIZE_%d_input_channels_3_hidden_channels_32_output_channels_3_nMessagePassingLayers_5_halo_%s.tar' %(RANK,SIZE,halo_mode)
                a = torch.load(data_path + '/' + model_name, map_location=torch.device('cpu'))['state_dict']
                params_full = torch.tensor([])
                for key in a.keys():
                    param = a[key].flatten()
                    params_full = torch.cat((params_full, param))

                DATA_FULL.append(params_full.sum())

        fig, ax = plt.subplots()
        ax.plot(DATA_FULL, marker='o')
        plt.show(block=False)

    if 1 == 0: 
        """
        Looking at consistency QoIs -- data produced from train_step_verification in main.py  
        """
        # no cos(pos), no edge fix 
        # path_32 = "./outputs/postproc/real_gnn/periodic_after_fix/gradient_data_cpu_nondeterministic_LOCAL/tgv_poly_1/float32"
        # path_64 = "./outputs/postproc/real_gnn/periodic_after_fix/gradient_data_cpu_nondeterministic_LOCAL/tgv_poly_1/float64"

        # with pos=0, with edge fix 
        # path_32 = "./outputs/postproc/real_gnn_test/periodic_after_fix_edges_2/gradient_data_cpu_nondeterministic_LOCAL/tgv_poly_1/float32"
        # path_64 = "./outputs/postproc/real_gnn_test/periodic_after_fix_edges_2/gradient_data_cpu_nondeterministic_LOCAL/tgv_poly_1/float64"

        # with cos(pos), with edge fix 
        # path_32 = "./outputs/postproc/real_gnn_test_2/periodic_after_fix_edges_2/gradient_data_cpu_nondeterministic_LOCAL/tgv_poly_1/float32"
        # path_64 = "./outputs/postproc/real_gnn_test_2/periodic_after_fix_edges_2/gradient_data_cpu_nondeterministic_LOCAL/tgv_poly_1/float64"

        # with cos(pos), with edge fix, with binary read 
        # path_32 = "./outputs/postproc/real_gnn_test_3/periodic_after_fix_edges_2/gradient_data_cpu_nondeterministic_LOCAL/tgv_poly_1/float32"
        # path_64 = "./outputs/postproc/real_gnn_test_3/periodic_after_fix_edges_2/gradient_data_cpu_nondeterministic_LOCAL/tgv_poly_1/float64"

        # with cos(pos), with edge fix, with binary read -- polaris 
        # path_32 = "./outputs/postproc/real_gnn_test_3/periodic_after_fix_edges_2/gradient_data_gpu_nondeterministic_POLARIS/tgv_poly_5/float32"
        # path_64 = "./outputs/postproc/real_gnn_test_3/periodic_after_fix_edges_2/gradient_data_gpu_nondeterministic_POLARIS/tgv_poly_5/float32"

        # new gnn 
        path_32 = "./outputs/postproc/real_gnn_test_4/periodic_after_fix_edges_2/gradient_data_gpu_nondeterministic_POLARIS/tgv_poly_1/float32"
        path_64 = "./outputs/postproc/real_gnn_test_4/periodic_after_fix_edges_2/gradient_data_gpu_nondeterministic_POLARIS/tgv_poly_1/float32"

        SIZE_LIST = [1,2,4,8]
        #SIZE_LIST = [1,2,4,8,16,32] 
        #SIZE_LIST = [4,8,16,32] 
        #SIZE_LIST = [8,16,32] 
        HALO_MODE_LIST = ['none', 'all_to_all']
        #HALO_MODE_LIST = ['all_to_all']
        #HALO_MODE_LIST = ['sendrecv']


        data_32 = {}
        data_64 = {}

        for halo_mode in HALO_MODE_LIST: 
            data_32[halo_mode] = []  
            data_64[halo_mode] = []  
            for SIZE in SIZE_LIST:
                data_temp_32 = np.zeros((SIZE,6))
                data_temp_64 = np.zeros((SIZE,6))
                for RANK in range(SIZE): 
                    
                    # # Toy gnn 
                    # str_temp = "TOY_RANK_%d_SIZE_%d_halo_%s.tar" %(RANK, SIZE, halo_mode) 

                    # Real gnn input channels 1 output channels 1  
                    # str_temp = "RANK_%d_SIZE_%d_input_channels_1_hidden_channels_1_output_channels_1_nMessagePassingLayers_5_halo_%s.tar" %(RANK, SIZE, halo_mode) 
                    
                    # Real gnn input channels 3 output channels 3
                    # str_temp = "RANK_%d_SIZE_%d_input_channels_3_hidden_channels_32_output_channels_3_nMessagePassingLayers_5_halo_%s.tar" %(RANK, SIZE, halo_mode) 

                    # New gnn format: 
                    mp = 6
                    str_temp = f"POLY_1_RANK_{RANK}_SIZE_{SIZE}_3_7_32_3_2_{mp}_{halo_mode}.tar" 

                    a = torch.load(path_32 + "/" + str_temp, map_location=torch.device('cpu')) 
                    #data_temp_32[RANK, :3] = a['total_sum_x_scaled']
                    data_temp_32[RANK, :3] = a['total_sum_y_scaled']
                    #data_temp_32[RANK, :3] = a['total_sum_pos_scaled']
                    data_temp_32[RANK, 3] = a['effective_nodes']
                    data_temp_32[RANK, 4] = a['loss']
                    data_temp_32[RANK, 5] = a['effective_edges']

                    a = torch.load(path_64 + "/" + str_temp, map_location=torch.device('cpu')) 
                    #data_temp_64[RANK, :3] = a['total_sum_x_scaled']
                    data_temp_64[RANK, :3] = a['total_sum_y_scaled']
                    #data_temp_64[RANK, :3] = a['total_sum_pos_scaled']
                    data_temp_64[RANK, 3] = a['effective_nodes']
                    data_temp_64[RANK, 4] = a['loss']
                    data_temp_64[RANK, 5] = a['effective_edges']

                data_32[halo_mode].append(data_temp_32)
                data_64[halo_mode].append(data_temp_64)

        # Plot components 
        ms=200
        colors={'none': 'red', 'all_to_all': 'blue'}
        ls={'none': '-', 'all_to_all': '-.'}
        fig, ax = plt.subplots(1,3,figsize=(16,5))
        for comp in range(3):
            for i in range(len(SIZE_LIST)): 
                for halo_mode in HALO_MODE_LIST: 
                    SIZE = SIZE_LIST[i]
                    ax[comp].scatter(np.ones(SIZE)*SIZE, data_32[halo_mode][i][:,comp], marker='^', 
                               color=colors[halo_mode], s=ms, facecolors='none',
                               linestyle=ls[halo_mode], linewidth=1.5,
                               label="CPU, FP32" if i == 0 else None)

                    ax[comp].scatter(np.ones(SIZE)*SIZE, data_64[halo_mode][i][:,comp], marker='s', 
                               color=colors[halo_mode], s=ms, facecolors='none', 
                               linestyle=ls[halo_mode], linewidth=1.5, 
                               label="CPU, FP64" if i == 0 else None)

                    ax[comp].set_title('Component %d' %(comp))
                    ax[comp].set_xlabel('Number of Ranks')

                    #ax[comp].set_ylim([0.0766, 0.0770])
                    ax[comp].set_xlim([0.9, 40])
                    ax[comp].set_xscale('log')
        #ax.set_xscale('log')
        #ax[0].legend(fancybox=False, framealpha=1, edgecolor='black', prop={'size': 14})
        plt.show(block=False)

        # # Plot loss
        # ms=200
        # colors={'none': 'black', 'all_to_all': 'blue'}
        # fig, ax = plt.subplots(figsize=(6,5))
        # for i in range(len(SIZE_LIST)): 
        #     for halo_mode in HALO_MODE_LIST: 
        #         SIZE = SIZE_LIST[i]

        #         ax.scatter(np.ones(SIZE)*SIZE, data_32[halo_mode][i][:,4], marker='^', 
        #                    color=colors[halo_mode], s=ms, facecolors='none', 
        #                    label="CPU, FP32" if i == 0 else None)

        #         ax.scatter(np.ones(SIZE)*SIZE, data_64[halo_mode][i][:,4], marker='s', 
        #                    color=colors[halo_mode], s=ms, facecolors='none', 
        #                    label="CPU, FP64" if i == 0 else None)

        #         ax.set_title('Loss')
        #         ax.set_xlabel('Number of Ranks')

        # #ax.set_xscale('log')
        # #ax.legend(fancybox=False, framealpha=1, edgecolor='black', prop={'size': 14})
        # ax.set_ylim([0.0763, 0.0768])
        # plt.show(block=False)


    if 1 == 0:
        """
        Looking at graph stats -- number of nodes, edges, etc. 
        """

        POLY_LIST = [1, 3, 5, 7]  
        SIZE_LIST = [1, 2, 4, 8, 16, 32, 64, 128]

        n_nodes_local = [] 
        n_nodes_halo = []
        n_edges = [] 
        for i in range(len(POLY_LIST)):
            POLY = POLY_LIST[i]
            n_nodes_local.append([])
            n_nodes_halo.append([])
            n_edges.append([])
            for j in range(len(SIZE_LIST)):
                SIZE = SIZE_LIST[j]
                n_nodes_local[i].append(np.zeros(SIZE))
                n_nodes_halo[i].append(np.zeros(SIZE))
                n_edges[i].append(np.zeros(SIZE))
                for RANK in range(SIZE):

                    try: 
                        str_temp = f"POLY_{POLY}_RANK_{RANK}_SIZE_{SIZE}_input_channels_3_hidden_channels_32_output_channels_3_nMessagePassingLayers_5_halo_all_to_all.tar"
                        a = torch.load("./outputs/GraphStatistics/" + str_temp)
                    except FileNotFoundError:
                        str_temp = f"POLY_{POLY}_RANK_{RANK}_SIZE_{SIZE}_input_channels_3_hidden_channels_32_output_channels_3_nMessagePassingLayers_2_halo_none.tar"
                        a = torch.load("./outputs/GraphStatistics/" + str_temp)

                    
                    n_nodes_local[i][j][RANK] = a['n_nodes_local'].item()
                    n_nodes_halo[i][j][RANK] = a['n_nodes_halo'].item()
                    n_edges[i][j][RANK] = a['n_edges']



        # Local nodes per rank 
        ms = 100
        fig, ax = plt.subplots(figsize=(8,7))
        for j in range(len(SIZE_LIST)):
            SIZE = SIZE_LIST[j]
            ax.scatter(SIZE*np.ones(SIZE), n_nodes_local[0][j], s=ms, color='black') 
            ax.scatter(SIZE*np.ones(SIZE), n_nodes_local[1][j], s=ms, color='blue') 
            ax.scatter(SIZE*np.ones(SIZE), n_nodes_local[2][j], s=ms, color='red') 
            ax.scatter(SIZE*np.ones(SIZE), n_nodes_local[3][j], s=ms, color='green') 
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.set_ylabel('Local Graph Nodes')
        ax.set_xlabel('Number of GPUs')
        plt.show(block=False)

        # Halo nodes per rank 
        ms = 100
        fig, ax = plt.subplots(figsize=(8,7))
        for j in range(len(SIZE_LIST)):
            SIZE = SIZE_LIST[j]
            ax.scatter(SIZE*np.ones(SIZE), n_nodes_halo[0][j], s=ms, color='black') 
            ax.scatter(SIZE*np.ones(SIZE), n_nodes_halo[1][j], s=ms, color='blue') 
            ax.scatter(SIZE*np.ones(SIZE), n_nodes_halo[2][j], s=ms, color='red') 
            ax.scatter(SIZE*np.ones(SIZE), n_nodes_halo[3][j], s=ms, color='green') 
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.set_ylabel('Halo Graph Nodes')
        ax.set_xlabel('Number of GPUs')
        plt.show(block=False)


        # Halo nodes / local nodes 
        ms = 100
        fig, ax = plt.subplots(figsize=(8,7))
        for j in range(len(SIZE_LIST)):
            SIZE = SIZE_LIST[j]
            ax.scatter(SIZE*np.ones(SIZE), n_nodes_halo[0][j]/n_nodes_local[0][j], s=ms, color='black') 
            ax.scatter(SIZE*np.ones(SIZE), n_nodes_halo[1][j]/n_nodes_local[1][j], s=ms, color='blue') 
            ax.scatter(SIZE*np.ones(SIZE), n_nodes_halo[2][j]/n_nodes_local[2][j], s=ms, color='red') 
            ax.scatter(SIZE*np.ones(SIZE), n_nodes_halo[3][j]/n_nodes_local[3][j], s=ms, color='green') 
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.set_ylabel('Halo Nodes / Local Nodes')
        ax.set_xlabel('Number of GPUs')
        plt.show(block=False)


        # # Total halo nodes (summed over all ranks)
        # ms = 100
        # fig, ax = plt.subplots(figsize=(8,7))
        # for j in range(len(SIZE_LIST)):
        #     SIZE = SIZE_LIST[j]
        #     ax.scatter(SIZE, np.sum(n_nodes_halo[0][j]), s=ms, color='black') 
        #     ax.scatter(SIZE, np.sum(n_nodes_halo[1][j]), s=ms, color='blue') 
        #     ax.scatter(SIZE, np.sum(n_nodes_halo[2][j]), s=ms, color='red') 
        #     ax.scatter(SIZE, np.sum(n_nodes_halo[3][j]), s=ms, color='green') 
        # ax.set_yscale('log')
        # ax.set_xscale('log')
        # ax.set_ylabel('Total Halo Graph Nodes')
        # ax.set_xlabel('Ranks')
        # plt.show(block=False)


        # Edges per rank 
        ms = 100
        fig, ax = plt.subplots(figsize=(8,7))
        for j in range(len(SIZE_LIST)):
            SIZE = SIZE_LIST[j]
            ax.scatter(SIZE*np.ones(SIZE), n_edges[0][j], s=ms, color='black') 
            ax.scatter(SIZE*np.ones(SIZE), n_edges[1][j], s=ms, color='blue') 
            ax.scatter(SIZE*np.ones(SIZE), n_edges[2][j], s=ms, color='red') 
            ax.scatter(SIZE*np.ones(SIZE), n_edges[3][j], s=ms, color='green') 
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.set_ylabel('Graph Edges')
        ax.set_xlabel('Number of GPUs')
        plt.show(block=False)

    if 1 == 1:
        """
        Looking at profiler outputs 
        """
        if 1 == 0: # data generation 
            POLY_LIST = [1,3,5] # nekrs polynomial order  
            N_MP_LIST = [2,4,6,8] # number of message passing layers 
            N_HC_LIST = [32] # number of hidden channels 

            # constants 
            seed = 12
            input_channels_node = 3
            input_channels_edge = 7
            output_channels = 3
            hidden_layers = 2 

            SIZE_LIST = [1,2,4,8,16,32,64,128]
            HALO_MODE_LIST = ['none', 'all_to_all']
            
            for poly in POLY_LIST:
                for n_mp in N_MP_LIST:
                    for n_hc in N_HC_LIST:
                        t_forwardPass = {}
                        for i in range(len(HALO_MODE_LIST)):
                            halo = HALO_MODE_LIST[i]
                            t_forwardPass[halo] = []
                            for j in range(len(SIZE_LIST)):
                                size = SIZE_LIST[j]
                                t_forwardPass[halo].append(np.zeros(size))
                                for k in range(size):
                                    rank = k 
                                    # file_str = f"POLY_{poly}_RANK_{rank}_SIZE_{size}_input_channels_3_hidden_channels_{n_hc}_output_channels_3_nMessagePassingLayers_{n_mp}_halo_{halo}.tar"
                                    file_str = f"POLY_{poly}_RANK_{rank}_SIZE_{size}_SEED_{seed}_{input_channels_node}_{input_channels_edge}_{n_hc}_{output_channels}_{hidden_layers}_{n_mp}_{halo}.tar"

                                    # load profile data 
                                    try: 
                                        temp_prof = torch.load("./outputs/profiles/" + file_str)
                                        #temp_prof = torch.load("./outputs/profiles_old/" + file_str)
                                        # print(temp_prof.table(sort_by="cpu_time_total", row_limit=10))
                                        key_list = [] 
                                        for key_id in range(len(temp_prof)):
                                            key_list.append(temp_prof[key_id].key)
                                        idx_key = key_list.index(f'[RANK {rank}] FORWARD PASS') 
                                        cuda_time = temp_prof[idx_key].cuda_time # in microseconds, averaged over many runs  

                                        t_forwardPass[halo][j][k] = cuda_time

                                    except FileNotFoundError:
                                        print(f"FileNotFound: {file_str}")
                                        cuda_time = 0
                                        t_forwardPass[halo][j][k] = cuda_time

                                    print(f"[POLY {poly}, N_MP {n_mp}, N_HC {n_hc}, SIZE {size}, RANK {rank}] -- cuda_time = {cuda_time} us")

                        
                        # Write the data 
                        for halo in HALO_MODE_LIST:
                            x_axis = np.array(SIZE_LIST)
                            y_axis_mean = np.zeros_like(x_axis)
                            y_axis_max = np.zeros_like(x_axis)
                            y_axis_min = np.zeros_like(x_axis)
                            for j in range(len(SIZE_LIST)):
                                y_axis_mean[j] = t_forwardPass[halo][j].mean()
                                y_axis_max[j] = t_forwardPass[halo][j].max()
                                y_axis_min[j] = t_forwardPass[halo][j].min()
                            
                            temp_name = f"POLY_{poly}_SEED_{seed}_{input_channels_node}_{input_channels_edge}_{n_hc}_{output_channels}_{hidden_layers}_{n_mp}_{halo}"
                            np.save(f"outputs/{temp_name}_mean.npy", y_axis_mean)
                            np.save(f"outputs/{temp_name}_max.npy", y_axis_min)
                            np.save(f"outputs/{temp_name}_min.npy", y_axis_max)

                        # # plot data 
                        # fig, ax = plt.subplots(figsize=(6,6))
                        # lw = 2 
                        # 
                        # halo = 'none'
                        # x_axis = np.array(SIZE_LIST)
                        # y_axis_mean = np.zeros_like(x_axis)
                        # y_axis_max = np.zeros_like(x_axis)
                        # y_axis_min = np.zeros_like(x_axis)
                        # for j in range(len(SIZE_LIST)):
                        #     y_axis_mean[j] = t_forwardPass[halo][j].mean()
                        #     y_axis_max[j] = t_forwardPass[halo][j].max()
                        #     y_axis_min[j] = t_forwardPass[halo][j].min()
                        # np.save(f"outputs/p_{poly}_{halo}_mean.npy", y_axis_mean)
                        # np.save(f"outputs/p_{poly}_{halo}_max.npy", y_axis_min)
                        # np.save(f"outputs/p_{poly}_{halo}_min.npy", y_axis_max)

                        # ax.plot(x_axis, y_axis_mean, color='black', marker='o', lw=lw, label='no halo')
                        # ax.fill_between(x_axis, y_axis_min, y_axis_max, color='black', lw=1, alpha=0.2)

                        # halo = 'all_to_all'
                        # x_axis = np.array(SIZE_LIST)
                        # y_axis_mean = np.zeros_like(x_axis)
                        # y_axis_max = np.zeros_like(x_axis)
                        # y_axis_min = np.zeros_like(x_axis)
                        # for j in range(len(SIZE_LIST)):
                        #     y_axis_mean[j] = t_forwardPass[halo][j].mean()
                        #     y_axis_max[j] = t_forwardPass[halo][j].max()
                        #     y_axis_min[j] = t_forwardPass[halo][j].min()
                        # 
                        # #np.save(f"outputs/p_{poly}_{halo}_mean.npy", y_axis_mean)
                        # #np.save(f"outputs/p_{poly}_{halo}_max.npy", y_axis_min)
                        # #np.save(f"outputs/p_{poly}_{halo}_min.npy", y_axis_max)

                        # ax.plot(x_axis, y_axis_mean, color='blue', marker='o', lw=lw, label='all_to_all')
                        # ax.fill_between(x_axis, y_axis_min, y_axis_max, color='blue', lw=1, alpha=0.2)

                        # ax.set_ylabel('Time [us]')
                        # ax.set_xlabel('Number of GPUs')
                        # ax.set_yscale('log')
                        # ax.set_xscale('log')
                        # ax.set_title('DistGNN Forward Pass (P = %d)' %(poly))
                        # ax.legend()
                        # plt.show(block=False)

        if 1 == 1:
            HALO_MODE_LIST = ['none', 'all_to_all']
            seed = 12
            input_channels_node = 3
            input_channels_edge = 7 
            output_channels = 3 
            hidden_layers = 2

            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # ~~~~ the effect of n_mp layers, for fixed poly and n_hc 
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            poly = 5
            n_hc = 32
            N_MP_LIST = [2,4,6,8]

            norm = Normalize(vmin=np.min(N_MP_LIST), vmax=np.max(N_MP_LIST))

            data_all_mean = [] 
            data_all_max = [] 
            data_all_min = []

            data_none_mean = []
            data_none_max = []
            data_none_min = []

            for n_mp in N_MP_LIST:
                halo = 'all_to_all'
                temp_name = f"POLY_{poly}_SEED_{seed}_{input_channels_node}_{input_channels_edge}_{n_hc}_{output_channels}_{hidden_layers}_{n_mp}_{halo}"
                data_all_mean.append(np.load(f"outputs/{temp_name}_mean.npy"))
                data_all_max.append(np.load(f"outputs/{temp_name}_max.npy"))
                data_all_min.append(np.load(f"outputs/{temp_name}_min.npy"))

                halo = 'none'
                temp_name = f"POLY_{poly}_SEED_{seed}_{input_channels_node}_{input_channels_edge}_{n_hc}_{output_channels}_{hidden_layers}_{n_mp}_{halo}"
                data_none_mean.append(np.load(f"outputs/{temp_name}_mean.npy"))
                data_none_max.append(np.load(f"outputs/{temp_name}_max.npy"))
                data_none_min.append(np.load(f"outputs/{temp_name}_min.npy"))

            x_axis = np.array([1,2,4,8,16,32,64,128])

            lw = 1.5 
            ms = 10

            fig, ax = plt.subplots(figsize=(5,6))
            for i in range(len(N_MP_LIST)):
                color = cm.viridis(norm(N_MP_LIST[i]))
                ax.plot(x_axis, data_none_mean[i], color=color, lw=lw, marker='o', ms=ms, mec='black')
                ax.plot(x_axis, data_all_mean[i], color=color, lw=lw, marker='s', ms=ms, mec='black', ls='--')

            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlabel('nGPU')
            ax.set_ylabel('Time [us]')
            ax.set_title(f"poly={poly}, hc={n_hc}")
            ax.set_ylim([1e2, 1e6])
            #ax.legend(fancybox=False, framealpha=1)
            # # left, bottom, width, height
            # cax = fig.add_axes([0.11, 0.21, 0.35, 0.03])  # Position and size of the color bar
            # sm = cm.ScalarMappable(cmap=cm.viridis, norm=norm)
            # sm.set_array([])
            # fig.colorbar(sm, cax=cax, orientation='horizontal')
            # ax.grid(False)
            plt.show(block=False)



            



            # # ~~~~ OLD PLOTTING 
            # # Plot all curves 
            # data_all_mean = [] 
            # data_all_max = [] 
            # data_all_min = []

            # data_none_mean = []
            # data_none_max = []
            # data_none_min = []

            # for poly in [1,3,5]:
            #     halo = 'all_to_all'
            #     data_all_mean.append(np.load(f"outputs/p_{poly}_{halo}_mean.npy"))
            #     data_all_max.append(np.load(f"outputs/p_{poly}_{halo}_max.npy"))
            #     data_all_min.append(np.load(f"outputs/p_{poly}_{halo}_min.npy"))

            #     halo = 'none'
            #     data_none_mean.append(np.load(f"outputs/p_{poly}_{halo}_mean.npy"))
            #     data_none_max.append(np.load(f"outputs/p_{poly}_{halo}_max.npy"))
            #     data_none_min.append(np.load(f"outputs/p_{poly}_{halo}_min.npy"))

            # x_axis = np.array([1,2,4,8,16,32,64,128])


            # lw = 2 
            # fig, ax = plt.subplots(figsize=(6,6))

            # ax.plot(x_axis, data_none_mean[0], color='black', lw=lw, marker='o', label='p=1, no halo')
            # ax.plot(x_axis, data_all_mean[0], color='black', ls='--', lw=lw, marker='s', label='p=1, all_to_all')

            # ax.plot(x_axis, data_none_mean[1], color='blue', lw=lw, marker='o', label='p=3, no halo')
            # ax.plot(x_axis, data_all_mean[1], color='blue', ls='--', lw=lw, marker='s', label='p=3, all_to_all')

            # ax.plot(x_axis[3:], data_none_mean[2][3:], color='red', lw=lw, marker='o', label='p=5, no halo')
            # ax.plot(x_axis[3:], data_all_mean[2][3:], color='red', ls='--', lw=lw, marker='s', label='p=5, all_to_all')

            # ax.set_xscale('log')
            # ax.set_yscale('log')
            # ax.set_xlabel('nGPU')
            # ax.set_ylabel('Time [us]')
            # #ax.legend(fancybox=False, framealpha=1)
            # plt.show(block=False)









        





