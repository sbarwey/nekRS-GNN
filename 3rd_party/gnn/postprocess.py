import numpy as np
import matplotlib.pyplot as plt
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
        # Load model and plot loss 
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
        
        #data_path = './outputs/postproc/gradient_data/tgv_poly_1'
        #data_path = './outputs/postproc/gradient_data_gpu_nondeterministic_repeat/tgv_poly_1/float64'
        data_path = './outputs/postproc/gradient_data_gpu_nondeterministic_repeat/tgv_2d_18_poly_1/float64'
        #data_path = './outputs/postproc/gradient_data_cpu_nondeterministic/tgv_poly_1'
        #data_path = './outputs/postproc/gradient_data_cpu_deterministic/tgv_poly_1'
        #data_path = './outputs/postproc/gradient_data/hemi_poly_9'
        #data_path = './outputs/postproc/gradient_data/tgv_poly_7'

        #halo_mode_list = ['all_to_all', 'sendrecv', 'none']
        halo_mode_list = ['all_to_all', 'none']
        halo_color = ['black', 'blue', 'red']

        # get dict keys 
        SIZE = 16
        for RANK in range(SIZE):
            for halo_mode in halo_mode_list:
                model_name = 'RANK_%d_SIZE_%d_input_channels_3_hidden_channels_32_output_channels_3_nMessagePassingLayers_5_halo_%s.tar' %(RANK,SIZE,halo_mode)
                a = torch.load(data_path + '/' + model_name, map_location=torch.device('cpu'))
                keys = list(a.keys())
                keys.remove('loss')
                break

        # Load the baseline -- SIZE 1 
        SIZE_1 = get_grad_data(SIZE=1, keys=keys, halo_mode_list=halo_mode_list) 
        SIZE_2 = get_grad_data(SIZE=2, keys=keys, halo_mode_list=halo_mode_list)
        SIZE_4 = get_grad_data(SIZE=4, keys=keys, halo_mode_list=halo_mode_list)
        SIZE_8 = get_grad_data(SIZE=8, keys=keys, halo_mode_list=halo_mode_list)
        SIZE_16 = get_grad_data(SIZE=16, keys=keys, halo_mode_list=halo_mode_list)
        #SIZE_32 = get_grad_data(SIZE=32, keys=keys)

        # # Plot: loss versus n_gpus  
        # SIZE_LIST = [1,2,4,8,16,32]
        # DATA_LIST = [SIZE_1, SIZE_2, SIZE_4, SIZE_8, SIZE_16, SIZE_32]

        SIZE_LIST = [1,2,4,8,16]
        DATA_LIST = [SIZE_1, SIZE_2, SIZE_4, SIZE_8, SIZE_16]

        #SIZE_LIST = [2,4,8, 16, 32]
        #DATA_LIST = [SIZE_2, SIZE_4, SIZE_8, SIZE_16, SIZE_32]

        #SIZE_LIST = [16, 32]
        #DATA_LIST = [SIZE_16, SIZE_32]

        # Loss plots 
        fig, ax = plt.subplots(figsize=(7,6))
        for i in range(len(SIZE_LIST)): 
            SIZE = SIZE_LIST[i]
            DATA = DATA_LIST[i]
            for RANK in range(SIZE):
                for k in range(len(halo_mode_list)):
                    halo_mode = halo_mode_list[k]
                    color = halo_color[k]
                    ax.plot(SIZE, DATA[RANK][halo_mode]['loss'], marker='o', ms=10, color=color) 

        ax.set_xlim([0.8,64])
        ax.set_xlabel('N GPUs')
        ax.set_ylabel('Loss')
        ax.set_xscale('log')
        plt.show(block=False)

        # # Weight gradient norm 
        # fig, ax = plt.subplots()
        # for i in range(len(SIZE_LIST)): 
        #     SIZE = SIZE_LIST[i]
        #     DATA = DATA_LIST[i]
        #     for RANK in range(SIZE):
        #         for k in range(len(halo_mode_list)):
        #             halo_mode = halo_mode_list[k]
        #             color = halo_color[k]
        #             ax.plot(SIZE, DATA[RANK][halo_mode]['grad'].norm(), marker='o', ms=10, color=color) 

        # ax.set_xlim([0.8,64])
        # ax.set_xscale('log')
        # ax.set_xlabel('N GPUs')
        # ax.set_ylabel('Weight gradient norm')
        # plt.show(block=False)


        # # Weight gradient PDF  
        # bins = 500
        # lw = 2
        # fig, ax = plt.subplots()
        # #a = np.histogram(np.log(np.abs(SIZE_4[3]['all_to_all']['grad'])), bins=bins)
        # a = np.histogram(np.log(np.abs(SIZE_16[3]['all_to_all']['grad'])), bins=bins)
        # ax.plot(a[1][:-1], a[0], color='black', label='Baseline', lw=lw)

        # #a = np.histogram(np.log(np.abs(SIZE_16[0]['all_to_all']['grad'])), bins=bins)
        # a = np.histogram(np.log(np.abs(SIZE_32[0]['all_to_all']['grad'])), bins=bins)
        # ax.plot(a[1][:-1], a[0], label='with halo', lw=lw)

        # #a = np.histogram(np.log(np.abs(SIZE_16[0]['sendrecv']['grad'])), bins=bins)
        # #ax.plot(a[1][:-1], a[0], label='send_recv', lw=lw)

        # #a = np.histogram(np.log(np.abs(SIZE_16[0]['none']['grad'])), bins=bins)
        # a = np.histogram(np.log(np.abs(SIZE_32[0]['none']['grad'])), bins=bins)
        # ax.plot(a[1][:-1], a[0], label='no halo', lw=lw)

        # ax.set_xlabel('log(abs(Weight gradient))')
        # ax.legend()
        # plt.show(block=False)


    # Model weight comparisons -- are the models the same? 
    if 1 == 0:
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


    # ~~~~ LOOKING AT LOSS METRIC 
    if 1 == 0: 
    
        path_gpu_32 = "./outputs/postproc/gradient_data_gpu_nondeterministic_repeat/tgv_poly_1/float32"
        path_cpu1_32 = "./outputs/postproc/gradient_data_cpu_nondeterministic/tgv_poly_1/float32"
        #path_cpu2_32 = "./outputs/postproc/gradient_data_cpu_deterministic/tgv_poly_1/float32"
        path_cpu2_32 = "./outputs/postproc/gradient_data_cpu_nondeterministic_LOCAL/tgv_poly_1/float32"
        path_gpu_64 = "./outputs/postproc/gradient_data_gpu_nondeterministic_repeat/tgv_poly_1/float64"
        path_cpu1_64 = "./outputs/postproc/gradient_data_cpu_nondeterministic/tgv_poly_1/float64"
        #path_cpu2_64 = "./outputs/postproc/gradient_data_cpu_deterministic/tgv_poly_1/float64"
        path_cpu2_64 = "./outputs/postproc/gradient_data_cpu_nondeterministic_LOCAL/tgv_poly_1/float64"


        #SIZE_LIST = [1,2,4,8,16,32] 
        SIZE_LIST = [1,2,4,8]
        HALO_MODE_LIST = ['none', 'all_to_all']

        data_gpu_32 = {} 
        data_cpu1_32 = {}
        data_cpu2_32 = {}
        data_gpu_64 = {}
        data_cpu1_64 = {}
        data_cpu2_64 = {} 

        for halo_mode in HALO_MODE_LIST: 
            data_gpu_32[halo_mode] = []  
            data_cpu1_32[halo_mode] = []  
            data_cpu2_32[halo_mode] = []  

            data_gpu_64[halo_mode] = []  
            data_cpu1_64[halo_mode] = []  
            data_cpu2_64[halo_mode] = []  

            for SIZE in SIZE_LIST:
                data_temp_gpu_32 = np.zeros(SIZE)
                data_temp_cpu1_32 = np.zeros(SIZE)
                data_temp_cpu2_32 = np.zeros(SIZE)
                data_temp_gpu_64 = np.zeros(SIZE)
                data_temp_cpu1_64 = np.zeros(SIZE)
                data_temp_cpu2_64 = np.zeros(SIZE)
                for RANK in range(SIZE): 
                    
                    str_temp = "RANK_%d_SIZE_%d_input_channels_3_hidden_channels_32_output_channels_3_nMessagePassingLayers_5_halo_%s.tar" %(RANK, SIZE, halo_mode) 

                    # gpu_32
                    a = torch.load(path_gpu_32 + "/" + str_temp, map_location=torch.device('cpu')) 
                    data_temp_gpu_32[RANK] = a['loss']

                    # cpu1_32
                    a = torch.load(path_cpu1_32 + "/" + str_temp, map_location=torch.device('cpu')) 
                    data_temp_cpu1_32[RANK] = a['loss']

                    # cpu2_32
                    a = torch.load(path_cpu2_32 + "/" + str_temp, map_location=torch.device('cpu')) 
                    data_temp_cpu2_32[RANK] = a['loss']

                    # gpu_64
                    a = torch.load(path_gpu_64 + "/" + str_temp, map_location=torch.device('cpu')) 
                    data_temp_gpu_64[RANK] = a['loss']

                    # cpu1_64
                    a = torch.load(path_cpu1_64 + "/" + str_temp, map_location=torch.device('cpu')) 
                    data_temp_cpu1_64[RANK] = a['loss']

                    # cpu2_64
                    a = torch.load(path_cpu2_64 + "/" + str_temp, map_location=torch.device('cpu')) 
                    data_temp_cpu2_64[RANK] = a['loss']


                data_gpu_32[halo_mode].append(data_temp_gpu_32)
                data_cpu1_32[halo_mode].append(data_temp_cpu1_32)
                data_cpu2_32[halo_mode].append(data_temp_cpu2_32)
                data_gpu_64[halo_mode].append(data_temp_gpu_64)
                data_cpu1_64[halo_mode].append(data_temp_cpu1_64)
                data_cpu2_64[halo_mode].append(data_temp_cpu2_64)


        # Plot
        ms=200
        colors={'none': 'black', 'all_to_all': 'blue'}
        fig, ax = plt.subplots()
        for i in range(len(SIZE_LIST)): 
            for halo_mode in HALO_MODE_LIST: 
                SIZE = SIZE_LIST[i]

                ax.scatter(np.ones(SIZE)*SIZE, data_gpu_32[halo_mode][i], marker='o', color=colors[halo_mode], s=ms, facecolors='none', 
                           label="GPU, FP32" if i == 0 else None)

                ax.scatter(np.ones(SIZE)*SIZE, data_cpu1_32[halo_mode][i], marker='s', color=colors[halo_mode], s=ms, facecolors='none', 
                           label="CPU, FP32" if i == 0 else None)

                ax.scatter(np.ones(SIZE)*SIZE, data_cpu2_32[halo_mode][i], marker='^', color=colors[halo_mode], s=ms, facecolors='none', 
                           label="CPU, FP32, Det." if i == 0 else None)

                # ax.scatter(np.ones(SIZE)*SIZE, data_gpu_64[halo_mode][i], marker='o', color=colors[halo_mode], s=ms, facecolors='none', 
                #            label="GPU, FP64" if i == 0 else None)

                # ax.scatter(np.ones(SIZE)*SIZE, data_cpu1_64[halo_mode][i], marker='s', color=colors[halo_mode], s=ms, facecolors='none', 
                #            label="CPU, FP64" if i == 0 else None)

                # ax.scatter(np.ones(SIZE)*SIZE, data_cpu2_64[halo_mode][i], marker='^', color=colors[halo_mode], s=ms, facecolors='none', 
                #            label="CPU, FP64, Det." if i == 0 else None)

        ax.set_xscale('log')
        ax.legend(fancybox=False, framealpha=1, edgecolor='black', prop={'size': 14})
        plt.show(block=False)

    # ~~~~ LOOKING AT NODE SUMMATION OF (A) INPUT TO GNN, and (B) OUTPUT OF GNN 
    if 1 == 1: 

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
        path_32 = "./outputs/postproc/real_gnn_test_3/periodic_after_fix_edges_2/gradient_data_cpu_nondeterministic_LOCAL/tgv_poly_1/float32"
        path_64 = "./outputs/postproc/real_gnn_test_3/periodic_after_fix_edges_2/gradient_data_cpu_nondeterministic_LOCAL/tgv_poly_1/float64"

        SIZE_LIST = [1,2,4,8] 
        #SIZE_LIST = [1,2,4] 
        #SIZE_LIST = [1,2,4,8,16,32] 
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
                    str_temp = "RANK_%d_SIZE_%d_input_channels_3_hidden_channels_32_output_channels_3_nMessagePassingLayers_5_halo_%s.tar" %(RANK, SIZE, halo_mode) 

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
