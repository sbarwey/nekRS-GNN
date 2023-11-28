import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 22})

import torch


def get_grad_data(SIZE, keys):
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

    if 1 == 1: 
        
        data_path = './outputs/postproc/gradient_data/tgv_poly_1'

        halo_mode_list = ['all_to_all', 'sendrecv', 'none']
        halo_color = ['black', 'blue', 'red']

        # get dict keys 
        SIZE = 1
        for RANK in range(SIZE):
            for halo_mode in halo_mode_list:
                model_name = 'RANK_%d_SIZE_%d_input_channels_3_hidden_channels_32_output_channels_3_nMessagePassingLayers_5_halo_%s.tar' %(RANK,SIZE,halo_mode)
                a = torch.load(data_path + '/' + model_name, map_location=torch.device('cpu'))
                keys = list(a.keys())
                keys.remove('loss')
                break


        # Load the baseline -- SIZE 1 
        SIZE_1 = get_grad_data(SIZE=1, keys=keys) 
        SIZE_2 = get_grad_data(SIZE=2, keys=keys)
        SIZE_4 = get_grad_data(SIZE=4, keys=keys)
        SIZE_8 = get_grad_data(SIZE=8, keys=keys)
        SIZE_16 = get_grad_data(SIZE=16, keys=keys)
        SIZE_32 = get_grad_data(SIZE=32, keys=keys)

        # Plot: loss versus n_gpus  
        SIZE_LIST = [1,2,4,8, 16, 32]
        DATA_LIST = [SIZE_1, SIZE_2, SIZE_4, SIZE_8, SIZE_16, SIZE_32]

        # Loss plots 
        fig, ax = plt.subplots()
        for i in range(len(SIZE_LIST)): 
            SIZE = SIZE_LIST[i]
            DATA = DATA_LIST[i]
            for RANK in range(SIZE):
                for k in range(len(halo_mode_list)):
                    halo_mode = halo_mode_list[k]
                    color = halo_color[k]
                    ax.plot(SIZE, DATA[RANK][halo_mode]['loss'], marker='o', ms=10, color=color) 

        ax.set_xlabel('N GPUs')
        ax.set_ylabel('Loss')
        ax.set_xscale('log')
        plt.show(block=False)

        # Weight gradient norm 
        fig, ax = plt.subplots()
        for i in range(len(SIZE_LIST)): 
            SIZE = SIZE_LIST[i]
            DATA = DATA_LIST[i]
            for RANK in range(SIZE):
                for k in range(len(halo_mode_list)):
                    halo_mode = halo_mode_list[k]
                    color = halo_color[k]
                    ax.plot(SIZE, DATA[RANK][halo_mode]['grad'].norm(), marker='o', ms=10, color=color) 

        ax.set_xlabel('N GPUs')
        ax.set_ylabel('Weight gradient norm')
        plt.show(block=False)


        # Weight gradient PDF  
        bins = 500
        lw = 2
        fig, ax = plt.subplots()
        a = np.histogram(np.log(np.abs(SIZE_4[3]['all_to_all']['grad'])), bins=bins)
        ax.plot(a[1][:-1], a[0], color='black', label='Baseline', lw=lw)

        a = np.histogram(np.log(np.abs(SIZE_16[0]['all_to_all']['grad'])), bins=bins)
        ax.plot(a[1][:-1], a[0], label='all_to_all', lw=lw)

        a = np.histogram(np.log(np.abs(SIZE_16[0]['sendrecv']['grad'])), bins=bins)
        ax.plot(a[1][:-1], a[0], label='send_recv', lw=lw)

        a = np.histogram(np.log(np.abs(SIZE_16[0]['none']['grad'])), bins=bins)
        ax.plot(a[1][:-1], a[0], label='no halo', lw=lw)

        ax.set_xlabel('log(abs(Weight gradient))')
        ax.legend()
        plt.show(block=False)






