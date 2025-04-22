import numpy as np
import re
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LogNorm
import matplotlib.cm as cm
import os,time 

plt.rcParams.update({'font.size': 22})

import torch


def plot_from_log(log_file_path_list, N_skip=224):
    steps_list = []
    loss_list = []
    lr_list = []
    t_forwardPass_list = []
    for log_file_path in log_file_path_list: 
        step_line_regex = re.compile(
        r'\[STEP\s+(\d+)\]\s+loss=([\deE\+\.-]+)\s+r_loss=([\deE\+\.-]+)\s+t_step=([\deE\+\.-]+)\s+sec\s+lr=([\deE\+\.-]+)')

        # For lines like: t_dataTransfer: 0.0001938 sec
        basic_time_regex = re.compile(
            r'^\[.*?\].*-\s+(t_\w+):\s+([\deE\+\.-]+)\s+sec'
        ) 

        # For lines that have nodes/sec info, e.g.: t_forwardPass: 0.1266 sec [2.8707e+05 nodes/sec]
        nodes_time_regex = re.compile(
            r'^\[.*?\].*-\s+(t_\w+):\s+([\deE\+\.-]+)\s+sec\s+\[([\deE\+\.-]+)\s+nodes/sec\]'
        )

        # For grad norm line: grad norm: 4.13608
        grad_norm_regex = re.compile(
            r'^\[.*?\].*-\s+grad norm:\s+([\deE\+\.-]+)'
        )

        steps = []
        losses = []
        r_losses = []
        t_steps = []
        lrs = []
        t_dataTransfer = []
        t_bufferInit = []
        t_forwardPass = []
        t_forwardPass_nodes = []
        t_loss_list = []
        t_loss_nodes = []
        t_backwardPass = []
        t_backwardPass_nodes = []
        t_optimizerStep = []
        grad_norms = []

        with open(log_file_path, 'r') as f:
            # Skip first N lines
            for _ in range(N_skip):
                next(f)

            current_step = None
            # We will read line by line
            for line in f:
                # Check if line is a step line
                step_match = step_line_regex.search(line)
                if step_match:
                    # We've encountered a new step line, which means previous step's data collection is done.
                    # Parse values
                    current_step = int(step_match.group(1))
                    steps.append(current_step)
                    losses.append(float(step_match.group(2)))
                    r_losses.append(float(step_match.group(3)))
                    t_steps.append(float(step_match.group(4)))
                    lrs.append(float(step_match.group(5)))
                    continue

                # If we have a current step set, we parse subsequent lines
                if current_step is not None:
                    # Check for times
                    nm = nodes_time_regex.search(line)
                    if nm:
                        key = nm.group(1)
                        val = float(nm.group(2))
                        nodes_val = float(nm.group(3))
                        if key == 't_forwardPass':
                            t_forwardPass.append(val)
                            t_forwardPass_nodes.append(nodes_val)
                        elif key == 't_loss':
                            t_loss_list.append(val)
                            t_loss_nodes.append(nodes_val)
                        elif key == 't_backwardPass':
                            t_backwardPass.append(val)
                            t_backwardPass_nodes.append(nodes_val)
                        else:
                            # If you have other timed metrics with nodes/sec pattern, handle them here
                            pass
                        continue

                    bm = basic_time_regex.search(line)
                    if bm:
                        key = bm.group(1)
                        val = float(bm.group(2))
                        if key == 't_dataTransfer':
                            t_dataTransfer.append(val)
                        elif key == 't_bufferInit':
                            t_bufferInit.append(val)
                        elif key == 't_optimizerStep':
                            t_optimizerStep.append(val)
                        # Add other basic times as needed
                        continue

                    gm = grad_norm_regex.search(line)
                    if gm:
                        grad_norms.append(float(gm.group(1)))
                        continue

        # Convert lists to numpy arrays
        steps = np.array(steps)
        losses = np.array(losses)
        r_losses = np.array(r_losses)
        t_steps = np.array(t_steps)
        lrs = np.array(lrs)
        t_dataTransfer = np.array(t_dataTransfer)
        t_bufferInit = np.array(t_bufferInit)
        t_forwardPass = np.array(t_forwardPass)
        t_forwardPass_nodes = np.array(t_forwardPass_nodes)
        t_loss_list = np.array(t_loss_list)
        t_loss_nodes = np.array(t_loss_nodes)
        t_backwardPass = np.array(t_backwardPass)
        t_backwardPass_nodes = np.array(t_backwardPass_nodes)
        t_optimizerStep = np.array(t_optimizerStep)
        grad_norms = np.array(grad_norms)

        steps_list.append(steps)
        loss_list.append(losses)
        lr_list.append(lrs)
        t_forwardPass_list.append(t_forwardPass)

        steps = np.concatenate(steps_list) 
        loss = np.concatenate(loss_list)
        lr = np.concatenate(lr_list)
        t_fp = np.concatenate(t_forwardPass_list)
        data = {}
        data['steps'] = steps
        data['loss'] = loss
        data['lr'] = lr
        data['t_fp'] = t_fp

    return data


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

    if 1 == 1:
        """
        [INFERENCE] - Plot rollout errors versus step for each feature 
        """
        def get_ordered_files(dir_path, header):
            # List files that start with the header
            files = [f for f in os.listdir(dir_path) if f.startswith(header) and not f.endswith('.png')]

            # Sort files based on the numeric part after the header and underscore
            files.sort(key=lambda f: int(re.search(r'_(\d+)', f).group(1)))

            # Return full file paths
            return [os.path.join(dir_path, f) for f in files]

        def get_traj_data(inference_path, trajectories, model_name, header="error"):
            traj_data = []
            for traj in trajectories:
                datapath = os.path.join(inference_path, traj, model_name)
                files = get_ordered_files(datapath, header)
                
                data = []
                print(f"Reading trajectory {traj} from {datapath}...")
                for file in files:
                    data.append(np.load(file))
                
                # Stack the list of arrays along a new batch dimension
                data = np.stack(data, axis=0)
                traj_data.append(data)
            
            return np.stack(traj_data, axis=0)

        # inference_path = "/Volumes/Novus_SB_14TB/nek/nekRS-GNN-devel/3rd_party/gnn/outputs/inference"
        inference_path = "./outputs/inference"
        trajectories = ["rollout_1", "rollout_2", "rollout_3", "rollout_4", "rollout_5"]

        # model_name = "POLY_3_SIZE_32_SEED_64_RESID_3_4_64_3_2_8_none"
        # err_64_none = get_traj_data(inference_path, trajectories, model_name)
        # err_64_none = np.abs(err_64_none).mean(axis=2)

        # model_name = "POLY_3_SIZE_32_SEED_64_RESID_3_4_64_3_2_8_all_to_all_opt"
        # err_64_a2a = get_traj_data(inference_path, trajectories, model_name)
        # err_64_a2a = np.abs(err_64_a2a).mean(axis=2)

        # model_name = "POLY_3_SIZE_32_SEED_64_RESID_3_4_128_3_2_8_none"
        # err_128_none = get_traj_data(inference_path, trajectories, model_name)
        # err_128_none = np.abs(err_128_none).mean(axis=2)

        # model_name = "POLY_3_SIZE_32_SEED_64_RESID_3_4_128_3_2_8_all_to_all_opt"
        # err_128_a2a = get_traj_data(inference_path, trajectories, model_name)
        # err_128_a2a = np.abs(err_128_a2a).mean(axis=2)

        # model_name = "POLY_3_SIZE_32_SEED_64_RESID_3_4_256_3_2_8_none"
        # err_256_none = get_traj_data(inference_path, trajectories, model_name)
        # err_256_none = np.abs(err_256_none).mean(axis=2)

        # model_name = "POLY_3_SIZE_32_SEED_64_RESID_3_4_256_3_2_8_all_to_all_opt"
        # err_256_a2a = get_traj_data(inference_path, trajectories, model_name)
        # err_256_a2a = np.abs(err_256_a2a).mean(axis=2)

        # err_list = [err_64_none, err_128_none, err_256_none, 
        #             err_64_a2a, err_128_a2a, err_256_a2a]
        # color_list = ["red", "red", "red", "blue", "blue", "blue"]
        # ls_list = ["-", "--", "-.", "-", "--", "-.",]               
        # label_list = ["HC=64, None", "HC=128, None", "HC=256, None", 
        #               "HC=64, N-A2A", "HC=128, N-A2A", "HC=256, N-A2A"]


        model_name = "POLY_3_SIZE_32_SEED_64_RESID_3_4_128_3_2_8_all_to_all_opt"
        err_128_a2a_1 = get_traj_data(inference_path, trajectories, model_name)
        err_128_a2a_1 = np.abs(err_128_a2a_1).mean(axis=2)

        model_name = "ROLLOUT_2_POLY_3_SIZE_32_SEED_64_RESID_3_4_128_3_2_8_all_to_all_opt"
        err_128_a2a_2 = get_traj_data(inference_path, trajectories, model_name)
        err_128_a2a_2 = np.abs(err_128_a2a_2).mean(axis=2)

        model_name = "ROLLOUT_5_POLY_3_SIZE_32_SEED_64_RESID_3_4_128_3_2_8_all_to_all_opt"
        err_128_a2a_5 = get_traj_data(inference_path, trajectories, model_name)
        err_128_a2a_5 = np.abs(err_128_a2a_5).mean(axis=2)

        model_name = "ROLLOUT_10_POLY_3_SIZE_32_SEED_64_RESID_3_4_128_3_2_8_all_to_all_opt"
        err_128_a2a_10 = get_traj_data(inference_path, trajectories, model_name)
        err_128_a2a_10 = np.abs(err_128_a2a_10).mean(axis=2)

        err_list = [err_128_a2a_1, err_128_a2a_2, err_128_a2a_5, err_128_a2a_10]
        color_list = ["red", "blue", "blue", "blue"]
        ls_list = ["-", "-", "--", "-.",]               
        label_list = ["K=1", "K=2", "K=5", "K=10"]

        # Determine the number of features from the first dataset (assumed common to all)
        num_features = 3
        steps = np.arange(err_list[0].shape[1])+1

        # Create a subplot for each feature
        fig, axes = plt.subplots(1, num_features, figsize=(5*num_features, 5), sharex=True, sharey=True)

        # Loop over each feature
        for feature in range(num_features):
            ax = axes[feature] if num_features > 1 else axes
            
            # Loop over each dataset
            for err, color, ls, label in zip(err_list, color_list, ls_list, label_list):
                # Extract data for this feature (across trajectories and steps)
                feature_data = err[:, :, feature]
                
                # Compute statistics along the trajectory axis (axis=0)
                avg = feature_data.mean(axis=0)
                min_val = feature_data.min(axis=0)
                max_val = feature_data.max(axis=0)
                
                # Plot the average as a bold line with the given style and label
                line = ax.plot(steps, avg, linewidth=2, color=color, linestyle=ls, label=label)[0]
                
                # Fill between the min and max bounds with a transparent shade of the same color
                ax.fill_between(steps, min_val, max_val, color=color, alpha=0.3)
            
            ax.set_title(f'Feature {feature}')
            ax.set_xlabel('Step')
            ax.set_ylabel('Abs. Error')
            #ax.set_yscale('log')
            #ax.set_xscale('log')
            ax.set_ylim([0,0.02])
            ax.legend(fancybox=False, framealpha=1, prop={'size': 10})

        plt.tight_layout()
        plt.show(block=False)
        plt.savefig('figure.png')
        
        asdf


    if 1 == 0:
        from scipy.stats import norm
        """
        [INFERENCE] Plot error distributions for single-step predictions 
        """
        inference_path = "/Volumes/Novus_SB_14TB/nek/nekRS-GNN-devel/3rd_party/gnn/outputs/inference"
        inference_mode = "single_step"
        model_name = "POLY_3_SIZE_32_SEED_64_RESID_3_4_256_3_2_8_none"
        datapath = f"{inference_path}/{inference_mode}/{model_name}/error_15.npy"
        error = np.load(datapath)

        stats = np.load("./datasets/data_stats.npz")
        data_mean = stats['mean']
        data_std = stats['std']

        # Load data 
        bins = 500

        fig, ax = plt.subplots(1, 3, figsize=(18, 5))
        for idx in range(3):
            # Select the data for the current error component
            data = error[:, idx]/data_std[0,idx]

            weights = np.ones_like(data) / len(data)
            
            # Plot the histogram as a probability density function
            ax[idx].hist(data, bins=bins, weights=weights, alpha=0.6, color='b')
            
            # # Define the range for x: you could use a wider range if needed
            # xmin, xmax = ax[idx].get_xlim()
            # xmin = -5
            # xmax = 5
            # x = np.linspace(xmin, xmax, 200)
            # 
            # # Gaussian (normal) distribution with mean=0 and std=1
            # p = norm.pdf(x, 0, 1)
            
            # # Overlay the Gaussian curve
            # ax[idx].plot(x, p, 'k', linewidth=2)
            
            ax[idx].set_title(f'Error Component {idx+1}')
            ax[idx].set_xlabel('Value')
            ax[idx].set_ylabel('Probability Density')
            ax[idx].set_yscale('log')
            ax[idx].set_xlim([-2,2])
        plt.tight_layout()  # Adjust spacing to prevent overlap
        plt.show(block=False)

        asdf

    if 1 == 0:
        """
        Autocorrelation studies 
        """

        # /Volumes/Novus_SB_14TB/nek/nekrs_cases/examples_v23_gnn/bfs_2/traj_poly_3/DT_1EM2/tinit_75.000000_dtfactor_1/data_rank_0_size_4

        # Load data 
        dtfac = 1
        traj_data_path = f"/Volumes/Novus_SB_14TB/nek/nekrs_cases/examples_v23_gnn/bfs_2/traj_poly_3/DT_1EM2/tinit_75.000000_dtfactor_{dtfac}"

        RANK = 0
        SIZE = 4
        data_dir = traj_data_path + f"/data_rank_{RANK}_size_{SIZE}"


        # mask bounding box 
        #x_range = [0,10]
        #y_range = [-1,1]

        x_range = [10,20]
        y_range = [-1,1]

        # ~~~~~~~ read positions ~~~~~~~ ~~~~~~~ 
        pos = []
        mask = []
        for RANK in range(SIZE): 
            pos_file = f"/Volumes/Novus_SB_14TB/nek/nekrs_cases/examples_v23_gnn/bfs_2/gnn_outputs_poly_3/pos_node_rank_{RANK}_size_{SIZE}.bin"
            pos_rank = np.fromfile(pos_file, dtype=np.float64).reshape((-1,3))
            mask_rank = (pos_rank[:, 0] >= x_range[0]) & (pos_rank[:, 0] <= x_range[1]) & (pos_rank[:, 1] >= y_range[0]) & (pos_rank[:, 1] <= y_range[1])

            pos.append(pos_rank)
            mask.append(mask_rank)
        pos = np.concatenate(pos)
        #mask = np.concatenate(mask)

        #gll_mask = pos[:,1] <= 1
        # mask = (pos[:, 0] >= x_range[0]) & (pos[:, 0] <= x_range[1]) & \
        #        (pos[:, 1] >= y_range[0]) & (pos[:, 1] <= y_range[1])

        # Get a vel field 
        U_temp = []
        for RANK in range(SIZE):
            data_dir = traj_data_path + f"/data_rank_{RANK}_size_{SIZE}"
            U_temp.append( np.fromfile(data_dir + "/" + "u_step_453.bin", dtype=np.float64).reshape((-1,3)) )
        U_temp = np.concatenate(U_temp)

        # plot:
        fig, ax = plt.subplots(figsize=(10,4))
        ax.scatter(pos[np.concatenate(mask),0], pos[np.concatenate(mask),1], 
                   c=U_temp[np.concatenate(mask),0], s=0.5)
        ax.set_aspect('equal')
        plt.show(block=False)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        files_temp = os.listdir(data_dir)
        files = [item for item in files_temp if 'p_step' not in item]
        files.sort(key=lambda x:int(x.split('_')[-1].split('.')[0]))

        # Get the mean field 
        U_full = []
        for i in range(len(files)):
            print(f"{files[i]}")
            U_temp = [] 
            for RANK in range(SIZE):
                data_dir = traj_data_path + f"/data_rank_{RANK}_size_{SIZE}"
                U_temp.append( np.fromfile(data_dir + "/" + files[i], dtype=np.float64).reshape((-1,3))[mask[RANK]] )
            U_temp = np.concatenate(U_temp)
            U_full.append(U_temp)
        U_full = np.stack(U_full)
        U_mean = np.mean(U_full, axis=0)
        np.save(f"./outputs/autocorrelation_analysis/U_mean_xrange_{x_range[0]}_{x_range[1]}_yrange_{y_range[0]}_{y_range[1]}.npy", U_mean)
        U_mean = np.load(f"./outputs/autocorrelation_analysis/U_mean_xrange_{x_range[0]}_{x_range[1]}_yrange_{y_range[0]}_{y_range[1]}.npy")

        fig, ax = plt.subplots(figsize=(5,3))
        ax.scatter(pos[np.concatenate(mask),0], pos[np.concatenate(mask),1], 
                   c=U_mean[:,1], s=1, vmin=-1, vmax=1)
        ax.set_aspect('equal')
        plt.show(block=False)


        # # Split the files into trajectory "chunks" -- needed for averaging 
        # num_chunks = 5
        # chunk_size = len(files) // num_chunks
        # trimmed_files = files[:chunk_size * num_chunks]
        # # Create the chunks
        # files_split = [trimmed_files[i * chunk_size : (i + 1) * chunk_size] for i in range(num_chunks)]

        # Use overlapping chunks 
        chunk_size = 100
        overlap = 60
        files_split = [files[i : i + chunk_size] for i in range(0, len(files) - chunk_size + 1, chunk_size - overlap)]

        rho_ens = []
        s = np.arange(len(files))
        for i in range(len(files_split)): 
            t_chunk = time.time()
            print(f"chunk {i}")
            files = files_split[i]
            # init autocorrelation data. rho[p] is temporal autocorrelation at delay "p"  
            # we have "num_chunks" amount of rho variables. In the end, we want to ensemble average over this.  
            rho = np.zeros((len(files), U_mean.shape[0], U_mean.shape[1]))

            # load the reference snap 
            U_ref = []
            for RANK in range(SIZE):
                U_ref.append( np.fromfile(data_dir + "/" + files[0], dtype=np.float64).reshape((-1,3))[mask[RANK]] )
            U_ref = np.concatenate(U_ref)
            u_ref = U_ref - U_mean

            for s in range(len(files)):
                print("\t",s)
                U_s = []
                for RANK in range(SIZE):
                    U_s.append( np.fromfile(data_dir + "/" + files[s], dtype=np.float64).reshape((-1,3))[mask[RANK]] )
                U_s = np.concatenate(U_s)
                u_s = U_s - U_mean
                rho[s] = u_ref * u_s

            rho_ens.append(rho)
            t_chunk = time.time() - t_chunk
            print(f"\t took {t_chunk}s")

        # Take the ensemble average IN TIME ONLY 
        rho_ens = np.stack(rho_ens) 
        print('computing time avg')
        rho_avg_time = np.mean(rho_ens, axis=0)
        rho_avg_time_norm = rho_avg_time / (rho_avg_time[0] + 1e-15)

        gll_id = list(range(12,434))
        comp = 0
        fig, ax = plt.subplots()
        ax.plot(rho_avg_time_norm[:, gll_id, comp], color='black')
        plt.show(block=False)

        # Take the ensemble average IN SPACE AND TIME 
        print('computing space-time avg')
        rho_avg_spacetime = np.mean(rho_ens, axis=(0,2))
        rho_avg_spacetime_norm = rho_avg_spacetime / rho_avg_spacetime[0]
        time_vec = np.arange(rho_avg_spacetime_norm.shape[0])*1e-2
        cfl_vec = np.arange(rho_avg_spacetime_norm.shape[0])*0.3

        fig, ax = plt.subplots(1,3,figsize=(14,7))
        for comp in range(3):
            ax[comp].plot(time_vec, rho_avg_time_norm[:, ::1000, comp], color='gray', alpha=0.5)
            ax[comp].plot(time_vec, rho_avg_spacetime_norm[:, comp], color='red', lw=2)
            ax[comp].set_ylim([-1,2])
            ax[comp].set_xscale('log')
            ax[comp].set_xlabel('dt')
            ax2 = ax[comp].twiny()
            ax2.plot(cfl_vec, rho_avg_spacetime_norm[:, comp], color='red', lw=2)
            ax2.set_xlabel('CFL')
            ax2.set_xscale('log')
            ax2.grid(False)
        plt.show(block=False)

    if 1 == 0:
        """
        Scatter plots for input/output in surrogate models (traj data)
        """
        SIZE = 8
        traj_data_path = "/Volumes/Novus_SB_14TB/nek/nekrs_cases/examples_v23_gnn/bfs_2/traj_poly_3/DT_1EM2"
        dtfac = 1 


        # Load data 
        data_0 = []
        data_1 = [] 
        data_10 = []
        data_100 = []
        for RANK in range(SIZE):
            print(f"Loading rank {RANK}...")
            data_dir = traj_data_path + f"/tinit_75.000000_dtfactor_{dtfac}/data_rank_{RANK}_size_{SIZE}"
            files_temp = os.listdir(data_dir)
            files = [item for item in files_temp if 'p_step' not in item]
            files.sort(key=lambda x:int(x.split('_')[-1].split('.')[0]))
            
            data_0.append( np.fromfile(data_dir + "/" + files[0], dtype=np.float64).reshape((-1,3)) )
            data_1.append( np.fromfile(data_dir + "/" + files[1], dtype=np.float64).reshape((-1,3)) )
            data_10.append( np.fromfile(data_dir + "/" + files[10], dtype=np.float64).reshape((-1,3)) )
            data_100.append( np.fromfile(data_dir + "/" + files[100], dtype=np.float64).reshape((-1,3)) ) 


        for RANK in range(SIZE): 
            print(f"plotting rank {RANK}")
            fig, ax = plt.subplots(3,1,figsize=(4,8))
            for c in range(3):
                dmin = data_0[RANK][:,c].min()
                dmax = data_0[RANK][:,c].max()
                #ax[c].scatter(data_0[RANK][:,c], data_1[RANK][:,c])
                #ax[c].scatter(data_0[RANK][:,c], data_10[RANK][:,c])
                ax[c].scatter(data_0[RANK][:,c], data_100[RANK][:,c])
                ax[c].set_xlim([dmin, dmax])
                ax[c].set_ylim([dmin, dmax])
                #ax[c].set_aspect('equal')
            plt.savefig(f"./outputs/postproc/bfs_figs/data_rank_{RANK}_dtfac_100.png", transparent=False, dpi=500)
            plt.close()

    if 1 == 1:
        """
        Load a model and plot its loss 
        """

        # ~~~ # modelpath = "/Users/sbarwey/Files/solvers/nekRS-GNN-devel/3rd_party/gnn/saved_models/bfs_factor_dt_1em2_1k_snaps/bfs_factor_10" 

        # ~~~ # lw = 2.5
        # ~~~ # fig, ax = plt.subplots(figsize=(6,7))

        # ~~~ # n_mp_plot = [1,2,4,8,12] 
        # ~~~ # n_hc_plot = [32,64,128,256]
        # ~~~ # n_hc_plot = [256]

        # ~~~ # min_val = min(n_mp_plot)
        # ~~~ # max_val = max(n_mp_plot)
        # ~~~ # normed_values = [0.4 + 0.6 * (val - min_val) / (max_val - min_val) for val in n_mp_plot]
        # ~~~ # colors = [plt.cm.Blues(value) for value in normed_values]
    
        # ~~~ # for n_hc in n_hc_plot:
        # ~~~ #     count = 0
        # ~~~ #     for n_mp in n_mp_plot: 
        # ~~~ #         color = colors[count]

        # ~~~ #         if (n_mp == 1) or (n_mp == 2):
        # ~~~ #             l_a2ao = torch.load(modelpath + f"/POLY_3_RANK_0_SIZE_32_SEED_12_3_4_{n_hc}_3_2_{n_mp}_all_to_all_opt.tar")
        # ~~~ #         else:
        # ~~~ #             if n_hc == 128:
        # ~~~ #                 l_a2ao = torch.load(modelpath + f"/POLY_3_RANK_0_SIZE_16_SEED_12_3_4_{n_hc}_3_2_{n_mp}_all_to_all_opt.tar")
        # ~~~ #             elif n_hc == 256:
        # ~~~ #                 l_a2ao = torch.load(modelpath + f"/POLY_3_RANK_0_SIZE_32_SEED_12_3_4_{n_hc}_3_2_{n_mp}_all_to_all_opt.tar")
        # ~~~ #             else:
        # ~~~ #                 l_a2ao = torch.load(modelpath + f"/POLY_3_RANK_0_SIZE_8_SEED_12_3_4_{n_hc}_3_2_{n_mp}_all_to_all_opt.tar")

        # ~~~ #             

        # ~~~ #         epochs_train = list(range(0, len(l_a2ao['loss_hist_train'])))
        # ~~~ #         epochs_test = list(range(1, len(l_a2ao['loss_hist_train'])+1))
        # ~~~ # 
        # ~~~ #         iters_train = np.array(list(range(0, len(l_a2ao['loss_hist_train_iter']))))/450
        # ~~~ #         iters_max = 4500

        # ~~~ #         if n_hc == 32:
        # ~~~ #             ls='--'
        # ~~~ #         if n_hc == 64:
        # ~~~ #             ls='-'
        # ~~~ #         if n_hc == 128:
        # ~~~ #             ls=':'
        # ~~~ #         if n_hc == 256:
        # ~~~ #             ls='-.'

        # ~~~ #         # # For slides: 
        # ~~~ #         # if (n_hc == 256 and n_mp == 12):
        # ~~~ #         #     pass
        # ~~~ #         # else:
        # ~~~ #         #     ax.plot(epochs_train, l_a2ao['loss_hist_train'], lw=lw, ls=ls, color=color, label=f"(mp,hc)=({n_mp},{n_hc})")
        # ~~~ #         ax.plot(epochs_train, l_a2ao['loss_hist_train'], lw=lw, ls=ls, color=color, label=f"(mp,hc)=({n_mp},{n_hc})")

        # ~~~ #         count += 1
 
        # ~~~ # ax.set_xlabel('Epochs')
        # ~~~ # ax.set_ylabel('Loss')
        # ~~~ # #ax.set_title(f'dt factor = {dtfac}')
        # ~~~ # ax.set_yscale('log')
        # ~~~ # ax.set_ylim([1e-3, 1e0])
        # ~~~ # ax.grid(False)
        # ~~~ # ax.legend(prop={'size': 12}, fancybox=False, framealpha=1)
        # ~~~ # plt.show(block=False)


        # ~~~ # # Aside: same model, effect of seed 
        # ~~~ # n_hc = 256
        # ~~~ # n_mp = 12 
        # ~~~ # l1 = torch.load(modelpath + f"/POLY_3_RANK_0_SIZE_32_SEED_12_3_4_{n_hc}_3_2_{n_mp}_all_to_all_opt.tar")
        # ~~~ # l2 = torch.load(modelpath + f"/POLY_3_RANK_0_SIZE_32_SEED_64_3_4_{n_hc}_3_2_{n_mp}_all_to_all_opt.tar")
        # ~~~ # l3 = torch.load(modelpath + f"/POLY_3_RANK_0_SIZE_32_SEED_96_3_4_{n_hc}_3_2_{n_mp}_all_to_all_opt.tar")
        # ~~~ # l4 = torch.load(modelpath + f"/POLY_3_RANK_0_SIZE_32_SEED_45_3_4_{n_hc}_3_2_{n_mp}_all_to_all_opt.tar")

        # ~~~ # lw = 2
        # ~~~ # fig, ax = plt.subplots(figsize=(6,7))
        # ~~~ # ax.plot(epochs_train, l1['loss_hist_train'], lw=lw, ls=ls, label=f"seed=12")
        # ~~~ # ax.plot(epochs_train, l2['loss_hist_train'], lw=lw, ls=ls, label=f"seed=64")
        # ~~~ # ax.plot(epochs_train, l3['loss_hist_train'], lw=lw, ls=ls, label=f"seed=96")
        # ~~~ # ax.plot(epochs_train, l4['loss_hist_train'], lw=lw, ls=ls, label=f"seed=45")
        # ~~~ # ax.legend(prop={'size': 12}, fancybox=False, framealpha=1)
        # ~~~ # ax.set_yscale('log')
        # ~~~ # plt.show(block=False)


        # Aside: plot loss from log 
        """
        MP = 8 , HC = 128 
        """
        log_file_path_list = ["./saved_models/bfs_factor_dt_1em2_10k_snaps/bfs_factor_10/dgnn_bfs_8_128_na2a.o3742127"]
        data_mp8_hc128 = plot_from_log(log_file_path_list, N_skip=224)

        log_file_path_list = ["./saved_models/bfs_factor_dt_1em2_10k_snaps/bfs_factor_10/dgnn_bfs_8_128_none.o3742130"]
        data_mp8_hc128_none = plot_from_log(log_file_path_list, N_skip=224)

        log_file_path_list = ["./saved_models/bfs_factor_dt_1em2_10k_snaps/bfs_factor_10/dgnn_bfs_8_128_na2a_rollout_2.o4193382"]
        data_mp8_hc128_ro2 = plot_from_log(log_file_path_list, N_skip=224)

        log_file_path_list = ["./saved_models/bfs_factor_dt_1em2_10k_snaps/bfs_factor_10/dgnn_bfs_8_128_na2a_rollout_5.o4193412"]
        data_mp8_hc128_ro5 = plot_from_log(log_file_path_list, N_skip=224)

        log_file_path_list = ["./saved_models/bfs_factor_dt_1em2_10k_snaps/bfs_factor_10/dgnn_bfs_8_128_na2a_rollout_10.o4193413"]
        data_mp8_hc128_ro10 = plot_from_log(log_file_path_list, N_skip=224)

        log_file_path_list = ["./saved_models/bfs_factor_dt_1em2_10k_snaps/bfs_factor_10/dgnn_bfs_8_128_na2a_rollout_1.o4292127"]
        data_mp8_hc128_ro1 = plot_from_log(log_file_path_list, N_skip=224)

        """
        MP = 8, HC = 64
        """
        log_file_path_list = ["./saved_models/bfs_factor_dt_1em2_10k_snaps/bfs_factor_10/dgnn_bfs_8_64_na2a.o3742126"]
        data_mp8_hc64 = plot_from_log(log_file_path_list, N_skip=224)

        log_file_path_list = ["./saved_models/bfs_factor_dt_1em2_10k_snaps/bfs_factor_10/dgnn_bfs_8_64_none.o3742129"]
        data_mp8_hc64_none = plot_from_log(log_file_path_list, N_skip=224)
        
        log_file_path_list = ["./saved_models/bfs_factor_dt_1em2_10k_snaps/bfs_factor_10/dgnn_bfs_8_64_na2a_varcons.o3742168"]
        data_mp8_hc64_last = plot_from_log(log_file_path_list, N_skip=224)
        
        """
        MP = 8, HC = 256
        """
        log_file_path_list = ["./saved_models/bfs_factor_dt_1em2_10k_snaps/bfs_factor_10/dgnn_bfs_8_256_na2a.o3742128"]
        data_mp8_hc256 = plot_from_log(log_file_path_list, N_skip=224)

        log_file_path_list = ["./saved_models/bfs_factor_dt_1em2_10k_snaps/bfs_factor_10/dgnn_bfs_8_256_none.o3742131"]
        data_mp8_hc256_none = plot_from_log(log_file_path_list, N_skip=224)


        fig, ax1 = plt.subplots(figsize=(8,7))
        ax2 = ax1.twinx()

        ax1.set_xlabel('Step')
        ax1.set_ylabel('Loss')

        # # Compare hc = 64, 128, 256 with and without halo 
        # ax1.plot(data_mp8_hc64_none['steps'], data_mp8_hc64_none['loss'], color='tab:red', alpha=0.2, label='MP=8, HC=64, None')
        # ax1.plot(data_mp8_hc64['steps'], data_mp8_hc64['loss'], color='tab:blue', alpha=0.2, label='MP=8, HC=64, N-A2A')

        # ax1.plot(data_mp8_hc128_none['steps'], data_mp8_hc128_none['loss'], color='tab:red', alpha=0.6, label='MP=8, HC=128, None')
        # ax1.plot(data_mp8_hc128['steps'], data_mp8_hc128['loss'], color='tab:blue', alpha=0.6, label='MP=8, HC=128, N-A2A')

        # ax1.plot(data_mp8_hc256_none['steps'], data_mp8_hc256_none['loss'], color='tab:red', alpha=1.0, label='MP=8, HC=256, None')
        # ax1.plot(data_mp8_hc256['steps'], data_mp8_hc256['loss'], color='tab:blue', alpha=1.0, label='MP=8, HC=256, N-A2A')

        # Message passing only in last layer
        # ax1.plot(data_mp8_hc64_last['steps'], data_mp8_hc64_last['loss'], color='tab:green', alpha=0.2, label='MP=8, HC=64, N-A2A-Last')

        # Compare with rollout fine-tuning
        ax1.plot(data_mp8_hc128['steps'], data_mp8_hc128['loss'], color='tab:blue', alpha=0.6, label='MP=8, HC=128, N-A2A')

        ax1.plot(data_mp8_hc128_ro2['steps'] + data_mp8_hc128['steps'][-1], 
                 data_mp8_hc128_ro2['loss'], color='tab:orange', alpha=0.6, label='K_max=2')
        
        ax1.plot(data_mp8_hc128_ro5['steps'] + data_mp8_hc128['steps'][-1], 
                 data_mp8_hc128_ro5['loss'], color='tab:green', alpha=0.6, label='K_max=5')
        
        ax1.plot(data_mp8_hc128_ro10['steps'] + data_mp8_hc128['steps'][-1], 
                 data_mp8_hc128_ro10['loss'], color='tab:red', alpha=0.6, label='K_max=10')
        
        ax1.plot(data_mp8_hc128_ro1['steps'] + data_mp8_hc128['steps'][-1], 
                 data_mp8_hc128_ro1['loss'], color='blue', alpha=0.6, label='K_max=1')

        ax1.grid(False)

        # Create a twin axis that shares the x-axis with ax1

        # Plot lrs on the secondary y-axis (ax2)
        color = 'tab:red'
        ax2.set_ylabel('Learning Rate', color=color)
        ax2.plot(data_mp8_hc128['steps'], data_mp8_hc128['lr'], color=color, label='LR')
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.grid(False)

        # Optionally, you can add a title and improve layout
        fig.tight_layout()

        ax1.set_yscale('log')
        ax1.set_ylim([5e-5, 2e-1])
        ax1.legend(prop={'size': 12})
        plt.show(block=False)



        asdf

        # n_params versus time 
        navg = 100
        lo = 24000
        hi = 25000
        params_mp8_hc64 = 405059
        loss_mp8_hc64 = data_mp8_hc64['loss'][lo:hi].mean()
        t_fp_mp8_hc64 = data_mp8_hc64['t_fp'][lo:hi].mean()

        params_mp8_hc128 = 1604739
        loss_mp8_hc128 = data_mp8_hc128['loss'][lo:hi].mean()
        t_fp_mp8_hc128 = data_mp8_hc128['t_fp'][lo:hi].mean()

        params_mp8_hc256 = 6387971 
        loss_mp8_hc256 = data_mp8_hc256['loss'][lo:hi].mean()
        t_fp_mp8_hc256 = data_mp8_hc256['t_fp'][lo:hi].mean()

        params_mp4_hc64 = 221763
        loss_mp4_hc64 = data_mp4_hc64['loss'][lo:hi].mean()
        t_fp_mp4_hc64 = data_mp4_hc64['t_fp'][lo:hi].mean()

        params_mp4_hc128 = 877699
        loss_mp4_hc128 = data_mp4_hc128['loss'][lo:hi].mean()
        t_fp_mp4_hc128 = data_mp4_hc128['t_fp'][lo:hi].mean()

        params_mp4_hc256 = 3492099
        loss_mp4_hc256 = data_mp4_hc256['loss'][lo:hi].mean()
        t_fp_mp4_hc256 = data_mp4_hc256['t_fp'][lo:hi].mean()

        fig, ax = plt.subplots(1,2, figsize=(12,5))
        ax[0].scatter(params_mp8_hc64, loss_mp8_hc64, color='black')
        ax[0].scatter(params_mp8_hc128, loss_mp8_hc128, color='black')
        ax[0].scatter(params_mp8_hc256, loss_mp8_hc256, color='red')
        ax[0].scatter(params_mp4_hc64, loss_mp4_hc64, color='red')
        ax[0].scatter(params_mp4_hc128, loss_mp4_hc128, color='black')
        ax[0].scatter(params_mp4_hc256, loss_mp4_hc256, color='black')
        ax[0].set_ylabel(f"loss [mean({lo}:{hi})]")
        ax[0].set_xlabel("no. parameters")
        
        ax[1].scatter(t_fp_mp8_hc64, loss_mp8_hc64, color='black')
        ax[1].scatter(t_fp_mp8_hc128, loss_mp8_hc128, color='black')
        ax[1].scatter(t_fp_mp8_hc256, loss_mp8_hc256, color='red')
        ax[1].scatter(t_fp_mp4_hc64, loss_mp4_hc64, color='red')
        ax[1].scatter(t_fp_mp4_hc128, loss_mp4_hc128, color='black')
        ax[1].scatter(t_fp_mp4_hc256, loss_mp4_hc256, color='black')
        ax[1].set_ylabel(f"loss [mean({lo}:{hi})]")
        ax[1].set_xlabel("evaluation time")

        #ax.set_xscale('log')
        #ax.set_yscale('log')
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
        Consistency in training (FOR PAPER)
        """
        POLY = 1
        #temp = '3_4_32_3_2_4'
        temp = '3_4_8_3_2_4' # small
        #temp = '3_4_32_3_5_4' # large 
        r1 = torch.load(f"./saved_models/POLY_{POLY}_RANK_0_SIZE_1_SEED_12_{temp}_none.tar")['loss_hist_train']
        r2 = torch.load(f"./saved_models/POLY_{POLY}_RANK_0_SIZE_8_SEED_12_{temp}_none.tar")['loss_hist_train'] 
        r3 = torch.load(f"./saved_models/POLY_{POLY}_RANK_0_SIZE_8_SEED_12_{temp}_all_to_all.tar")['loss_hist_train']
        lw = 2.5 
        fig, ax = plt.subplots(figsize=(6,5))
        ax.plot(np.arange(len(r1))+1, r1, color='black', lw=lw)
        ax.plot(np.arange(len(r2))+1, r2, color='blue', lw=lw, ls='--')
        ax.plot(np.arange(len(r3))+1, r3, color='red', lw=lw, ls='--')
        ax.set_yscale('log')
        #ax.grid(False)
        ax.set_xlim([1,1500])
        plt.savefig("./outputs/postproc/gnn_verification_for_paper/training_v2.png", dpi=600, transparent=True)
        plt.show(block=False)



    if 1 == 0:
        """
        Looking at consistency in training -- loss versus iter. 
        """
        POLY = 1
        #SIZE_LIST = [1,2,4,8]
        SIZE_LIST = [1,4]
        COLOR_LIST = ['tab:blue', 'tab:orange', 'tab:red', 'tab:green']
        #HALO_LIST = ['none', 'all_to_all', 'send_recv']
        HALO_LIST = ['none', 'all_to_all']

        losses = [] 
        fig, ax = plt.subplots(figsize=(12,6))
        for i in range(len(SIZE_LIST)):
            for j in range(len(HALO_LIST)):
                size = SIZE_LIST[i]
                halo = HALO_LIST[j]
                if (size == 1) and (halo != 'none'):
                    continue 

                # # old gnn 
                # mp = 4 
                # model_path_1 = f"./saved_models/old_gnn/real_grad/POLY_{POLY}_RANK_0_SIZE_{size}_SEED_12_input_channels_3_hidden_channels_32_output_channels_3_nMessagePassingLayers_{mp}_halo_{halo}.tar"
                # model_path_2 = f"./saved_models/old_gnn/hardcode_grad/POLY_{POLY}_RANK_0_SIZE_{size}_SEED_12_input_channels_3_hidden_channels_32_output_channels_3_nMessagePassingLayers_{mp}_halo_{halo}.tar"

                # new gnn 
                mp = 4
                #model_path_1 = f"./saved_models/new_gnn/real_grad/POLY_{POLY}_RANK_0_SIZE_{size}_SEED_12_3_7_32_3_2_{mp}_{halo}.tar"
                model_path_1 = f"./saved_models/POLY_{POLY}_RANK_0_SIZE_{size}_SEED_12_3_4_32_3_2_{mp}_{halo}.tar"
                #model_path_2 = f"./saved_models/new_gnn/hardcode_grad/POLY_{POLY}_RANK_0_SIZE_{size}_SEED_12_3_7_32_3_2_{mp}_{halo}.tar"

                a_1 = torch.load(model_path_1)
                loss_1 = a_1['loss_hist_train']

                #a_2 = torch.load(model_path_2)
                #loss_2 = a_2['loss_hist_train']

                losses.append(loss_1)

                if halo == 'none':
                    marker='o'
                    ls='-'
                elif halo == 'all_to_all':
                    marker='s'
                    ls='--'
                else:
                    marker='^'
                    ls='--'

                color = COLOR_LIST[i]
                ax.plot(np.arange(len(loss_1))+1, loss_1, 
                        marker=marker, color=color, ls=ls, mew=1., lw=1., ms=14, fillstyle='none',
                        label=f"{halo} -- {size} ranks")
                #ax.plot(loss_2, marker='s', color=color, ls=ls, lw=1, ms=15, fillstyle='none')

        ax.set_xlabel('Iterations')
        ax.set_ylabel('Loss')
        ax.set_yscale('log')
        ax.legend(fancybox=False, framealpha=1)
        #ax.set_xlim([1,50])

        plt.show(block=False)

    if 1 == 0:
        """
        Looking at profiler outputs (NEW -- custom timers) 
        """
        timer_dir = "./outputs/profiles/new_timers" 
        
        SIZE_LIST = [1,2,4]
        POLY = 1
        data_all_to_all = [] 
        data_send_recv = []
        data_none = []

        for SIZE in SIZE_LIST:
            data_all_to_all.append( torch.load(f"{timer_dir}/timers_avg_POLY_1_RANK_0_SIZE_{SIZE}_SEED_12_3_4_32_3_2_4_all_to_all.tar") )
            data_send_recv.append( torch.load(f"{timer_dir}/timers_avg_POLY_1_RANK_0_SIZE_{SIZE}_SEED_12_3_4_32_3_2_4_send_recv.tar") )
            data_none.append( torch.load(f"{timer_dir}/timers_avg_POLY_1_RANK_0_SIZE_{SIZE}_SEED_12_3_4_32_3_2_4_none.tar") )

        qoi = "forwardPass"
        qoi = "backwardPass"
        qoi = "loss"
        lb = 10
        fig, ax = plt.subplots()
        for i in range(len(SIZE_LIST)):
            ax.plot( SIZE_LIST[i], data_all_to_all[i][qoi][-lb:-1].mean(), marker='o', lw=0, ms=12, color='black' )  
            ax.plot( SIZE_LIST[i], data_all_to_all[i][qoi][-lb:-1].min(), marker='+', lw=0, ms=12, color='black' )  
            ax.plot( SIZE_LIST[i], data_all_to_all[i][qoi][-lb:-1].max(), marker='+', lw=0, ms=12, color='black' )  

            ax.plot( SIZE_LIST[i], data_send_recv[i][qoi][-lb:-1].mean(), marker='o', lw=0, ms=12, color='red' )
            ax.plot( SIZE_LIST[i], data_send_recv[i][qoi][-lb:-1].min(), marker='+', lw=0, ms=12, color='red' )
            ax.plot( SIZE_LIST[i], data_send_recv[i][qoi][-lb:-1].max(), marker='+', lw=0, ms=12, color='red' )

            ax.plot( SIZE_LIST[i], data_none[i][qoi][-lb:-1].mean(), marker='o', lw=0, ms=12, color='blue' )
            ax.plot( SIZE_LIST[i], data_none[i][qoi][-lb:-1].min(), marker='+', lw=0, ms=12, color='blue' )
            ax.plot( SIZE_LIST[i], data_none[i][qoi][-lb:-1].max(), marker='+', lw=0, ms=12, color='blue' )
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
        # path_32 = "./outputs/postproc/real_gnn_test_4/periodic_after_fix_edges_2/gradient_data_gpu_nondeterministic_POLARIS/tgv_poly_1/float32"
        # path_64 = "./outputs/postproc/real_gnn_test_4/periodic_after_fix_edges_2/gradient_data_gpu_nondeterministic_POLARIS/tgv_poly_1/float32"

        # For paper: 
        path_32 = "./outputs/postproc/gnn_verification_for_paper/tgv_poly_1/float32"
        path_64 = "./outputs/postproc/gnn_verification_for_paper/tgv_poly_1/float32"


        #SIZE_LIST = [1,2,4,8]
        SIZE_LIST = [1,2,4,8,16,32,64] 
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
                    mp = 4
                    seed = 12
                    #str_temp = f"POLY_1_RANK_{RANK}_SIZE_{SIZE}_SEED_{seed}_3_4_32_3_2_{mp}_{halo_mode}.tar" 
                    str_temp = f"POLY_1_RANK_{RANK}_SIZE_{SIZE}_SEED_{seed}_3_4_8_3_2_{mp}_{halo_mode}.tar" 

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

        # # Plot components 
        # ms=250
        # colors={'none': 'red', 'all_to_all': 'blue', 'send_recv': 'green'}
        # ls={'none': '-', 'all_to_all': '-.', 'send_recv': '--'}
        # fig, ax = plt.subplots(1,3,figsize=(16,5))
        # for comp in range(3):
        #     for i in range(len(SIZE_LIST)): 
        #         for halo_mode in HALO_MODE_LIST: 
        #             SIZE = SIZE_LIST[i]
        #             ax[comp].scatter(np.ones(SIZE)*SIZE, data_32[halo_mode][i][:,4], marker='^', 
        #                        color=colors[halo_mode], s=ms, facecolors='none',
        #                        linestyle=ls[halo_mode], linewidth=2,
        #                        label="CPU, FP32" if i == 0 else None)

        #             # ax[comp].scatter(np.ones(SIZE)*SIZE, data_64[halo_mode][i][:,4], marker='s', 
        #             #            color=colors[halo_mode], s=ms, facecolors='none', 
        #             #            linestyle=ls[halo_mode], linewidth=2, 
        #             #            label="CPU, FP64" if i == 0 else None)

        #             ax[comp].set_title('Component %d' %(comp))
        #             ax[comp].set_xlabel('Number of Ranks')

        #             #ax[comp].set_ylim([0.0766, 0.0770])
        #             ax[comp].set_xlim([0.9, 40])
        #             ax[comp].set_xscale('log')
        # #ax.set_xscale('log')
        # #ax[0].legend(fancybox=False, framealpha=1, edgecolor='black', prop={'size': 14})
        # plt.show(block=False)

        
        # Plot loss -- FOR PAPER 
        ms=120
        colors={'none': 'black', 'all_to_all': 'red', 'send_recv': 'green'}
        markers={'none': 's', 'all_to_all': 'o'}
        ls={'none': '-', 'all_to_all': '-', 'send_recv': '-'}
        fig, ax = plt.subplots(figsize=(6,5))
        for i in range(len(SIZE_LIST)): 
            for halo_mode in HALO_MODE_LIST: 
                SIZE = SIZE_LIST[i]
                ax.scatter(np.ones(SIZE)*SIZE, data_32[halo_mode][i][:,4], marker=markers[halo_mode], 
                           color=colors[halo_mode], s=ms, facecolors=colors[halo_mode],
                           linestyle=ls[halo_mode], linewidth=2,
                           label="CPU, FP32" if i == 0 else None)
        ax.set_xlabel('Number of Ranks')
        ax.set_ylabel('Loss')
        #ax.set_xlim([0.9, 40])
        ax.set_xscale('log')
        #ax[0].legend(fancybox=False, framealpha=1, edgecolor='black', prop={'size': 14})
        #plt.savefig('./outputs/postproc/gnn_verification_for_paper/consistency_v2.png', dpi=600, transparent=True)
        plt.show(block=False)


        

    if 1 == 0:
        """
        Looking at graph stats -- number of nodes, edges, etc. 
        """

        # POLY_LIST = [1, 3, 5, 7]  
        NELE_LIST = [8, 16, 20, 24, 32, 40, 48, 56, 64]
        #NELE_LIST = [16, 32, 64, 128, 256, 512]
        POLY_LIST = [5]  
        SIZE_LIST = [1, 2, 4, 8, 16, 32, 64] # 128]

        n_nodes_local_ele = []
        n_nodes_halo_ele = []
        n_edges_ele = []
        for e in range(len(NELE_LIST)):
            Nele = NELE_LIST[e]
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

                        str_temp_1 = f"POLY_{POLY}_RANK_{RANK}_SIZE_{SIZE}_SEED_12_3_7_32_3_2_4_none.tar"
                        str_temp_2 = f"POLY_{POLY}_RANK_{RANK}_SIZE_{SIZE}_SEED_12_3_4_32_3_2_4_none.tar"
                        if os.path.exists(f"./outputs/GraphStatistics/weak_scaling/ne_{Nele}/" + str_temp_1):
                            a = torch.load(f"./outputs/GraphStatistics/weak_scaling/ne_{Nele}/" + str_temp_1)
                            #a = torch.load(f"./outputs/GraphStatistics/weak_scaling/ne_{Nele}_v2/" + str_temp_1)
                        elif os.path.exists(f"./outputs/GraphStatistics/weak_scaling/ne_{Nele}/" + str_temp_2):
                            a = torch.load(f"./outputs/GraphStatistics/weak_scaling/ne_{Nele}/" + str_temp_2)
                            #a = torch.load(f"./outputs/GraphStatistics/weak_scaling/ne_{Nele}_v2/" + str_temp_2)
                        else:
                            a = {}
                            a['n_nodes_local'] = torch.tensor(0)
                            a['n_nodes_halo'] = torch.tensor(0)
                            a['n_edges'] = 0


                        # ~~~~ old 
                        # try: 
                        #     str_temp = f"POLY_{POLY}_RANK_{RANK}_SIZE_{SIZE}_input_channels_3_hidden_channels_32_output_channels_3_nMessagePassingLayers_5_halo_all_to_all.tar"
                        #     a = torch.load("./outputs/GraphStatistics/" + str_temp)
                        # except FileNotFoundError:
                        #     str_temp = f"POLY_{POLY}_RANK_{RANK}_SIZE_{SIZE}_input_channels_3_hidden_channels_32_output_channels_3_nMessagePassingLayers_2_halo_none.tar"
                        #     a = torch.load("./outputs/GraphStatistics/" + str_temp)

                        
                        n_nodes_local[i][j][RANK] = a['n_nodes_local'].item()
                        n_nodes_halo[i][j][RANK] = a['n_nodes_halo'].item()
                        n_edges[i][j][RANK] = a['n_edges']

            n_nodes_local_ele.append(n_nodes_local)
            n_nodes_halo_ele.append(n_nodes_halo)
            n_edges_ele.append(n_edges)


        # Local nodes per rank 
        ms = 100
        fig, ax = plt.subplots(figsize=(8,7))
        for e in range(len(NELE_LIST)):
            for j in range(len(SIZE_LIST)):
                SIZE = SIZE_LIST[j]
                ax.scatter(SIZE*np.ones(SIZE), n_nodes_local_ele[e][0][j], s=ms, color='black')
                ax.text(SIZE, n_nodes_local_ele[e][0][j][0], NELE_LIST[e], color='blue')
                print(f"Nele={NELE_LIST[e]}, SIZE={SIZE}, nodes={n_nodes_local_ele[e][0][j][0]}")
                #ax.scatter(SIZE*np.ones(SIZE), n_nodes_local_ele[e][1][j], s=ms, color='blue', marker=marker[e]) 
                #ax.scatter(SIZE*np.ones(SIZE), n_nodes_local_ele[e][2][j], s=ms, color='red', marker=marker[e]) 
                #ax.scatter(SIZE*np.ones(SIZE), n_nodes_local[3][j], s=ms, color='green') 
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.set_ylabel('Local Graph Nodes')
        ax.set_xlabel('Number of GPUs')
        #ax.legend(framealpha=1)
        plt.show(block=False)

        # ~~~~ # # Halo nodes per rank 
        # ~~~~ # ms = 100
        # ~~~~ # fig, ax = plt.subplots(figsize=(8,7))
        # ~~~~ # for j in range(len(SIZE_LIST)):
        # ~~~~ #     SIZE = SIZE_LIST[j]
        # ~~~~ #     ax.scatter(SIZE*np.ones(SIZE), n_nodes_halo[0][j], s=ms, color='black') 
        # ~~~~ #     ax.scatter(SIZE*np.ones(SIZE), n_nodes_halo[1][j], s=ms, color='blue') 
        # ~~~~ #     ax.scatter(SIZE*np.ones(SIZE), n_nodes_halo[2][j], s=ms, color='red') 
        # ~~~~ #     ax.scatter(SIZE*np.ones(SIZE), n_nodes_halo[3][j], s=ms, color='green') 
        # ~~~~ # ax.set_yscale('log')
        # ~~~~ # ax.set_xscale('log')
        # ~~~~ # ax.set_ylabel('Halo Graph Nodes')
        # ~~~~ # ax.set_xlabel('Number of GPUs')
        # ~~~~ # plt.show(block=False)


        # ~~~~ # # Halo nodes / local nodes 
        # ~~~~ # ms = 100
        # ~~~~ # fig, ax = plt.subplots(figsize=(8,7))
        # ~~~~ # for j in range(len(SIZE_LIST)):
        # ~~~~ #     SIZE = SIZE_LIST[j]
        # ~~~~ #     ax.scatter(SIZE*np.ones(SIZE), n_nodes_halo[0][j]/n_nodes_local[0][j], s=ms, color='black') 
        # ~~~~ #     ax.scatter(SIZE*np.ones(SIZE), n_nodes_halo[1][j]/n_nodes_local[1][j], s=ms, color='blue') 
        # ~~~~ #     ax.scatter(SIZE*np.ones(SIZE), n_nodes_halo[2][j]/n_nodes_local[2][j], s=ms, color='red') 
        # ~~~~ #     ax.scatter(SIZE*np.ones(SIZE), n_nodes_halo[3][j]/n_nodes_local[3][j], s=ms, color='green') 
        # ~~~~ # ax.set_yscale('log')
        # ~~~~ # ax.set_xscale('log')
        # ~~~~ # ax.set_ylabel('Halo Nodes / Local Nodes')
        # ~~~~ # ax.set_xlabel('Number of GPUs')
        # ~~~~ # plt.show(block=False)


        # ~~~~ # # # Total halo nodes (summed over all ranks)
        # ~~~~ # # ms = 100
        # ~~~~ # # fig, ax = plt.subplots(figsize=(8,7))
        # ~~~~ # # for j in range(len(SIZE_LIST)):
        # ~~~~ # #     SIZE = SIZE_LIST[j]
        # ~~~~ # #     ax.scatter(SIZE, np.sum(n_nodes_halo[0][j]), s=ms, color='black') 
        # ~~~~ # #     ax.scatter(SIZE, np.sum(n_nodes_halo[1][j]), s=ms, color='blue') 
        # ~~~~ # #     ax.scatter(SIZE, np.sum(n_nodes_halo[2][j]), s=ms, color='red') 
        # ~~~~ # #     ax.scatter(SIZE, np.sum(n_nodes_halo[3][j]), s=ms, color='green') 
        # ~~~~ # # ax.set_yscale('log')
        # ~~~~ # # ax.set_xscale('log')
        # ~~~~ # # ax.set_ylabel('Total Halo Graph Nodes')
        # ~~~~ # # ax.set_xlabel('Ranks')
        # ~~~~ # # plt.show(block=False)


        # ~~~~ # # Edges per rank 
        # ~~~~ # ms = 100
        # ~~~~ # fig, ax = plt.subplots(figsize=(8,7))
        # ~~~~ # for j in range(len(SIZE_LIST)):
        # ~~~~ #     SIZE = SIZE_LIST[j]
        # ~~~~ #     ax.scatter(SIZE*np.ones(SIZE), n_edges[0][j], s=ms, color='black') 
        # ~~~~ #     ax.scatter(SIZE*np.ones(SIZE), n_edges[1][j], s=ms, color='blue') 
        # ~~~~ #     ax.scatter(SIZE*np.ones(SIZE), n_edges[2][j], s=ms, color='red') 
        # ~~~~ #     ax.scatter(SIZE*np.ones(SIZE), n_edges[3][j], s=ms, color='green') 
        # ~~~~ # ax.set_yscale('log')
        # ~~~~ # ax.set_xscale('log')
        # ~~~~ # ax.set_ylabel('Graph Edges')
        # ~~~~ # ax.set_xlabel('Number of GPUs')
        # ~~~~ # plt.show(block=False)

    if 1 == 0:
        """
        Looking at profiler outputs 
        """
        if 1 == 0: # data generation 
            profile_path = "./outputs/profiles/weak_scale_v2_updated/"
            POLY_LIST = [3,5] # nekrs polynomial order  
            N_MP_LIST = [2,4,6,8] # number of message passing layers 
            N_HC_LIST = [8,16,32] # number of hidden channels 

            # constants 
            seed = 12
            input_channels_node = 3
            input_channels_edge = 4
            output_channels = 3
            hidden_layers = 2 

            SIZE_LIST = [1,2,4,8,16,32,64]
            #SIZE_LIST = [1,2,4,8,16,32,64,128]
            HALO_MODE_LIST = ['none', 'all_to_all', 'send_recv']

            for poly in POLY_LIST:
                for n_mp in N_MP_LIST:
                    for n_hc in N_HC_LIST:
                        t_forwardPass_cuda = {}
                        t_forwardPass_cpu = {}
                        t_indexAdd_cuda = {}
                        t_indexAdd_cpu = {}
                        for i in range(len(HALO_MODE_LIST)):
                            halo = HALO_MODE_LIST[i]
                            t_forwardPass_cuda[halo] = []
                            t_forwardPass_cpu[halo] = []
                            t_indexAdd_cuda[halo] = []
                            t_indexAdd_cpu[halo] = []
                            for j in range(len(SIZE_LIST)):
                                size = SIZE_LIST[j]
                                t_forwardPass_cuda[halo].append(np.zeros(size))
                                t_forwardPass_cpu[halo].append(np.zeros(size))
                                t_indexAdd_cuda[halo].append(np.zeros(size))
                                t_indexAdd_cpu[halo].append(np.zeros(size))
                                for k in range(size):
                                    rank = k 
                                    # file_str = f"POLY_{poly}_RANK_{rank}_SIZE_{size}_input_channels_3_hidden_channels_{n_hc}_output_channels_3_nMessagePassingLayers_{n_mp}_halo_{halo}.tar"
                                    file_str = f"POLY_{poly}_RANK_{rank}_SIZE_{size}_SEED_{seed}_{input_channels_node}_{input_channels_edge}_{n_hc}_{output_channels}_{hidden_layers}_{n_mp}_{halo}.tar"

                                    # load profile data 
                                    try: 
                                        temp_prof = torch.load(profile_path + file_str)
                                        #temp_prof = torch.load("./outputs/profiles_old/" + file_str)
                                        # print(temp_prof.table(sort_by="cpu_time_total", row_limit=10))
                                        key_list = [] 
                                        for key_id in range(len(temp_prof)):
                                            key_list.append(temp_prof[key_id].key)
                                        idx_key = key_list.index(f'[RANK {rank}] FORWARD PASS') 
                                        cuda_time = temp_prof[idx_key].cuda_time # in microseconds,averaged over runs  
                                        cpu_time = temp_prof[idx_key].cpu_time # in microseconds  

                                        if 'aten::index_add_' in key_list:
                                            idx_key = key_list.index('aten::index_add_')
                                            indexAdd_cuda_time = temp_prof[idx_key].cuda_time
                                            indexAdd_cpu_time = temp_prof[idx_key].cpu_time
                                        else:
                                            indexAdd_cuda_time = 0.
                                            indexAdd_cpu_time = 0.

                                    except FileNotFoundError:
                                        print(f"FileNotFound: {file_str}")
                                        cuda_time = 0
                                        cpu_time = 0
                                        indexAdd_cuda_time = 0
                                        indexAdd_cpu_time = 0

                                    t_forwardPass_cuda[halo][j][k] = cuda_time
                                    t_forwardPass_cpu[halo][j][k] = cpu_time
                                    t_indexAdd_cuda[halo][j][k] = indexAdd_cuda_time
                                    t_indexAdd_cpu[halo][j][k] = indexAdd_cpu_time
                                    print(f"[POLY {poly}, N_MP {n_mp}, N_HC {n_hc}, SIZE {size}, RANK {rank}] -- cuda_time = {cuda_time} us, indexAdd_cuda_time = {indexAdd_cuda_time} us")

                        
                        # Write the data 
                        for halo in HALO_MODE_LIST:
                            x_axis = np.array(SIZE_LIST)
                            y_axis_mean = np.zeros_like(x_axis)
                            y_axis_max = np.zeros_like(x_axis)
                            y_axis_min = np.zeros_like(x_axis)
                            for j in range(len(SIZE_LIST)):
                                y_axis_mean[j] = t_forwardPass_cuda[halo][j].mean()
                                y_axis_max[j] = t_forwardPass_cuda[halo][j].max()
                                y_axis_min[j] = t_forwardPass_cuda[halo][j].min()
                            
                            temp_name = f"POLY_{poly}_SEED_{seed}_{input_channels_node}_{input_channels_edge}_{n_hc}_{output_channels}_{hidden_layers}_{n_mp}_{halo}"
                            np.save(profile_path + f"{temp_name}_mean_cuda.npy", y_axis_mean)
                            np.save(profile_path + f"{temp_name}_max_cuda.npy", y_axis_min)
                            np.save(profile_path + f"{temp_name}_min_cuda.npy", y_axis_max)

                        for halo in HALO_MODE_LIST:
                            x_axis = np.array(SIZE_LIST)
                            y_axis_mean = np.zeros_like(x_axis)
                            y_axis_max = np.zeros_like(x_axis)
                            y_axis_min = np.zeros_like(x_axis)
                            for j in range(len(SIZE_LIST)):
                                y_axis_mean[j] = t_forwardPass_cpu[halo][j].mean()
                                y_axis_max[j] = t_forwardPass_cpu[halo][j].max()
                                y_axis_min[j] = t_forwardPass_cpu[halo][j].min()
                            
                            temp_name = f"POLY_{poly}_SEED_{seed}_{input_channels_node}_{input_channels_edge}_{n_hc}_{output_channels}_{hidden_layers}_{n_mp}_{halo}"
                            np.save(profile_path + f"{temp_name}_mean_cpu.npy", y_axis_mean)
                            np.save(profile_path + f"{temp_name}_max_cpu.npy", y_axis_min)
                            np.save(profile_path + f"{temp_name}_min_cpu.npy", y_axis_max)

                        # Write the data -- indexAdd
                        for halo in HALO_MODE_LIST:
                            x_axis = np.array(SIZE_LIST)
                            y_axis_mean = np.zeros_like(x_axis)
                            y_axis_max = np.zeros_like(x_axis)
                            y_axis_min = np.zeros_like(x_axis)
                            for j in range(len(SIZE_LIST)):
                                y_axis_mean[j] = t_indexAdd_cuda[halo][j].mean()
                                y_axis_max[j] = t_indexAdd_cuda[halo][j].max()
                                y_axis_min[j] = t_indexAdd_cuda[halo][j].min()
                            
                            temp_name = f"POLY_{poly}_SEED_{seed}_{input_channels_node}_{input_channels_edge}_{n_hc}_{output_channels}_{hidden_layers}_{n_mp}_{halo}"
                            np.save(profile_path + f"{temp_name}_mean_indexAdd_cuda.npy", y_axis_mean)
                            np.save(profile_path + f"{temp_name}_max_indexAdd_cuda.npy", y_axis_min)
                            np.save(profile_path + f"{temp_name}_min_indexAdd_cuda.npy", y_axis_max)

                        for halo in HALO_MODE_LIST:
                            x_axis = np.array(SIZE_LIST)
                            y_axis_mean = np.zeros_like(x_axis)
                            y_axis_max = np.zeros_like(x_axis)
                            y_axis_min = np.zeros_like(x_axis)
                            for j in range(len(SIZE_LIST)):
                                y_axis_mean[j] = t_indexAdd_cpu[halo][j].mean()
                                y_axis_max[j] = t_indexAdd_cpu[halo][j].max()
                                y_axis_min[j] = t_indexAdd_cpu[halo][j].min()
                            
                            temp_name = f"POLY_{poly}_SEED_{seed}_{input_channels_node}_{input_channels_edge}_{n_hc}_{output_channels}_{hidden_layers}_{n_mp}_{halo}"
                            np.save(profile_path + f"{temp_name}_mean_indexAdd_cpu.npy", y_axis_mean)
                            np.save(profile_path + f"{temp_name}_max_indexAdd_cpu.npy", y_axis_min)
                            np.save(profile_path + f"{temp_name}_min_indexAdd_cpu.npy", y_axis_max)

        # Scaling plots 
        if 1 == 1:
            profile_path = "./outputs/profiles/weak_scale_v2_updated/"
            HALO_MODE_LIST = ['none', 'all_to_all', 'send_recv']
            seed = 12
            input_channels_node = 3
            input_channels_edge = 4
            output_channels = 3 
            hidden_layers = 2

            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # ~~~~ the effect of n_mp layers, for fixed poly and n_hc 
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            poly = 5
            n_hc = 8
            N_MP_LIST = [2,4,6,8]
            #N_MP_LIST = [2,6]
            data_str = 'cuda'
            #data_str = 'cpu'
            #data_str = 'indexAdd_cuda'
            #data_str = 'indexAdd_cpu'

            if poly==3:
                #n_nodes_local = 116400
                #n_nodes_local = np.array([110592, 117504, 117504, 117504, 128304, 128304, 128304]) # includes halo 
                n_nodes_local = np.array([110592, 112896, 112896, 112896, 116400, 116400, 116400]) # no halo nodes
            if poly==5:
                #n_nodes_local = 528080
                #n_nodes_local = np.array([512000, 531200, 531200, 531200, 560720, 560720, 560720]) # includes halo 
                n_nodes_local = np.array([512000, 518400, 518400, 518400, 528080, 528080, 528080]) # no halo nodes 

            norm = Normalize(vmin=np.min(N_MP_LIST), vmax=np.max(N_MP_LIST))

            data_all_mean = [] 
            data_all_max = [] 
            data_all_min = []

            data_none_mean = []
            data_none_max = []
            data_none_min = []

            data_sr_mean = []
            data_sr_max = []
            data_sr_min = []

            for n_mp in N_MP_LIST:
                halo = 'all_to_all'
                temp_name = f"POLY_{poly}_SEED_{seed}_{input_channels_node}_{input_channels_edge}_{n_hc}_{output_channels}_{hidden_layers}_{n_mp}_{halo}"
                data_all_mean.append(np.load(profile_path + f"{temp_name}_mean_{data_str}.npy"))
                data_all_max.append(np.load(profile_path + f"{temp_name}_max_{data_str}.npy"))
                data_all_min.append(np.load(profile_path + f"{temp_name}_min_{data_str}.npy"))

                halo = 'none'
                temp_name = f"POLY_{poly}_SEED_{seed}_{input_channels_node}_{input_channels_edge}_{n_hc}_{output_channels}_{hidden_layers}_{n_mp}_{halo}"
                data_none_mean.append(np.load(profile_path + f"{temp_name}_mean_{data_str}.npy"))
                data_none_max.append(np.load(profile_path + f"{temp_name}_max_{data_str}.npy"))
                data_none_min.append(np.load(profile_path + f"{temp_name}_min_{data_str}.npy"))

                halo = 'send_recv'
                temp_name = f"POLY_{poly}_SEED_{seed}_{input_channels_node}_{input_channels_edge}_{n_hc}_{output_channels}_{hidden_layers}_{n_mp}_{halo}"
                data_sr_mean.append(np.load(profile_path + f"{temp_name}_mean_{data_str}.npy"))
                data_sr_max.append(np.load(profile_path + f"{temp_name}_max_{data_str}.npy"))
                data_sr_min.append(np.load(profile_path + f"{temp_name}_min_{data_str}.npy"))

            #x_axis = np.array([1,2,4,8,16,32,64,128])
            x_axis = np.array([1,2,4,8,16,32,64])

            # Get total throughput: n_nodes / time 
            eff_all = [] 
            for i in range(len(N_MP_LIST)):
                t_fp = data_all_mean[i]
                total_nodes = (n_nodes_local*x_axis)
                total_throughput = total_nodes/t_fp
                serial_throughput = total_throughput[0]
                eff = total_throughput/(serial_throughput * x_axis)
                eff_all.append(eff)

            eff_sr = [] 
            for i in range(len(N_MP_LIST)):
                t_fp = data_sr_mean[i]
                total_nodes = (n_nodes_local*x_axis)
                total_throughput = total_nodes/t_fp
                serial_throughput = total_throughput[0]
                eff = total_throughput/(serial_throughput * x_axis)
                eff_sr.append(eff)

            lw = 1.5 
            ms = 14
            fig, ax = plt.subplots(1,2,figsize=(10,6), sharex=True)
            for i in range(len(N_MP_LIST)):
                color = cm.viridis(norm(N_MP_LIST[i]))
                ax[0].plot(x_axis[:-1], data_none_mean[i][:-1], color=color, lw=lw, marker='o', ms=ms, mew=1.5, fillstyle='none', mec=color, label='none')
                ax[0].plot(x_axis[:-1], data_all_mean[i][:-1],  color=color, lw=lw, marker='s', ms=ms, mew=1.5, fillstyle='none', mec=color, ls='--', label='all_to_all')
                ax[0].plot(x_axis[:-1], data_sr_mean[i][:-1],   color=color, lw=lw, marker='^', ms=ms, mew=1.5, fillstyle='none', mec=color, ls='-.', label='send_recv')
                ax[1].plot(x_axis, eff_all[i]*100,    color=color, lw=lw, marker='s', ms=ms, mew=1.5, fillstyle='none', mec=color, ls='--')
                ax[1].plot(x_axis, eff_sr[i]*100,     color=color, lw=lw, marker='^', ms=ms, mew=1.5, fillstyle='none', mec=color, ls='-.')

            ax[0].set_xscale('log')
            ax[0].set_yscale('log')
            ax[0].set_xlabel('nGPU')
            ax[0].set_ylabel(f'{data_str} time [us]')
            ax[0].set_title(f"poly={poly}, hc={n_hc}")
            ax[0].set_ylim([1e3, 1e6])
            #ax[0].legend(fancybox=False, framealpha=1)

            ax[1].set_xscale('log')
            ax[1].set_xlabel('nGPU')
            ax[1].set_ylabel('Throughput Efficiency [%]')
            ax[1].set_title(f"poly={poly}, hc={n_hc}")
            ax[1].set_ylim([0., 105])

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


    if 1 == 0:
        """
        Test profiler output 
        """
        a = torch.load('./outputs/profiles/weak_scale/POLY_3_RANK_3_SIZE_32_SEED_12_3_4_32_3_2_4_all_to_all.tar')
        b = torch.load('./outputs/profiles/weak_scale/POLY_3_RANK_3_SIZE_32_SEED_12_3_4_32_3_2_4_none.tar')

        a_keys = [] 
        b_keys = []
        for i in range(len(a)):
            a_keys.append(a[i].key)
        for i in range(len(b)):
            b_keys.append(b[i].key)

        c = [item for item in a if item not in b]



    if 1 == 0:
        """
        check multiscale 
        """

        gnn_outputs_path = "./outputs/temp/gnn_outputs_poly_3"
        #gnn_outputs_path = "./outputs/temp/gnn_outputs_poly_3_multiscale"
        path_to_pos = f"{gnn_outputs_path}/pos_node_rank_0_size_8.bin"
        path_to_ei = f"{gnn_outputs_path}/edge_index_rank_0_size_8.bin"
        path_to_eid = f"{gnn_outputs_path}/node_element_ids_rank_0_size_8.bin"

        pos = np.fromfile(path_to_pos, dtype=np.float64).reshape((-1,3))
        ei = np.fromfile(path_to_ei, dtype=np.int32).reshape((-1,2)).T 
        ei = ei.astype(np.int64)
        eid = np.fromfile(path_to_eid, dtype=np.int32)

        # Plot graph 
        element_id = 0
        mask = eid == element_id
        pos_e = pos[mask]
       
        send = list(ei[0,:])
        recv = list(ei[1,:])
        last_idx = len(recv) - recv[::-1].index(63) - 1
        ei_e = ei[:, :last_idx]


        # remove all edges above 63 
        send = ei_e[0,:]
        idx_keep = send <= 63
        ei_e = ei_e[:, idx_keep]


        from torch_geometric.data import Data
        import torch_geometric.utils as utils

        data_plot = Data(pos = torch.tensor(pos_e), x = torch.tensor(pos_e), edge_index = torch.tensor(ei_e)) 
        G = utils.to_networkx(data=data_plot)
        pos = dict(enumerate(np.array(data_plot.pos)))
        node_xyz = np.array([pos[v] for v in sorted(G)])
        edge_xyz = np.array([(pos[u], pos[v]) for u, v in G.edges()])
        pos = data_plot.pos
        ms = 100
        lw_edge = 2 
        lw_marker = 0.1 

        fig = plt.figure(figsize=(12,8))
        ax = fig.add_subplot(111, projection="3d")
        
        # Plot the edges
        count = 0 
        for vizedge in edge_xyz:
            ax.plot(*vizedge.T, color="black", lw=lw_edge, alpha=0.1)
            #ax.plot(*vizedge.T, color="black", alpha=0.3)
            count += 1

        # plot the nodes 
        #ax.scatter(*pos.T, s=ms, ec='none', lw=lw_marker, c='red', alpha=1)
        ax.scatter(*pos.T, s=ms, ec='black', lw=lw_marker, c=pos[:,2], alpha=1, cmap='Reds')

        ax.set_axis_off()
        ax.grid(False)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.view_init(elev=34, azim=-24, roll=0)
        ax.set_aspect('equal')
        fig.tight_layout()
        plt.show(block=False)

