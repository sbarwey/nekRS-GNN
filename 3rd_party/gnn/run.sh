case_path="/Users/sbarwey/Files/solvers/nekRS-GNN-devel/examples/tgv/gnn_outputs_poly_1/"

mpirun -np 1 python main.py backend=gloo halo_swap_mode=none gnn_outputs_path=${case_path}
mpirun -np 2 python main.py backend=gloo halo_swap_mode=none gnn_outputs_path=${case_path}
mpirun -np 4 python main.py backend=gloo halo_swap_mode=none gnn_outputs_path=${case_path}
mpirun -np 8 python main.py backend=gloo halo_swap_mode=none gnn_outputs_path=${case_path}

mpirun -np 1 python main.py backend=gloo halo_swap_mode=all_to_all gnn_outputs_path=${case_path}
mpirun -np 2 python main.py backend=gloo halo_swap_mode=all_to_all gnn_outputs_path=${case_path}
mpirun -np 4 python main.py backend=gloo halo_swap_mode=all_to_all gnn_outputs_path=${case_path} 
mpirun -np 8 python main.py backend=gloo halo_swap_mode=all_to_all gnn_outputs_path=${case_path}

mpirun -np 1 python main.py backend=gloo halo_swap_mode=send_recv gnn_outputs_path=${case_path}
mpirun -np 2 python main.py backend=gloo halo_swap_mode=send_recv gnn_outputs_path=${case_path}
mpirun -np 4 python main.py backend=gloo halo_swap_mode=send_recv gnn_outputs_path=${case_path} 
mpirun -np 8 python main.py backend=gloo halo_swap_mode=send_recv gnn_outputs_path=${case_path}
