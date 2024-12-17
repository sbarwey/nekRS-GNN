import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
import time

# Load mesh 
#mesh = point_cloud.delaunay_3d()
mesh = pv.read("/Users/sbarwey/Files/gmsh_files/bfs_nek/bfs.msh")

# visualize mesh 
# p = pv.Plotter()
# p.add_mesh(mesh, show_edges=True, color='white')  # You can customize appearance
# p.show_bounds(
#     grid='front',          # Options: 'front', 'back', 'all'. Where to display the grid.
#     location='outer',      # Options: 'all', 'front', 'back', 'outer'.
#     all_edges=True,        # Show all edges of the bounding box
#     #corner_factor=0.5,     # Fractional position of the labels along the axis
#     xtitle='X Axis',       # Label for the X-axis
#     ytitle='Y Axis',       # Label for the Y-axis
#     ztitle='Z Axis',       # Label for the Z-axis
#     fmt="%.2f",            # Format for the tick labels
#     font_size=12,          # Font size of the labels
#     color='black',         # Color of the labels and ticks
#     show_xlabels=True,     # Show X-axis labels
#     show_ylabels=True,     # Show Y-axis labels
#     show_zlabels=True,     # Show Z-axis labels
# )
# p.add_axes()
# p.show()

# Load 3d coordinates and velocity field  
data_path = "/Users/sbarwey/Files/solvers/nekRS-GNN-devel/3rd_party/gnn/outputs/inference"
N_snaps = 5 
for i in range(N_snaps):
    pos_path = data_path + f"/pos_{i}.npy"
    vel_path = data_path + f"/target_{i}.npy"

    pos = np.load(pos_path)
    vel = np.load(vel_path)

    # 1. create a pyvista point cloud 
    point_cloud = pv.PolyData(pos)
    point_cloud['vel_x'] = vel[:,0]
    point_cloud['vel_y'] = vel[:,1]
    point_cloud['vel_z'] = vel[:,2]
    point_cloud['vel_mag'] = np.linalg.norm(vel, axis=1)


    print("interpolating...")
    t_interp = time.time()
    imesh = mesh.interpolate(point_cloud, n_points=6)
    #imesh = mesh.interpolate(point_cloud, radius=0.2)
    t_interp = time.time() - t_interp
    print(f"interpolation took {t_interp} sec")

    # # 3. extract contours 
    # scalars = 'vel_mag'
    # iso_values = [0.1, 0.2, 0.3]
    # contours = imesh.contour(isosurfaces=iso_values, scalars=scalars)

    # ~~~~ # # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~~ # # 4. visualize contours 
    # ~~~~ # plotter = pv.Plotter()
    # ~~~~ # plotter.add_mesh(
    # ~~~~ #     contours,
    # ~~~~ #     scalars='vel_mag',
    # ~~~~ #     cmap='viridis',
    # ~~~~ #     opacity=0.6,
    # ~~~~ #     show_scalar_bar=True,
    # ~~~~ # )
    # ~~~~ # 
    # ~~~~ # # Optionally add the original point cloud for reference
    # ~~~~ # plotter.add_mesh(
    # ~~~~ #     imesh,
    # ~~~~ #     color='white',
    # ~~~~ #     show_edges=True,
    # ~~~~ #     edge_color="black",
    # ~~~~ #     opacity=0.2,
    # ~~~~ # )
    # ~~~~ # 
    # ~~~~ # # Display axes and show the plot
    # ~~~~ # plotter.show_bounds(
    # ~~~~ #     grid='front',          # Options: 'front', 'back', 'all'. Where to display the grid.
    # ~~~~ #     location='outer',      # Options: 'all', 'front', 'back', 'outer'.
    # ~~~~ #     all_edges=True,        # Show all edges of the bounding box
    # ~~~~ #     #corner_factor=0.5,     # Fractional position of the labels along the axis
    # ~~~~ #     xtitle='X Axis',       # Label for the X-axis
    # ~~~~ #     ytitle='Y Axis',       # Label for the Y-axis
    # ~~~~ #     ztitle='Z Axis',       # Label for the Z-axis
    # ~~~~ #     fmt="%.2f",            # Format for the tick labels
    # ~~~~ #     font_size=12,          # Font size of the labels
    # ~~~~ #     color='black',         # Color of the labels and ticks
    # ~~~~ #     show_xlabels=True,     # Show X-axis labels
    # ~~~~ #     show_ylabels=True,     # Show Y-axis labels
    # ~~~~ #     show_zlabels=True,     # Show Z-axis labels
    # ~~~~ # )
    # ~~~~ # plotter.add_axes()
    # ~~~~ # plotter.show()

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Plot planar surfaces 
    # Example: Slice at z = 0.5
    origin = (0, 0, 1)
    normal = (0, 0, 1)

    # Extract the slice
    slice_plane = imesh.slice(
        origin=origin,
        normal=normal,
        generate_triangles=True  # Ensures the output is a triangulated surface
    )

    # Initialize the plotter
    plotter = pv.Plotter(off_screen=True)

    # Add the slice to the plotter
    plotter.add_mesh(
        slice_plane,
        clim = [-0.5,0.5],
        scalars='vel_x',
        cmap='seismic',  # Choose a colormap
        show_scalar_bar=True,
        show_edges=False
    )

    # Set the camera to look along the Z-axis
    slice_origin = slice_plane.center
    z_pos = origin[2]  # Z position of your plane
    distance = 50      # Distance from the plane (adjust as needed)
    plotter.camera_position = [
        (slice_origin[0], slice_origin[1], z_pos + distance),  # Camera position above the plane
        (slice_origin[0], slice_origin[1], z_pos),             # Focal point (center of the plane)
        (0, 1, 0),                                 # View-up vector
    ]

    # Display axes and show the plot
    plotter.show_bounds(
        grid='front',          # Options: 'front', 'back', 'all'. Where to display the grid.
        location='outer',      # Options: 'all', 'front', 'back', 'outer'.
        all_edges=True,        # Show all edges of the bounding box
        corner_factor=0.5,     # Fractional position of the labels along the axis
        xtitle='X Axis',       # Label for the X-axis
        ytitle='Y Axis',       # Label for the Y-axis
        ztitle='Z Axis',       # Label for the Z-axis
        fmt="%.2f",            # Format for the tick labels
        font_size=12,          # Font size of the labels
        color='black',         # Color of the labels and ticks
        show_xlabels=True,     # Show X-axis labels
        show_ylabels=True,     # Show Y-axis labels
        show_zlabels=True,     # Show Z-axis labels
    )
    plotter.add_axes()
    plotter.add_text(f"Step {i}", position="upper_left", font_size=20, color='black')
    plotter.show(screenshot=f"{data_path}/plot_{i}.png")
