"""
Postprocess trained model (no DDP) 
"""
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 22})

import torch



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

