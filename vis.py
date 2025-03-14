#import imageio
import os
import numpy as np
import matplotlib.pyplot as plt

save_path = 'vis'

with open('progress.txt', 'r') as f:
    lines = f.read().splitlines()

for i,l in enumerate(lines):
    lines[i] = l.split('\t')

if not os.path.exists(save_path):
    os.mkdir(save_path)

i_x = lines[0].index('Epoch')
lines = np.array(lines)

x = lines[1:, i_x].astype(np.int)
#print(x)
for i in range(0, lines.shape[1]):
    yname = lines[0,i]
    xname = 'steps * 1000'
    plt.xlabel(xname)
    plt.ylabel(yname)
    y = lines[1:, i]
    if y[1] == '':
        for j in range(1, len(y)):
            if y[j] == '':
                y[j] = y[j-1]
    plt.plot(x, y.astype(np.float))
    plt.savefig(os.path.join(save_path, yname+'.png'))
    plt.cla()


# smooth the curves

def smooth(arr, weight=0.9): #weight是平滑度，tensorboard 默认0.6
    last = arr[0]
    smoothed = []
    for point in arr:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return np.array(smoothed)

save_path = 'vis_smooth'
if not os.path.exists(save_path):
    os.mkdir(save_path)

for i in range(0, lines.shape[1]):
    yname = lines[0,i]
    xname = 'steps * 1000'
    plt.xlabel(xname)
    plt.ylabel(yname)
    plt.plot(x, smooth(lines[1:, i].astype(np.float)))
    plt.savefig(os.path.join(save_path, yname+'.png'))
    plt.cla()