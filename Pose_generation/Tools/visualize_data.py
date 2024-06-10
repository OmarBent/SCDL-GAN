import numpy as np
import scipy.io as sio

import matplotlib.pyplot as plt
from .centeredscaled import centeredscaled
#import viz_florence as viz


def visualize_skeleton(skeleton):
    fig = plt.figure(figsize=(19.2, 10.8))
    ax = plt.gca(projection='3d')
    ob = viz.Ax3DPose(ax)
    ob.update(skeleton)
    plt.show(block=False)
    fig.canvas.draw()
    plt.pause(0.8)

def visualize_sequence(sequence):
    sequence = np.asarray(sequence)
    nframes = sequence.shape[0]

    # === Plot and animate ===
    fig = plt.figure(figsize=(19.2, 10.8))
    ax = plt.gca(projection='3d')

    ob = viz.Ax3DPose(ax)

    for i in range(nframes):
        print(i)
        ob.update(centeredscaled(sequence[i, :]))

        plt.show(block=False)
        fig.canvas.draw()
        # plt.axis('off')
        plt.pause(0.01)
    plt.close()


# Plot generated videos in one figure
def qualitative_results():
    w = 10
    h = 10
    fig = plt.figure(figsize=(8, 8))
    columns = 4
    rows = 5
    for i in range(1, columns * rows + 1):
        img = np.random.randint(10, size=(h, w))
        fig.add_subplot(rows, columns, i)
        plt.imshow(img)
    plt.show()


if __name__ == '__main__':
    codes = sio.loadmat()
