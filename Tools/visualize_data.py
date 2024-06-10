import numpy as np
import scipy.io as sio
import Tools.viz_florence as viz
import matplotlib.pyplot as plt
from Tools.centeredscaled import centeredscaled


def visualize_sequence(sequence):
    sequence = np.asarray(sequence)
    nframes = sequence.shape[0]

    # === Plot and animate ===
    fig = plt.figure(figsize=(19.2, 10.8))
    ax = plt.gca(projection='3d')

    ob = viz.Ax3DPose(ax)

    for i in range(nframes):
        ob.update(centeredscaled(sequence[i, :]))

        plt.show(block=False)
        fig.canvas.draw()
        # plt.axis('off')
        plt.pause(0.8)
    plt.close()


if __name__ == '__main__':
    codes = sio.loadmat()