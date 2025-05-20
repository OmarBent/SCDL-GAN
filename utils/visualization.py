"""
Functions to visualize human poses and feature space using 3D plotting and t-SNE embedding.
"""

import numpy as np
import matplotlib.pyplot as plt
import pylab
import scipy.io as sio
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE

from utils.kendall_ops import centeredscaled, resampling
from utils.data_utils import tsne_data_format_exp


class Ax3DPose:
    def __init__(self, ax, lcolor="#3498db", rcolor="#e74c3c"):
        """
        Initialize a 3D pose visualizer.

        Args:
            ax: Matplotlib 3D axis object.
            lcolor: Color for the left side of the body.
            rcolor: Color for the right side of the body.
        """
        self.I = np.array([1, 2, 2, 4, 5, 2, 7, 8, 3, 10, 11, 3, 13, 14]) - 1  # Start joints
        self.J = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]) - 1  # End joints
        self.LR = np.array([0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0], dtype=bool)  # Left/right flag

        self.ax = ax
        self.plots = []
        vals = np.zeros((15, 3))  # Placeholder for joint coordinates

        # Create line plots for each limb segment
        for i in np.arange(len(self.I)):
            x = [vals[self.I[i], 0], vals[self.J[i], 0]]
            y = [vals[self.I[i], 1], vals[self.J[i], 1]]
            z = [vals[self.I[i], 2], vals[self.J[i], 2]]
            color = lcolor if self.LR[i] else rcolor
            self.plots.append(self.ax.plot(x, y, z, lw=3, marker='.', markersize=10, c=color))

        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")
        self.ax.set_zlabel("z")

    def update(self, channels, lcolor="#3498db", rcolor="#e74c3c"):
        """
        Update the pose with new joint coordinates.

        Args:
            channels: A flat array of shape (45,) representing 15 joints × 3 coordinates.
            lcolor: Color for the left side.
            rcolor: Color for the right side.
        """
        assert channels.size == 45, f"Expected 45 values for 15 joints, got {channels.size}"
        vals = channels.reshape(15, 3)

        for i in np.arange(len(self.I)):
            x = [vals[self.I[i], 0], vals[self.J[i], 0]]
            y = [vals[self.I[i], 1], vals[self.J[i], 1]]
            z = [vals[self.I[i], 2], vals[self.J[i], 2]]
            self.plots[i][0].set_xdata(x)
            self.plots[i][0].set_ydata(y)
            self.plots[i][0].set_3d_properties(z)
            self.plots[i][0].set_color(lcolor if self.LR[i] else rcolor)

        # Center the view around the root joint
        r = 1
        xroot = vals[0, 0]
        yroot = vals[0, 1]
        zroot = vals[0, 2]
        self.ax.set_xlim3d([xroot - r, xroot + r])
        self.ax.set_ylim3d([yroot - r, yroot + r])
        self.ax.set_zlim3d([zroot - r, zroot + r])
        self.ax.set_aspect('auto')


def visualize_sequence(sequence):
    """
    Visualize a sequence of poses in 3D.

    Args:
        sequence: Numpy array of shape (n_frames, 15, 3)
    """
    sequence = np.asarray(sequence)
    nframes = sequence.shape[0]

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')

    visualizer = Ax3DPose(ax)

    for i in range(nframes):
        visualizer.update(centeredscaled(sequence[i, :]))
        plt.show(block=False)
        fig.canvas.draw()
        plt.pause(0.8)

    plt.close()


def viz_tsne(opt):
    """
    Perform t-SNE embedding on pose sequences and visualize the result.

    Args:
        opt: Parsed configuration object with attributes:
             - dataroot: Path to the dataset folder
             - seq_length: Target sequence length for re-sampling
    """
    print("[INFO] Loading sequences and labels...")
    sequences = sio.loadmat(f"{opt.dataroot}/sequences")['sequences'].squeeze()
    labels = sio.loadmat(f"{opt.dataroot}/labels")['labels'].squeeze()

    print("[INFO] Re-sampling sequences...")
    s = np.linspace(0, 1, opt.seq_length)
    res_sequences = [resampling(seq, s)[0] for seq in sequences]

    print("[INFO] Formatting data for t-SNE...")
    x, y = tsne_data_format_exp(res_sequences, labels, opt)

    print("[INFO] Running t-SNE...")
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    Y = tsne.fit_transform(x)

    print("[INFO] Plotting t-SNE results...")
    pylab.scatter(Y[:, 0], Y[:, 1], 20, y)
    pylab.title("t-SNE Embedding of Pose Sequences")
    pylab.xlabel("Component 1")
    pylab.ylabel("Component 2")
    pylab.show()
