import math
import numpy as np
from .procrustes import procrustes
from .weighted_karcher import weighted_karcher


def centered_scaled(X):
    """
    Removes translation and scale variabilities
    :param X: matrix of 2-D or 3-D landmarks
    :return:
      X0: normalized matrix of 2-D or 3-D landmarks
    """
    muX = X.mean(0)
    X0 = X - muX
    ssX = (X0 ** 2.).sum()
    # centered Frobenius norm
    normX = np.sqrt(ssX)
    # scale to equal (unit) norm
    X0 = X0 / normX
    return X0


def geodesic_distance(X, Y):
    """
    Compute geodesic distance between two points X and Y
    :param X: matrix of 2-D or 3-D landmarks
    :param Y: matrix of 2-D or 3-D landmarks
    :return:
      distance: distance's value
    """

    X0 = centered_scaled(X)
    Y0 = centered_scaled(Y)
    d, Y0, T, b, c = procrustes(X0, Y0, compute_optimal_scale=True)
    distance = float(math.acos(np.matrix.trace(np.matmul(X0,Y0.T))))

    return distance


def linear_reconstruction(generated_samples, dictionary, opt):
    """
    Convert sparse codes to 3-D cordinates x, y and z
    :param sequences: (n_seq, n_sc, n_frames) matrix with:
        n_seq : number of sequences
        n_sc : dimension of space codes
        n_frames : number of frames
    :param dictionary: (n, m, d) matrix with
        n : number of atoms of the dictionary
        m : number of joint points
        d : dimension of the joint points (2-D or 3-D)
    :return:
      recons_sequence: matrix of skeletons
    """

    codes = []
    transformations = []
    for i in range(len(generated_samples)):
        codes.append(generated_samples[i][:opt.feature_dim, :])
        transformations.append(generated_samples[i][opt.feature_dim:opt.feature_dim+opt.transformation_dim, :])

    recons_sequence = []
    nb_frames = codes[0].shape[1]
    nb_landmarks = dictionary.shape[1]
    landmark_dim = dictionary.shape[2]

    for c in range(len(codes)):
        code = codes[c]
        transf = transformations[c]
        if code.shape[0] != nb_frames:
            code = code.transpose()

        sum = np.zeros((nb_frames, nb_landmarks, landmark_dim))

        for i in np.arange(len(code)):
            t = 0
            for j in np.arange(len(dictionary)):
                t += (code[i, j] * dictionary[j])
                sum[i, :, :] += (code[i, j] * dictionary[j])

            # Denormalize
            sum[i, :, :] = sum[i, :, :] * transf[2, i]
            for l in range(nb_landmarks):
                sum[i, l, 0] = sum[i, l, 0] + transf[0, i]
                sum[i, l, 1] = sum[i, l, 1] + transf[1, i]

        recons_sequence.append(sum)

    return np.array(recons_sequence)


def non_linear_reconstruction(codes, dictionary, opt):
    """
    We use the weighted karcher mean algorithm to convert sparse codes to 3-D cordinates x, y and z
    :param sequences: (n_seq, n_sc, n_frames) matrix with:
        n_seq : number of sequences
        n_sc : dimension of space codes
        n_frames : number of frames
    :param dictionary: (n, m, d) matrix with
        n : number of atoms of the dictionary
        m : number of joint points
        d : dimension of the joint points (2-D or 3-D)
    :return:
      reconstructed_sequence: matrix of skeletons
      """

    nb_sequences = len(codes)
    nb_frames = codes[0].shape[0]
    nb_joints = dictionary[0].shape[0]
    joint_dim = dictionary[0].shape[1]
    reconstructed_sequences = np.zeros((nb_sequences, nb_frames, nb_joints, joint_dim))

    for i in range(nb_sequences):
        print(i)
        for j in range(nb_frames):
            print(j)
            reconstructed_sequences[i, j, :, :] = weighted_karcher(codes[i][j], dictionary)
            #print('Reconstruction time for sequence {0} - frame {1} : {2} seconds'.format(str(i), str(j), str(end-start)))
    return reconstructed_sequences


def reconstruction(codes, dictionary, opt, linear=True):
    """
    Choose the reconstruction function to use
    :param sequences: (n_seq, n_sc, n_frames) matrix with:
        n_seq : number of sequences
        n_sc : dimension of space codes
        n_frames : number of frames
    :param dictionary: (n, m, d) matrix with
        n : number of atoms of the dictionary
        m : number of joint points
        d : dimension of the joint points (2-D or 3-D)
    :param linear: whether to use linear reconstruction or non linear reconstruction
    :return:
      reconstructed_sequence: matrix of skeletons
    """

    if linear:
        reconstructed_sequence = linear_reconstruction(codes, dictionary, opt)
    elif not linear:
        reconstructed_sequence = non_linear_reconstruction(codes, dictionary, opt)
    else:
        raise(ValueError, "Unrecognized linear value: %d" % linear)
    return reconstructed_sequence


def tsne_data_format(codes, data, opt):

    x = np.zeros((len(codes) + len(data), (opt.feature_dim + opt.transformation_dim)*opt.seq_length))  # Data-points
    # y = labels  # Labels
    y = np.zeros((len(codes) + len(data)))  # Labels
    for i in range(len(codes)):
        x[i] = codes[i].reshape(1, (opt.feature_dim + opt.transformation_dim)*opt.seq_length)
        y[i] = 1  # Fake
    for i in range(len(data)):
        x[i + len(codes)] = data[i, :, :].reshape(1, (opt.feature_dim + opt.transformation_dim) * opt.seq_length)
        y[i + len(codes)] = 2  # Real

    return x, y


def tsne_data_format_exp(codes, labels, opt):

    dim = np.asarray(codes[0]).shape[0] * np.asarray(codes[0]).shape[1] * np.asarray(codes[0]).shape[2]
    x = np.zeros((len(codes) * 2, dim+1))  # Data-points
    y = np.zeros((len(codes) * 2))  # Labels
    for i in range(len(codes)):
        x[i, :dim] = codes[i].reshape(1, dim)
        x[i, dim] = 0
        y[i] = 0
        x[i + len(codes), :dim] = codes[i].reshape(1, dim)
        x[i + len(codes), dim] = labels[i] #np.true_divide(np.asarray(labels[i]), opt.n_classes)  # Real
        y[i + len(codes)] = 1



    return x, y
