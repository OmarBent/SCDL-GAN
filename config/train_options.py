from config.base_options import BaseOptions


class TrainOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)

        # SCDL config
        self.parser.add_argument('--feature_dim', type=int, default=20, help='dimension of the encoder latent space - size of the dictionary')
        self.parser.add_argument('--transformation_dim', type=int, default=4, help='dimension of the filtered transformations')
        self.parser.add_argument("--step", type=int, default=3, help="step size when selecting training frames from each sequence - for k-means")
        self.parser.add_argument("--sparsity", type=int, default=0.0003, help="Sparsity regularization parameter")
        self.parser.add_argument("--k", type=int, default=20, help="size of dictionary") #TODO k and feature are the same
        self.parser.add_argument("--per_class", type=int, default=0, help="0: global dictionary - 1: class-specific dictionary")
        self.parser.add_argument("--karcher", type=int, default=1, help="1: kmeans with Karcher mean computations 0: linear mean computations")

        # Pose-GAN config
        self.parser.add_argument('--arch', nargs='?', type=str, default='IWCGAN',help='Architecture to use [\'IWGAN, IWCGAN \']')
        self.parser.add_argument("--n_epochs", type=int, default=1000, help="number of epochs of GAN training")
        self.parser.add_argument("--PG_batch_size", type=int, default=32, help="size of the batches")
        self.parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
        self.parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
        self.parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
        self.parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
        self.parser.add_argument("--n_critic", type=int, default=5, help="number of training steps for discriminator per iter")
        self.parser.add_argument("--clip_value", type=float, default=0.01, help="lower and upper clip value for disc. weights")
        self.parser.add_argument("--C_SCALE", type=int, default=1, help="How to scale critic's loss relative to WGAN loss")
        self.parser.add_argument("--G_SCALE", type=int, default=1, help="How to scale generator's loss relative to WGAN loss")
        self.parser.add_argument("--checkpoint", type=str, default='../checkpoint/', help="How to scale generator's loss relative to WGAN loss")
        self.parser.add_argument("--restore_mode", type=str, default=False, help="if True, it will load saved model from --checkpoint and continue to train")

        self.parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        self.parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')

        self.isTrain = True

        # LSTM classification config
        self.parser.add_argument("--train_subjects", type=list, default=[3, 4, 5, 6, 10], help="training subjects")
        self.parser.add_argument("--b_size", type=int, default=16, help="batch size")
        self.parser.add_argument("--lstm_size", type=int, default=64, help="neuron size")
        self.parser.add_argument("--dropout_prob", type=float, default=0.03, help="probability of dropout")
        self.parser.add_argument("--nb_epochs", type=int, default=100, help="number of epochs for LSTM")

