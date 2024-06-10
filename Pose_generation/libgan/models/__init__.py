#from libgan.models.IWCGAN import *
from .iwgan import *
from .con_iwgan import *


def get_model(opt, version=None):
    #model = _get_model_instance(name)

    if opt.arch in ['IWGAN']:
        model = iwgan(opt.latent_dim, opt.feature_dim, opt.transformation_dim, opt.seq_length)
    elif opt.arch in ['IWCGAN']:
        model = con_iwgan(opt.n_classes, opt.latent_dim, opt.feature_dim, opt.transformation_dim, opt.seq_length)
    return model


def _get_model_instance(name):
    try:
        return {
            # 'IWGAN': IWGAN
            # 'IWCGAN':IWCGAN
        }[name]
    except:
        print('Model {} not available'.format(name))
