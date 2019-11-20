import copy
import logging
import functools

from ptsemseg.loss.loss import cross_entropy2d
from ptsemseg.loss.loss import cross_entropy1d
from ptsemseg.loss.loss import bootstrapped_cross_entropy2d
from ptsemseg.loss.loss import multi_scale_cross_entropy2d
from ptsemseg.loss.loss import mse
from ptsemseg.loss.loss import masked_mse


logger = logging.getLogger('ptsemseg')

key2loss = {'cross_entropy_2d': cross_entropy2d,
            'cross_entropy_1d': cross_entropy1d,
            'masked_mse': masked_mse,
            'bootstrapped_cross_entropy': bootstrapped_cross_entropy2d,
            'multi_scale_cross_entropy': multi_scale_cross_entropy2d,
            'mse': mse}

def get_loss_function(cfg, task=None):
    if task is None:
        logger.error("No task was specified when setting up loss function.")
        raise ValueError("No task was specified when setting up loss function.")

    if cfg['training']['loss'][task] is None:
        logger.info("Using default cross entropy loss")
        return cross_entropy2d

    else:
        loss_dict = cfg['training']['loss'][task]
        loss_name = loss_dict['name']
        loss_params = {k:v for k,v in loss_dict.items() if k != 'name'}

        if loss_name not in key2loss:
            raise NotImplementedError('Loss {} not implemented'.format(loss_name))

        logger.info('Using {} with {} params'.format(loss_name,
                                                     loss_params))
        return functools.partial(key2loss[loss_name], **loss_params)
