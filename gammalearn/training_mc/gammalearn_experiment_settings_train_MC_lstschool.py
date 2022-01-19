import collections
import os
import importlib
from pathlib import Path
import math
import numpy as np
import torch
from torch.optim import lr_scheduler

from torchmetrics.classification import Accuracy, AUROC
from pytorch_lightning.profiler import SimpleProfiler, AdvancedProfiler, PyTorchProfiler

import gammalearn.criterions as criterions
import gammalearn.optimizers as optimizers
import gammalearn.steps as steps
from gammalearn.callbacks import (LogGradientNorm, LogModelWeightNorm, LogModelParameters,
                                  LogUncertaintyLogVars, LogGradNormWeights, LogReLUActivations,
                                  LogLinearGradient, LogFeatures, WriteDL2Files)
import gammalearn.utils as utils
import gammalearn.datasets as dsets
from gammalearn.constants import GAMMA_ID, PROTON_ID, ELECTRON_ID
from gammalearn.metrics import AUCMultiClass


# Experiment settings
main_directory = Path(__file__).parent.joinpath('../../data/gammalearn/experiments/').absolute().as_posix()
"""str: mandatory, where the experiments are stored"""
experiment_name = '20220121_lstschool_training_mc'
"""str: mandatory, the name of the experiment. Should be different
for each experiment, except if one wants to resume an old experiment
"""
info = 'Training on MC example for LST analysis school'
"""str: optional"""
gpus = 0
"""int or list: mandatory, the umber of gpus to use. If -1, run on all GPUS, 
if None/0 run on CPU. If list, run on GPUS of list.
"""
log_every_n_steps = 5
"""int: optional, the interval in term of iterations for on screen
data printing during experiment
"""
window_size = 100
"""int: optional, the interval in term of stored values for metric moving computation"""
save_every = 1
"""int: optional, the interval in term of epochs for saving the model parameters.
If save_every < 1, the model is not saved.
If not provided, the model is not saved.
"""
random_seed = 1
"""int: optional, the manual seed to make experiments more reproducible"""
monitor_gpus = True
"""bool: optional, whether or not monitoring the gpu utilization"""

dataset_class = dsets.MemoryLSTDataset
# dataset_class = dsets.FileLSTDataset
"""Dataset: mandatory, the Dataset class to load the data. Currently 2 classes are available, MemoryLSTDataset that 
loads images in memory, and FileLSTDataset that loads images from files during training.
"""
dataset_parameters = {'camera_type': 'LST_LSTCam',
                      'group_by': 'image',
                      'use_time': True,
                      'particle_dict': {GAMMA_ID: 0,
                                        PROTON_ID: 1,
                                        # ELECTRON_ID: 2,
                                        },
                      # 'subarray': [1],
                      }
"""dict: mandatory, the parameters of the dataset.
camera_type is mandatory and can be:
'LST_LSTCam', 'MST_NectarCam', 'MST_FlashCam', 'SST_ASTRICam', 'SST1M_DigiCam', 'SST_CHEC', 'MST-SCT_SCTCam'.
group_by is mandatory and can be 'image', 'event_all_tels', 'event_triggered_tels'.
particle_dict is mandatory and maps cta particle types with class id. e.g. gamma (0) is class 0, 
proton (101) is class 1 and electron (1) is class 2.
use_time (optional): whether or not to use time information
subarray (optional): the list of telescope ids to select as a subarray
"""
data_transform = {'data': [dsets.NumpyToTensor(),
                           ],
                  'target': [
                      dsets.NumpyToTensor()
                  ],
                  'telescope': [
                      dsets.NumpyToTensor()
                  ]
                  }
"""dict: optional, dictionary of transform functions to apply to the samples of data from the Dataset"""
preprocessing_workers = 4
"""int: optional, the max number of workers to create dataset."""
dataloader_workers = 4
"""int: optional, the max number of workers for the data loaders. If 0, data are loaded from the main thread."""
mp_start_method = 'fork'
"""str: optional, the method to start new process in [fork, spawn]"""
targets = collections.OrderedDict({
    'energy': {
        'output_shape': 1,
        'loss': torch.nn.L1Loss(reduction='none'),
        'loss_weight': 1,
        'metrics': {
            # 'functions': ,
        }
    },
    'impact': {
        'output_shape': 2,
        'loss': torch.nn.L1Loss(reduction='none'),
        'loss_weight': 1,
        'metrics': {}
    },
    'direction': {
        'output_shape': 2,
        'loss': torch.nn.L1Loss(reduction='none'),
        'loss_weight': 1,
        'metrics': {}
    },
    'class': {
        'label_shape': 1,
        'loss': criterions.nll_nn,
        'loss_weight': 1,
        'metrics': {
            'Accuracy': Accuracy(threshold=0.5),
            'AUC': AUCMultiClass(buffer_size=window_size,
                                 compute_on_step=True),
            # 'AUC': AUROC(pos_label=dataset_parameters['particle_dict'][GAMMA_ID],
            #              num_classes=len(dataset_parameters['particle_dict']),
            #              compute_on_step=True
            #              )
        }
    },
})
"""dict: mandatory, defines for every objectives of the experiment
the loss function and its weight
"""

# Net settings
net_definition_file = utils.nets_definition_path()
"""str: mandatory, the file where to find the net definition to use"""
# Load the network definitions module #
spec = importlib.util.spec_from_file_location("nets", net_definition_file)
nets = importlib.util.module_from_spec(spec)
spec.loader.exec_module(nets)
########################################
model_net = nets.GammaPhysNet
"""nn.Module: mandatory, the network for the experiment. Is a class
inheriting from nn.Module defining the operations of the network and
the forward method to pass data through it
"""
net_parameters_dic = {
    "num_layers": 3,
    "init": "kaiming",
    "batch_norm": True,
    # "init": "orthogonal",
    "num_channels": 2,
    "block_features": [16, 32, 64],
    "attention_layer": (nets.DualAttention, {"ratio": 16}),
    # "attention_layer": (nets.SqueezeExcite, {"ratio": 4}),
    # "attention_layer": None,
    "fc_width": 256,
    "non_linearity": torch.nn.ReLU,
    "last_bias_init": None,
}
"""dict: mandatory, the parameters of the network. Depends on the
network chosen
"""
# checkpoint_path = main_directory + '/test_install/checkpoint_epoch=3.ckpt'
"""str: optional, the path where to find the backup of the model to resume"""

profiler = None
# profiler = {'profiler': SimpleProfiler,
#             'options': dict(extended=True)
#             }
"""str: optional, the profiler to use"""

######################################################################################################################
train = True
"""bool: mandatory, whether or not to train the model"""
# Data settings
train_folders = [
    '../../data/mc/DL1/proton/training/',
    '../../data/mc/DL1/gamma-diffuse/training/',
]  # TODO fill your folder path
"""list: mandatory, the folders where to find the hdf5 data files"""

validating_ratio = 0.2
"""float: mandatory, the ratio of data to create the validating set"""
split_by_file = False
"""bool: optional, whether to split data at the file level or at the data level"""
max_epochs = 4
"""int: mandatory, the maximum number of epochs for the experiment"""
batch_size = 8
"""int: mandatory, the size of the mini-batch"""
image_filter = {
    # utils.intensity_filter: {'intensity': [50, np.inf]},
    # utils.cleaning_filter: {'picture_thresh': 6, 'boundary_thresh': 3,
    #                         'keep_isolated_pixels': False, 'min_number_picture_neighbors': 2},
    # utils.leakage_filter: {'leakage2_cut': 0.2, 'picture_thresh': 6, 'boundary_thresh': 3,
    #                        'keep_isolated_pixels': False, 'min_number_picture_neighbors': 2},
}
"""dict: optional, the filter(s) to apply to the dataset at image level"""
event_filter = {
    # utils.energyband_filter: {'energy': [0.02, 2], 'filter_only_gammas': True},  # in TeV
    # utils.emission_cone_filter: {'max_angle': 0.0698},
    # utils.impact_distance_filter: {'max_distance': 200},
    # utils.telescope_multiplicity_filter: {'multiplicity': 2},
}
"""dict: optional, the filter(s) to apply to the dataset"""

# data_augment = {'function': dsets.augment_via_rotation,
#                 'kwargs': {'thetas': [2 * math.pi / 3, 4 * math.pi / 3],
#                            'num_workers': 8}
#                 }
"""dict: optional, dictionary describing the function to use for dataset augmentation"""
# dataset_size = 2000
"""int: optional, the max size of the dataset"""
# files_max_number = 1
"""int: optional, the max number of files to use for the dataset"""

pin_memory = True
"""bool: optional, whether or not to pin memory in dataloader"""


# Training settings
loss_options = {
    'conditional': True,
    'gamma_class': dataset_parameters['particle_dict'][0],
    'logvar_coeff': [2, 2, 2, 0.5],  # for uncertainty
    'penalty': 0,  # for uncertainty
}
"""dict: mandatory, defines for every objectives of the experiment
the loss function and its weight
"""
compute_loss = criterions.MultilossBalancing(targets, **loss_options)
"""function: mandatory, the function to compute the loss"""
optimizer_dic = {
    # 'network': optimizers.load_sgd,
    'network': optimizers.load_adam,
    'loss_balancing': optimizers.load_adam
}
"""dict: mandatory, the optimizers to use for the experiment.
One may want to use several optimizers in case of GAN for example
"""
optimizer_parameters = {
    'network': {'learning_rate': 1e-2,
                'weight_decay': 1e-7,
                # 'momentum': 0.9,
                # 'nesterov': True
                },
    'loss_balancing': {'learning_rate': 0.025,
                       'weight_decay': 1e-4,
                       },
}
"""dict: mandatory, defines the parameters for every optimizers to use"""
# regularization = {'function': 'gradient_penalty',
#                   'weight': 10}
"""dict: optional, regularization to use during the training process. See in optimizers.py for 
available regularization functions. If `function` is set to 'gradient_penalty', the training step must be 
`training_step_mt_gradient_penalty`."""
training_step = steps.training_step_mt
# training_step = steps.training_step_gradnorm
# training_step = steps.training_step_mt_gradient_penalty
"""function: mandatory, the function to compute the training step"""
eval_step = steps.eval_step_mt
"""function: mandatory, the function to compute the validating step"""
check_val_every_n_epoch = 1
"""int: optional, the interval in term of epoch for validating the model"""
lr_schedulers = {
    # lr_scheduler.StepLR: {'network': {'gamma': 0.1,
    #                                   'step_size': 2,
    #                                   },
    #                       },
    lr_scheduler.ReduceLROnPlateau: {'network': {'factor': 0.1,
                                                 'patience': 2,
                                                 },
                                     },
    # lr_scheduler.MultiStepLR: {'network': {'gamma': 0.1,
    #                                        'milestones': [10, 15, 18],
    #                                        },
    #                            },
    # lr_scheduler.ExponentialLR: {'network': {'gamma': 0.9,
    #                                          },
    #                              },
}
"""dict: optional, defines the learning rate schedulers"""
# callbacks
training_callbacks = [
    LogGradientNorm(),
    LogModelWeightNorm(),
    LogModelParameters(),
    LogUncertaintyLogVars(),
    # LogGradNormWeights(),
    LogReLUActivations(),
    LogLinearGradient(),
    # LogFeatures(),  # Do not use during training !! Very costly !!
]
"""dict: list of callbacks
"""

######################################################################################################################
# Testing settings
test = True
"""bool: mandatory, whether or not to test the model at the end of training"""
test_step = steps.test_step_mt
"""function: mandatory, the function to compute the validating step"""
test_folders = [
    '../data/mc/DL1/gamma/testing/',
    '../data/mc/DL1/proton/testing/',
    '../data/mc/DL1/electron/testing/',
]
"""list of str: optional, the folders containing the hdf5 data files for the test
"""
dl2_path = ''
"""str: optional, path to store dl2 files"""
test_dataset_parameters = {
    # 'subarray': [1],
}
"""dict: optional, the parameters of the dataset specific to the test operation.
"""
test_image_filter = {
    utils.intensity_filter: {'intensity': [10, np.inf]},
    # # utils.cleaning_filter: {'picture_thresh': 6, 'boundary_thresh': 3,
    # #                         'keep_isolated_pixels': False, 'min_number_picture_neighbors': 2},
    # utils.leakage_filter: {'leakage2_cut': 0.2, 'picture_thresh': 6, 'boundary_thresh': 3,
    #                        'keep_isolated_pixels': False, 'min_number_picture_neighbors': 2},
}
"""dict: optional, filter(s) to apply to the test set at image level"""
test_event_filter = {
    # utils.energyband_filter: {'energy': [0.02, 2], 'filter_only_gammas': True},  # in TeV
    # utils.emission_cone_filter: {'max_angle': 0.0698},
    # utils.impact_distance_filter: {'max_distance': 200},
    # utils.telescope_multiplicity_filter: {'multiplicity': 2},
}
"""dict: optional, filter(s) to apply to the test set"""
# test_file_max_number = 1
"""int: optional, the max number of files to use for the dataset"""
test_batch_size = 10
"""int: optional, the size of the mini-batch for the test"""
test_callbacks = [
    WriteDL2Files()
]
"""dict: list of callbacks
"""
