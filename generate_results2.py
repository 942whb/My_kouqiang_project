import os, json, sys
import os.path as osp
import argparse
import warnings
from tqdm import tqdm
import numpy
import time
from utils.evaluation import evaluate
import numpy as np
from skimage.io import imsave
from skimage.util import img_as_ubyte
from skimage.transform import resize
import torch
from utils.model_saving_loading import str2bool
from models.get_model import get_arch
from utils.get_loaders import get_test_datasets
from utils.model_saving_loading import load_model
from sobel import edge_conv2d

# argument parsing
parser = argparse.ArgumentParser()
required_named = parser.add_argument_group('required arguments')
required_named.add_argument('--dataset', type=str, help='generate results for which dataset', required=True)
parser.add_argument('--public', type=str2bool, nargs='?', const=True, default=True, help='public or private data')
parser.add_argument('--experiment_path', help='experiments/subfolder where checkpoint is', default=None)
parser.add_argument('--tta', type=str, default='from_preds', help='test-time augmentation (no/from_logits/from_preds)')
parser.add_argument('--binarize', type=str, default='otsu', help='binarization scheme (\'otsu\')')
parser.add_argument('--config_file', type=str, default=None, help='experiments/name_of_config_file, overrides everything')
# im_size overrides config file
parser.add_argument('--im_size', help='delimited list input, could be 600,400', type=str, default='512')
parser.add_argument('--device', type=str, default='cpu', help='where to run the training code (e.g. "cpu" or "cuda:0") [default: %(default)s]')
parser.add_argument('--result_path', type=str, default='results', help='path to save predictions (defaults to results')
parser.add_argument('--csv_path', type=str, default='data/DRIVE/train.csv', help='path to training data csv')

def flip_ud(tens):
    return torch.flip(tens, dims=[1])

def flip_lr(tens):
    return torch.flip(tens, dims=[2])

def flip_lrud(tens):
    return torch.flip(tens, dims=[1, 2])

def create_pred(model, inputs_c1, inputs_c2, label_c1, label_c2, tta='no'):
    act = torch.sigmoid if model.n_classes == 1 else torch.nn.Softmax(dim=0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    inputs_c1, inputs_c2, label_c1, label_c2 = inputs_c1.to(device), inputs_c2.to(device), label_c1.to(device), label_c2.to(device)
    with torch.no_grad():
        logits_c1 = model(inputs_c1)
        logits_c2 = model(inputs_c2)
        
    

    return logits_c1, logits_c2, label_c1, label_c2

def save_pred(full_pred, save_results_path, im_name):
    os.makedirs(save_results_path, exist_ok=True)
    im_name = im_name.rsplit('/', 1)[-1]
    save_name = osp.join(save_results_path, im_name[:-4] + '.png')

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # this casts preds to int, loses precision but meh
        imsave(save_name, img_as_ubyte(full_pred))

if __name__ == '__main__':
    '''
    Example:
    python generate_results.py --config_file experiments/unet_drive/config.cfg --dataset DRIVE
    '''

    args = parser.parse_args()

    if args.device.startswith("cuda"):
        # In case one has multiple devices, we must first set the one
        # we would like to use so pytorch can find it.
        os.environ['CUDA_VISIBLE_DEVICES'] = args.device.split(":",1)[1]
        if not torch.cuda.is_available():
            raise RuntimeError("cuda is not currently available!")
        print(f"* Running prediction on device '{args.device}'...")
        device = torch.device("cuda")
    else:  #cpu
        device = torch.device(args.device)

    dataset = args.dataset
    binarize = args.binarize
    tta = args.tta
    public = str2bool(args.public)

    # parse config file if provided
    config_file = args.config_file
    if config_file is not None:
        if not osp.isfile(config_file): raise Exception('non-existent config file')
        with open(args.config_file, 'r') as f:
            args.__dict__.update(json.load(f))
    experiment_path = args.experiment_path # this should exist in a config file
    model_name = args.model_name

    if experiment_path is None: raise Exception('must specify path to experiment')

    im_size = tuple([int(item) for item in args.im_size.split(',')])
    if isinstance(im_size, tuple) and len(im_size)==1:
        tg_size = (im_size[0], im_size[0])
    elif isinstance(im_size, tuple) and len(im_size)==2:
        tg_size = (im_size[0], im_size[1])
    else:
        sys.exit('im_size should be a number or a tuple of two numbers')

    if public: data_path = osp.join('data', dataset)
    else: data_path = osp.join('private_data', dataset)

    csv_path = args.csv_path
    print('* Reading test data from ' + osp.join(data_path, csv_path))
    test_dataset = get_test_datasets(data_path, csv_path=csv_path, tg_size=tg_size)
    print('* Instantiating model  = ' + str(model_name))
    model = get_arch(model_name).to(device)
    if model_name == 'wnet': model.mode='eval'

    print('* Loading trained weights from ' + experiment_path)
    try:
        model, stats = load_model(model, experiment_path, device)
    except RuntimeError:
        sys.exit('---- bad config specification (check layers, n_classes, etc.) ---- ')
    model.eval()

    save_results_path = osp.join(args.result_path, dataset, experiment_path)
    print('* Saving predictions to ' + save_results_path)
    times = []
    logitsc1_all, logitsc2_all, labelc1_all, labelc2_all = [], [], [], []
    for i in tqdm(range(len(test_dataset))):
        inputs_c1, inputs_c2, label_c1, label_c2 = test_dataset[i]
        inputs_c1 = inputs_c1.unsqueeze(0)
        inputs_c2 = inputs_c2.unsqueeze(0)
        label_c1 = label_c1.unsqueeze(0)
        label_c2 = label_c2.unsqueeze(0)
        start_time = time.perf_counter()
        logits_c1, logits_c2, label_c1, label_c2 = create_pred(model, inputs_c1, inputs_c2, label_c1, label_c2, tta=tta)
        times.append(time.perf_counter() - start_time)
        logitsc1_all.extend(logits_c1)
        logitsc2_all.extend(logits_c2)
        labelc1_all.extend(label_c1)
        labelc2_all.extend(label_c2)

        #save_pred(full_pred, save_results_path, im_name)
    logits_all = logitsc1_all+logitsc2_all
    labels_all = labelc1_all+labelc2_all
    acc, dice = evaluate(logits_all, labels_all, 1)
    print(f"* Average image time: {numpy.mean(times):g}s")
    print('ACC:',acc)
