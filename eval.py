import argparse
import os

import crossView

import numpy as np

import torch
from torch.utils.data import DataLoader

import cv2
import tqdm

from utils import mean_IU, mean_precision
from utils import both_mean_IU, both_mean_precision
from opt import get_eval_args as get_args

from PIL import Image
import matplotlib.pyplot as PLT
import matplotlib.cm as mpl_color_map

from skimage import color
from skimage.io._plugins.pil_plugin import ndarray_to_pil
import torchvision.transforms.functional as F
def gen_lut():
  """
  Generate a label colormap compatible with opencv lookup table, based on
  Rick Szelski algorithm in `Computer Vision: Algorithms and Applications`,
  appendix C2 `Pseudocolor Generation`.
  :Returns:0
    color_lut : opencv compatible color lookup table
  """
  tobits = lambda x, o: np.array(list(np.binary_repr(x, 24)[o::-3]), np.uint8)
  arr = np.arange(256)
  r = np.concatenate([np.packbits(tobits(x, -3)) for x in arr])
  g = np.concatenate([np.packbits(tobits(x, -2)) for x in arr])
  b = np.concatenate([np.packbits(tobits(x, -1)) for x in arr])
  return np.concatenate([[[b]], [[g]], [[r]]]).T

def labels2rgb(labels, lut, type):
  """
  Convert a label image to an rgb image using a lookup table
  :Parameters:
    labels : an image of type np.uint8 2D array
    lut : a lookup table of shape (256, 3) and type np.uint8
  :Returns:
    colorized_labels : a colorized label image
  """
  if type == "both":
    lut[0] =np.array([[204,   102,   0]], dtype=np.uint8)
    lut[1] =np.array([[235,   206,   135]], dtype=np.uint8)
    lut[2] =np.array([[0,   0,   0]], dtype=np.uint8)
  elif type == "dynamic":
    lut[0] =np.array([[0,   0,   0]], dtype=np.uint8)
    lut[1] =np.array([[204,   102,   0]], dtype=np.uint8)
  else:
    lut[0] =np.array([[0,   0,   0]], dtype=np.uint8)
    lut[1] =np.array([[235,   206,   135]], dtype=np.uint8)
 
  return cv2.LUT(cv2.merge((labels, labels, labels)), lut)


def readlines(filename):
    """Read all the lines in a text file and return as a list
    """
    with open(filename, 'r') as f:
        lines = f.read().splitlines()
    return lines


def load_model(models, model_path):
    """Load model(s) from disk
    """
    model_path = os.path.expanduser(model_path)

    assert os.path.isdir(model_path), \
        "Cannot find folder {}".format(model_path)
    print("loading model from folder {}".format(model_path))

    for key in models.keys():
        print("Loading {} weights...".format(key))
        path = os.path.join(model_path, "{}.pth".format(key))
        model_dict = models[key].state_dict()
        pretrained_dict = torch.load(path, map_location='cuda:0')
        pretrained_dict = {
            k: v for k,
                     v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        models[key].load_state_dict(model_dict)
    return models


def evaluate():
    opt = get_args()

    # Loading Pretarined Model
    models = {}
    models["encoder"] = crossView.Encoder(18, opt.height, opt.width, True)
    models["label_encoder"] = crossView.Encoder(18, opt.height, opt.width, True)
    models['CycledViewProjection'] = crossView.CycledViewProjection(in_dim=8)
    models["CrossViewTransformer"] = crossView.CrossViewTransformer(128)
    
    models["decoder"] = crossView.Decoder(
        models["encoder"].resnet_encoder.num_ch_enc, opt.num_class)
    models["transform_decoder"] = crossView.Decoder(
        models["encoder"].resnet_encoder.num_ch_enc, opt.num_class, "transform_decoder")

    for key in models.keys():
        models[key].to("cuda")

    models = load_model(models, opt.pretrained_path)

    # Loading Validation/Testing Dataset

    # Data Loaders
 
    dataset_dict = {"3Dobject": crossView.KITTIObject,
                    "odometry": crossView.KITTIOdometry,
                    "argo": crossView.Argoverse,
                    "raw": crossView.KITTIRAW}

    dataset = dataset_dict[opt.split]
    fpath = os.path.join(
        os.path.dirname(__file__),
        "splits",
        opt.split,
        "{}_files.txt")
    test_filenames = readlines(fpath.format("val"))
    test_dataset = dataset(opt, test_filenames, is_train=False)
    test_loader = DataLoader(
        test_dataset,
        1,
        False,
        num_workers=opt.num_workers,
        pin_memory=True,
        drop_last=True)

    iou, mAP = np.array([0., 0.]), np.array([0., 0.])
    if opt.type =="both":
        iou, mAP = np.array([0., 0., 0.]), np.array([0., 0., 0.])
    for batch_idx, inputs in tqdm.tqdm(enumerate(test_loader)):
        with torch.no_grad():
            outputs = process_batch(opt, models, inputs)
        save_topview(
                outputs["topview"],
                os.path.join(opt.out_dir,opt.split,opt.type,"{}.png".format(inputs["filename"][0])),opt.type)
        pred = np.squeeze(torch.argmax(
                outputs["topview"].detach(),
                1).cpu().numpy())

        true = np.squeeze(inputs[opt.type + "_gt"].detach().cpu().numpy())
        if opt.type =="both":
            iou += both_mean_IU(pred, true)
            mAP += both_mean_precision(pred, true)
        else:
            iou += mean_IU(pred, true)
            mAP += mean_precision(pred, true)
    iou /= len(test_loader)
    mAP /= len(test_loader)
    if opt.type =="both":
        print("Evaluation Results: mIOU: %.4f, %.4f, %.4f, mAP: %.4f, %.4f, %.4f" % (iou[0], iou[1], iou[2], mAP[0], mAP[1], mAP[2]))
    else:
        print("Evaluation Results: mIOU: %.4f, mAP: %.4f" % (iou[1], mAP[1]))


def process_batch(opt, models, inputs):
    outputs = {}
    # print(inputs["filename"])
    for key, input_ in inputs.items():

        if key != "filename":
            inputs[key] = input_.to("cuda")

    features = models["encoder"](inputs["color"])

    label = inputs[opt.type+"_gt"]

    label = torch.stack([label,label,label],dim=1)
    label = F.resize(label, opt.height).float()
    label_features = models["label_encoder"](label) #[6,128,8,8]

    # Cross-view Transformation Module
    x_feature = features
    transform_feature, retransform_features, label_transform_features, label_retransform_features = models["CycledViewProjection"](features, label_features)
    features = models["CrossViewTransformer"](features, transform_feature, retransform_features)

    outputs["topview"] = models["decoder"](features)
    outputs["transform_topview"] = models["transform_decoder"](transform_feature) 
    return outputs


def save_topview(tv, name_dest_im, type):
    dir_name = os.path.dirname(name_dest_im)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    tv_np = tv.squeeze().cpu().numpy()
    if type == "both":
        true_top_view = np.argmax(tv_np, axis=0)
    else:
        true_top_view = np.zeros((tv_np.shape[1], tv_np.shape[2]))
        true_top_view[tv_np[1] > tv_np[0]] = 1.
    
    lut = gen_lut()
    rgb = labels2rgb(true_top_view.astype(np.uint8), lut, type)
    cv2.imwrite(name_dest_im, rgb)    
    
if __name__ == "__main__":
    evaluate()
