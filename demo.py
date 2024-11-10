import argparse
import glob
import os
import random

import cv2
import editdistance
import numpy
import torch
from PIL import Image
from torch.backends import cudnn
from torchvision import transforms

from Model import TTR
from Model.trainer import Trainer
from dataset.charMapper import CharMapper
from train_logging import get_logger, over_write_args_from_file
from train_utils import getModel


def getTransforms():
    return transforms.Compose([
        transforms.Resize((32, 128), transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5)
    ])


@torch.no_grad()
def main_worker(args):
    assert args.seed is not None
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    numpy.random.seed(args.seed)
    cudnn.deterministic = True
    cudnn.benchmark = True

    # set logger
    logger_level = "INFO"
    logger = get_logger(args.save_name, None, logger_level)
    logger.warning(f"USE GPU: {args.gpu} for training")
    logger.info(args)

    # get model
    model = TTR({"img_size": [32, 128],
                 "patch_size": [4, 4],
                 "embed_dim": 512,
                 "num_heads": 8,
                 "position_attention_hidden": 64,
                 "mask_ratio": 0.0
                 })
    model.load_state_dict(torch.load(args.ckpt)["model"])
    model.eval()
    model = model.cuda(args.gpu)

    # dataset
    preprocess = getTransforms()
    image_paths = glob.glob(os.path.join(args.input_dir, "*"))
    with open(args.gt_txt, "r") as fp:
        lines = fp.readlines()

    gt = {}
    for l in lines:
        img_name, label = l.strip().split('\t')
        gt[img_name] = label
    mapper = CharMapper(letters=args.dataset["val"]["letters"], maxLength=args.dataset["val"]["maxLength"])

    ned = 0
    corr = 0
    ed = 0
    counter = 0

    for image_path in image_paths:
        image = Image.open(image_path)
        image = preprocess(image)

        label = gt[os.path.basename(image_path)]
        label = mapper.text2label(label)

        image = image.unsqueeze(0).to("cuda:0")
        outputs = model(image)
        preds = torch.max(outputs, dim=2)[1]
        pred_text = mapper.reverseMapper(preds[0])

        dist = editdistance.eval(pred_text, label)
        ed += dist
        ned += dist / max(len(pred_text), len(label))

        if dist == 0:
            corr += 1
        else:
            print(f"{counter} ", "*" * 25, f"{dist == 0}")
            print("Pred : ", pred_text)
            print("  GT : ", label)

            cv2.imshow("False", cv2.imread(image_path))
            cv2.waitKey(0)

        counter += 1

    print("Acc : %.2f" % (corr / counter * 100))
    print("ED : %.4f" % (ed / counter))
    print("NED : %.2f" % ((1 - (ned / counter)) * 100))
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Scene Text Recognition Study!')
    parser.add_argument('--config', type=str, default='./config/mvt_tr_real.yaml')
    parser.add_argument('--ckpt', type=str, default='./experiments/real_train/model_best.pth')
    parser.add_argument('--input-dir', type=str, default='data/TS-TR/test')
    parser.add_argument('--gt-txt', type=str, default='data/TS-TR/test.txt')
    args = parser.parse_args()
    over_write_args_from_file(args, args.config)
    main_worker(args)
