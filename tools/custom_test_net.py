# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Set up custom environment before nearly anything else is imported
# NOTE: this should be the first import (no not reorder)
from maskrcnn_benchmark.utils.env import setup_environment  # noqa F401 isort:skip

import argparse
import os

import torch
from maskrcnn_benchmark.config import cfg
# from maskrcnn_benchmark.data import make_data_loader
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.utils.collect_env import collect_env_info
from maskrcnn_benchmark.utils.comm import synchronize, get_rank
from maskrcnn_benchmark.utils.logger import setup_logger
from maskrcnn_benchmark.utils.miscellaneous import mkdir

import json
from flask import Flask, request
from maskrcnn_benchmark.engine.MyInference import inference
from maskrcnn_benchmark.data.build import my_make_data_loader


def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Inference")
    parser.add_argument(
        "--config-file",
        default="/home/qinjianbo/SRC/maskrcnn-benchmark/configs/e2e_faster_rcnn_R_50_FPN_1x.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    distributed = num_gpus > 1

    if distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
        synchronize()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    save_dir = ""
    logger = setup_logger("maskrcnn_benchmark", save_dir, get_rank())
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(cfg)

    logger.info("Collecting env info (might take some time)")
    logger.info("\n" + collect_env_info())

    model = build_detection_model(cfg)
    model.to(cfg.MODEL.DEVICE)

    output_dir = cfg.OUTPUT_DIR
    checkpointer = DetectronCheckpointer(cfg, model, save_dir=output_dir)
    _ = checkpointer.load(cfg.MODEL.WEIGHT)

    output_folders = [None] * len(cfg.DATASETS.TEST)
    '''
    dataset_names = cfg.DATASETS.TEST
    if cfg.OUTPUT_DIR:
        for idx, dataset_name in enumerate(dataset_names):
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
            mkdir(output_folder)
            output_folders[idx] = output_folder
    '''
    # data_loaders_inference = make_data_loader(cfg, is_train=False, is_distributed=distributed)
    return model, cfg, distributed


def inference_server(model, cfg, dataset_name, distributed, image_path):
    data_loaders_inference = my_make_data_loader(cfg, is_train=False, distributed=distributed, image_path=image_path)
    result = inference(
        model,
        data_loaders_inference,
        dataset_name=dataset_name,
        box_only=cfg.MODEL.RPN_ONLY,
        device=cfg.MODEL.DEVICE,
    )
    synchronize()

    return result


app = Flask(__name__)


@app.route("/inference", method=['POST'])
def index():
    jsoned = request.json
    dataset_name = jsoned.get("dataset_name")
    image_path = jsoned.get("image_path")
    result = inference_server(model=model, cfg=cfg, dataset_name=dataset_name, distributed=distributed,
                              image_path=image_path)
    return result


model = ''
cfg = ''
distributed = ''

if __name__ == "__main__":
    model, cfg, distributed = main()
    app.run(debug=True, host='localhost', port=8001)
