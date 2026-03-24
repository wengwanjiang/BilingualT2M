import os
import sys
import json
import datetime
import logging
import os.path as osp
from typing import Union


import numpy as np
from tqdm.auto import tqdm
from omegaconf import OmegaConf

import torch
from torch.utils.data import DataLoader

from mld.config import parse_args
from mld.data.get_data import get_dataset
from mld.models.modeltype.mld import MLD
from mld.models.modeltype.vae import VAE
from mld.utils.utils import print_table, set_seed, move_batch_to_device

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def get_metric_statistics(values: np.ndarray, replication_times: int) -> tuple:
    mean = np.mean(values, axis=0)
    std = np.std(values, axis=0)
    conf_interval = 1.96 * std / np.sqrt(replication_times)
    return mean, conf_interval


@torch.no_grad()
def test_one_epoch(model: Union[VAE, MLD], dataloader: DataLoader, device: torch.device) -> dict:
    for batch in tqdm(dataloader):
        batch = move_batch_to_device(batch, device)
        model.test_step(batch)
    metrics = model.allsplit_epoch_end()
    return metrics


def main():
    cfg = parse_args()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    set_seed(cfg.SEED_VALUE)

    name_time_str = osp.join(cfg.NAME, datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S"))
    cfg.output_dir = osp.join(cfg.TEST_FOLDER, name_time_str)
    os.makedirs(cfg.output_dir, exist_ok=False)

    steam_handler = logging.StreamHandler(sys.stdout)
    file_handler = logging.FileHandler(osp.join(cfg.output_dir, 'output.log'))
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
                        datefmt="%m/%d/%Y %H:%M:%S",
                        handlers=[steam_handler, file_handler])
    logger = logging.getLogger(__name__)

    OmegaConf.save(cfg, osp.join(cfg.output_dir, 'config.yaml'))

    state_dict = torch.load(cfg.TEST.CHECKPOINTS, map_location="cpu")["state_dict"]
    logger.info("Loading checkpoints from {}".format(cfg.TEST.CHECKPOINTS))

    # Step 1: Check if the checkpoint is VAE-based.
    is_vae = False
    vae_key = 'vae.skel_embedding.weight'
    if vae_key in state_dict:
        is_vae = True
    logger.info(f'Is VAE: {is_vae}')

    is_mld = "denoiser.time_embedding.linear_1.weight" in state_dict
    logger.info(f"Is MLD: {is_mld}")

    target_model_class = MLD if is_mld else VAE

    dataset = get_dataset(cfg)
    test_dataloader = dataset.test_dataloader()
    model = target_model_class(cfg, dataset)
    model.to(device)
    model.eval()
    model.requires_grad_(False)
    logger.info(model.load_state_dict(state_dict))

    all_metrics = {}
    replication_times = cfg.TEST.REPLICATION_TIMES
    max_num_samples = cfg.TEST.get('MAX_NUM_SAMPLES')
    name_list = test_dataloader.dataset.name_list
    # calculate metrics
    for i in range(20):
        if max_num_samples is not None:
            chosen_list = np.random.choice(name_list, max_num_samples, replace=False)
            test_dataloader.dataset.name_list = chosen_list

        metrics_type = ", ".join(cfg.METRIC.TYPE)
        logger.info(f"Evaluating {metrics_type} - Replication {i}")
        metrics = test_one_epoch(model, test_dataloader, device)
        logger.info(metrics)
        # with open('/data/wwj/AAAI26/whs/visualization/BiMD_woCLA_EN2CN.pkl', 'wb') as f:
        #     import pickle
        #     pickle.dump(model._eval_debug_lists, f)
        #     exit(0)
        if "TM2TMetrics" in metrics_type and cfg.TEST.DO_MM_TEST:
            # mm metrics
            logger.info(f"Evaluating MultiModality - Replication {i}")
            dataset.mm_mode(True)
            test_mm_dataloader = dataset.test_dataloader()
            mm_metrics = test_one_epoch(model, test_mm_dataloader, device)
            metrics.update(mm_metrics)
            dataset.mm_mode(False)

        print_table(f"Metrics@Replication-{i}", metrics)
        logger.info(metrics)

        for key, item in metrics.items():
            if key not in all_metrics:
                all_metrics[key] = [item]
            else:
                all_metrics[key] += [item]

    all_metrics_new = dict()
    for key, item in all_metrics.items():
        mean, conf_interval = get_metric_statistics(np.array(item), replication_times)
        all_metrics_new[key + "/mean"] = mean
        all_metrics_new[key + "/conf_interval"] = conf_interval
    print_table(f"Mean Metrics", all_metrics_new)
    all_metrics_new.update(all_metrics)
    # save metrics to file
    metric_file = osp.join(cfg.output_dir, f"metrics.json")
    with open(metric_file, "w", encoding="utf-8") as f:
        json.dump(all_metrics_new, f, indent=4)
    logger.info(f"Testing done, the metrics are saved to {str(metric_file)}")


if __name__ == "__main__":
    main()
