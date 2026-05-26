import os
import math
import json
import argparse
import pickle as pkl

from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader

import _init_paths
from core.engine import Engine
import datasets
import models
from core.utils import AverageMeter
from core.config import config, update_config
from core.eval import eval_predictions, display_results, eval
import models.loss as loss

import torch.multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor

torch.manual_seed(0)
torch.cuda.manual_seed(0)


def parse_args():
    parser = argparse.ArgumentParser(description='Test localization network')

    parser.add_argument('--cfg', required=True, type=str)
    args, rest = parser.parse_known_args()

    update_config(args.cfg)

    parser.add_argument('--gpus', type=str)
    parser.add_argument('--workers', type=int)
    parser.add_argument('--dataDir', type=str)
    parser.add_argument('--modelDir', type=str)
    parser.add_argument('--logDir', type=str)

    parser.add_argument(
        '--split',
        default='val',
        required=True,
        choices=['train', 'val', 'test', 'template'],
        type=str
    )

    parser.add_argument('--verbose', default=False, action="store_true")
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--result', type=str, default='final_predictions.json')

    args = parser.parse_args()
    return args


def reset_config(config, args):
    if args.gpus:
        config.GPUS = args.gpus
    if args.workers:
        config.WORKERS = args.workers
    if args.dataDir:
        config.DATA_DIR = args.dataDir
    if args.modelDir:
        config.OUTPUT_DIR = args.modelDir
    if args.logDir:
        config.LOG_DIR = args.logDir
    if args.verbose:
        config.VERBOSE = args.verbose
    if args.result:
        config.RESULT = args.result

    if args.debug:
        config.DEBUG = True


def nms(dets, thresh=0.4, top_k=-1):
    if len(dets) == 0:
        return []

    order = np.arange(0, len(dets), 1)
    dets = np.array(dets)

    x1 = dets[:, 0]
    x2 = dets[:, 1]
    lengths = x2 - x1

    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(i)

        if len(keep) == top_k:
            break

        xx1 = np.maximum(x1[i], x1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])

        inter = np.maximum(0.0, xx2 - xx1)
        ovr = inter / (lengths[i] + lengths[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return dets[keep]


def save_prediction_to_file(result_path, predictions, annotations):
    result_json = {
        "version": "10_head_ensemble",
        "challenge": "ego4d_nlq_challenge",
        "results": []
    }

    for qid, segs in predictions.items():

        annotation = annotations[qid]

        segs = sorted(segs, key=lambda x: x[2], reverse=True)
        segs = nms(
            segs,
            thresh=config.TEST.NMS_THRESH,
            top_k=5
        ).tolist()

        result = {
            "clip_uid": annotation["clip"],
            "annotation_uid": qid.split("_")[0],
            "query_idx": annotation["query_idx"],
            "predicted_times": [x[:2] for x in segs]
        }

        result_json["results"].append(result)

    with open(result_path, "w") as f:
        json.dump(result_json, f)

    print("Saved:", result_path)


if __name__ == '__main__':

    args = parse_args()
    reset_config(config, args)

    device = ("cuda" if torch.cuda.is_available() else "cpu")

    model = getattr(models, config.MODEL.NAME)()

    model_checkpoint = torch.load(config.MODEL.CHECKPOINT)
    model.load_state_dict(model_checkpoint)

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    model = model.to(device)
    model.eval()

    def get_proposal_results(scores, durations):

        out_sorted_times = []
        
        for score, duration in zip(scores, durations):
            

            T = score.shape[-1]

            score_cpu = score.cpu().detach().numpy()

            sorted_indexs = np.dstack(
                np.unravel_index(
                    np.argsort(score_cpu.ravel())[::-1],
                    (T, T)
                )
            ).tolist()

            sorted_indexs = np.array(
                [x for x in sorted_indexs[0] if x[0] <= x[1]]
            ).astype(float)

            sorted_scores = np.array(
                [score_cpu[0, int(x[0]), int(x[1])] for x in sorted_indexs]
            )
 

            # BEFORE NORMALIZATION STATS
            # mean_before = sorted_scores.mean()
            # std_before = sorted_scores.std()

            # print(f"[Before Norm] Mean: {mean_before:.4f}, Std: {std_before:.4f}")
#MIN-MAX NORMALIZATION
            min_s = sorted_scores.min()
            max_s = sorted_scores.max()

            if max_s - min_s > 1e-6:
                sorted_scores = (sorted_scores - min_s) / (max_s - min_s)
            


            


            # AFTER NORMALIZATION STATS
            # mean_after = sorted_scores.mean()
            # std_after = sorted_scores.std()

            # print(f"[After Norm ] Mean: {mean_after:.4f}, Std: {std_after:.4f}")

            sorted_indexs[:, 1] += 1

            sorted_indexs = torch.from_numpy(sorted_indexs).cuda()

            target_size = config.DATASET.NUM_SAMPLE_CLIPS // config.DATASET.TARGET_STRIDE

            sorted_time = (
                sorted_indexs.float() / target_size * duration
            ).tolist()

            results = []

            # TOP 5 PER HEAD
            for t, s in zip(sorted_time[:5], sorted_scores[:5]):
                results.append([t[0], t[1], s])

            out_sorted_times.append(results)

        return out_sorted_times

    def network(sample):

        textual_input = sample['batch_word_vectors'].cuda()
        textual_mask = sample['batch_txt_mask'].cuda()
        visual_input = sample['batch_vis_input'].cuda()
        map_gt = sample['batch_map_gt'].cuda()
        duration = sample['batch_duration']

        prediction, map_mask = model(
            textual_input,
            textual_mask,
            visual_input
        )

        loss_value, joint_prob = getattr(
            loss,
            config.LOSS.NAME
        )(
            prediction,
            map_mask,
            map_gt,
            config.LOSS.PARAMS
        )


        sorted_times = get_proposal_results(joint_prob, duration)

        return loss_value, sorted_times

    all_predictions = {}
    all_annotations = {}

    
    
    
    for head_idx in range(10):

        print(f"\n========== HEAD {head_idx} ==========")

        test_dataset = getattr(
            datasets,
            config.DATASET.NAME
        )(
            args.split, head_idx=head_idx        )
    
        dataloader = DataLoader(
                test_dataset,
                batch_size=config.TRAIN.BATCH_SIZE,
                shuffle=False,
                num_workers=config.WORKERS,
                pin_memory=False,
                collate_fn=datasets.collate_fn
            )

        for sample in tqdm(dataloader):

            _, batch_output = network(sample)

            batch_idxs = sample["batch_anno_idxs"]

            for local_i, anno_idx in enumerate(batch_idxs):

                ann = test_dataset.annotations[anno_idx]
                qid = ann["query_uid"]

                if qid not in all_predictions:
                    all_predictions[qid] = []
                    all_annotations[qid] = ann

                offset = ann["window"][0]

                for seg in batch_output[local_i]:
                    all_predictions[qid].append([
                        seg[0] + offset,
                        seg[1] + offset,
                        seg[2]
                    ])

    

    print("\n========== FINAL MERGE ==========")

    save_prediction_to_file(
        config.RESULT,
        all_predictions,
        all_annotations
    )

    print("Done.")
