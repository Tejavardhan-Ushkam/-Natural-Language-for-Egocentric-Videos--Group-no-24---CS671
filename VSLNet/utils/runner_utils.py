import copy
import glob
import json
import os
import random

import numpy as np
import torch
import torch.backends.cudnn
import torch.utils.data
from tqdm import tqdm

import utils.evaluate_ego4d_nlq as ego4d_eval
from utils.data_util import index_to_time


def build_gt_lookup(ground_truth):
    gt_dict = {}

    for video in ground_truth['videos']:   
        for clip in video['clips']:
            clip_uid = clip['clip_uid']

            for ann in clip['annotations']:
                ann_uid = ann['annotation_uid']

                for idx, q in enumerate(ann['language_queries']):
                    gt_interval = (q['clip_start_sec'], q['clip_end_sec'])

                    key = (clip_uid, ann_uid, idx)
                    gt_dict[key] = gt_interval

    return gt_dict
    
    
def compute_overlap(gt_start, gt_end, pred_start, pred_end):
    return max(0, min(gt_end, pred_end) - max(gt_start, pred_start))
    
def compute_best_score(gt_interval, pred_intervals):
    gt_start, gt_end = gt_interval
    gt_length = gt_end - gt_start

    if gt_length <= 0:
        return 0.0, 0.0, 0.0

    best_score = -1
    best_start_extra = float("inf")
    best_end_extra = float("inf")

    for pred_start, pred_end in pred_intervals:
        overlap = compute_overlap(gt_start, gt_end, pred_start, pred_end)
        score = overlap / gt_length

        start_extra = max(0, gt_start - pred_start)
        end_extra = max(0, pred_end - gt_end)

        # first see primary: score
        # tie breaker: smaller extras
        if (score > best_score) or (
            score == best_score and (start_extra + end_extra < best_start_extra + best_end_extra)
        ):
            best_score = score
            best_start_extra = start_extra
            best_end_extra = end_extra

    return best_score, best_end_extra, best_start_extra

def set_th_config(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def filter_checkpoints(model_dir, suffix="t7", max_to_keep=5):
    model_paths = glob.glob(os.path.join(model_dir, "*.{}".format(suffix)))
    if len(model_paths) > max_to_keep:
        model_file_dict = dict()
        suffix_len = len(suffix) + 1
        for model_path in model_paths:
            step = int(os.path.basename(model_path).split("_")[1][0:-suffix_len])
            model_file_dict[step] = model_path
        sorted_tuples = sorted(model_file_dict.items())
        unused_tuples = sorted_tuples[0:-max_to_keep]
        for _, model_path in unused_tuples:
            os.remove(model_path)


def get_last_checkpoint(model_dir, suffix="t7"):
    model_filenames = glob.glob(os.path.join(model_dir, "*.{}".format(suffix)))
    model_file_dict = dict()
    suffix_len = len(suffix) + 1
    for model_filename in model_filenames:
        step = int(os.path.basename(model_filename).split("_")[1][0:-suffix_len])
        model_file_dict[step] = model_filename
    sorted_tuples = sorted(model_file_dict.items())
    last_checkpoint = sorted_tuples[-1]
    return last_checkpoint[1]


def convert_length_to_mask(lengths):
    max_len = lengths.max().item()
    mask = torch.arange(max_len, device=lengths.device).expand(
        lengths.size()[0], max_len
    ) < lengths.unsqueeze(1)
    mask = mask.float()
    return mask


def eval_test(
    model,
    data_loader,
    device,
    mode="test",
    result_save_path=None,
    gt_json_path=None,
    epoch=None,
    global_step=None,
):
    predictions = []
    
    with torch.no_grad():
        # print("Hlo")
        for idx, (records, vfeats, vfeat_lens, word_ids, char_ids) in tqdm(
            enumerate(data_loader),
            total=len(data_loader),
            desc="evaluate {}".format(mode),
        ):
            # print("in here!!")
            # prepare features
            vfeats, vfeat_lens = vfeats.to(device), vfeat_lens.to(device)
            # print("in here:",vfeats[0])
            # print(vfeat_lens[0])

            if isinstance(word_ids, dict):
                word_ids = {key: val.to(device) for key, val in word_ids.items()}
                # generate mask
                query_mask = (
                    (torch.zeros_like(word_ids["input_ids"]) != word_ids["input_ids"])
                    .float()
                    .to(device)
                )
            else:
                word_ids, char_ids = word_ids.to(device), char_ids.to(device)
                # generate mask
                query_mask = (torch.zeros_like(word_ids) != word_ids).float().to(device)

            # generate mask
            video_mask = convert_length_to_mask(vfeat_lens).to(device)
            # compute predicted results
            _, start_logits, end_logits = model(
                word_ids, char_ids, vfeats, video_mask, query_mask
            )
            # print(f"Shape {start_logits.shape}, \n Shape {end_logits.shape}")
            start_indices, end_indices = model.extract_index(start_logits, end_logits)
            start_indices = start_indices.cpu().numpy()
            end_indices = end_indices.cpu().numpy()

            # Record output and use standard evalution script for NLQ.
            for record, starts, ends in zip(records, start_indices, end_indices):
                # Convert all indices to times.
                timewindow_predictions = []
                for start, end in zip(starts, ends):
                    start_time, end_time = index_to_time(
                        start, end, record["v_len"], record["duration"]
                    )
                    timewindow_predictions.append([float(start_time), float(end_time)])
                #print("Timewindow_predictions:",timewindow_predictions)
                new_datum = {
                    "clip_uid": record["vid"],
                    "annotation_uid": record["annotation_uid"],
                    "query_idx": int(record["query_idx"]),
                    "predicted_times": copy.deepcopy(timewindow_predictions),
                }
                predictions.append(new_datum)

    # Save predictions if path is provided.
    if result_save_path:
        with open(result_save_path, "w") as file_id:
            json.dump(
                {
                    "version": "1.0",
                    "challenge": "ego4d_nlq_challenge",
                    "results": predictions,
                }, file_id
            )

    # Evaluate if ground truth JSON file is provided.
    print("gt_json_path:",gt_json_path)
    if gt_json_path:
        print("gt inside:",gt_json_path)
        with open(gt_json_path) as file_id:
            ground_truth = json.load(file_id)
        thresholds = [0.3, 0.5, 0.01]
        topK = [1, 3, 5]
        # print("Predictions:", predictions)
        # print("Ground truth:", ground_truth["videos"][0]["clips"][0]["annotations"][0])
        # print("thresholds: ", thresholds)
        # print("topK: ", topK)
        # print("Ground truth first entry-", ground_truth["videos"][0]["clips"][0]["annotations"][0])
        results, mIoU = ego4d_eval.evaluate_nlq_performance(
            predictions, ground_truth, thresholds, topK
        )
        # print("Results: ", results)
        # print("mIoU: ", mIoU)
        #print("Predictions-", predictions[:2])
        #print("GT:",type(ground_truth))
        #print(ground_truth.keys())
        gt_dict = build_gt_lookup(ground_truth)
        scores = []
        skipped=0
        
        scores = []
        end_extras = []
        start_extras = []
        
        topk_scores = {k: [] for k in range(1, 11)}
        topk_end_extras = {k: [] for k in range(1, 11)}
        topk_start_extras = {k: [] for k in range(1, 11)}

        for pred in predictions:
        	key = (pred['clip_uid'],pred['annotation_uid'], pred['query_idx'])
        	
        	if key not in gt_dict:
        		skipped+=1
        		continue
        	gt_interval = gt_dict[key]
        	pred_intervals = pred['predicted_times']
        	#print("GT Interval=",gt_interval)
        	#print("All prediction intervals=",pred_intervals)

        	
        	#best_score, end_extra, start_extra = compute_best_score(gt_interval, pred_intervals)
        	#print("Best score=",best_score)
        	#print("End extra=", end_extra)
        	#print("Start extra=",start_extra)
        	
        	#scores.append(best_score)
        	#end_extras.append(end_extra)
        	#start_extras.append(start_extra)
        	
        	#for i in range(1,5):
        	#	best_score_i, end_extra_i, start_extra_i = compute_best_score(gt_interval, pred_intervals[:i])
        	
        	for k in range(1, 11):
        		    best_score_k, end_extra_k, start_extra_k = compute_best_score(gt_interval, pred_intervals[:k])
        		    topk_scores[k].append(best_score_k)
        		    topk_end_extras[k].append(end_extra_k)
        		    topk_start_extras[k].append(start_extra_k)
    
        	
        #final_score = sum(scores) / len(scores) if scores else 0.0
        #print("Final score:",final_score)
        #print("Avg End Extra =", sum(end_extras)/len(end_extras))
        #print("Avg Start Extra =", sum(start_extras)/len(start_extras))
        final_results = {}
        
        for k in range(1, 11):
        	final_results[k] = {
			"score": sum(topk_scores[k]) / len(topk_scores[k]) if topk_scores[k] else 0.0,
			"end_extra": sum(topk_end_extras[k]) / len(topk_end_extras[k]) if topk_end_extras[k] else 0.0,
			"start_extra": sum(topk_start_extras[k]) / len(topk_start_extras[k]) if topk_start_extras[k] else 0.0,
		    }
		    
        if(mode=='train'):
        	final_vslnet_score_file = "final_vslnet_score_train.txt"
        elif(mode=='val'):
        	final_vslnet_score_file = "final_vslnet_score_val.txt"
        else:
        	final_vslnet_score_file = "final_vslnet_score_test.txt"
        
        #with open(final_vslnet_score_file, "a") as f:
        #	f.write(f"Epoch {epoch}, Step {global_step}\n")
        #	f.write(f"Final score: {final_score:.6f}\n")
        #	f.write(f"Avg End Extra: {sum(end_extras)/len(end_extras) if end_extras else 0.0:.6f}\n")
        #	f.write(f"Avg Start Extra: {sum(start_extras)/len(start_extras) if start_extras else 0.0:.6f}\n")
        #	f.write("-" * 40 + "\n")
        
        with open(final_vslnet_score_file, "a") as f:
        	f.write(f"Epoch {epoch}, Step {global_step}\n")
        	
        	for k in range(1, 11):
        		f.write(f"\nTop-{k} Results:\n")
        		f.write(f"  Score: {final_results[k]['score']:.6f}\n")
        		f.write(f"  Avg End Extra: {final_results[k]['end_extra']:.6f}\n")
        		f.write(f"  Avg Start Extra: {final_results[k]['start_extra']:.6f}\n")
        	f.write("-" * 40 + "\n")
        	
        
        #evaluate(predictions, ground_truth)

        title = f"Epoch {epoch}, Step {global_step}"
        display_results = ego4d_eval.display_results(
            results, mIoU, thresholds, topK, title=title
        )
    else:
        results = None
        mIoU = None
        display_results = None
    return results, mIoU, display_results
