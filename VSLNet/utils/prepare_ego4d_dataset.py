#! /usr/bin/env python
"""
Prepare Ego4d episodic memory NLQ for model training (NO feature extraction).
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import json
import math
import os


CANONICAL_VIDEO_FPS = 30.0
FEATURE_WINDOW_SIZE = 16.0
FEATURES_PER_SEC = CANONICAL_VIDEO_FPS / FEATURE_WINDOW_SIZE


def get_nearest_frame(time, floor_or_ceil=None):
    return floor_or_ceil(int(time * CANONICAL_VIDEO_FPS / FEATURE_WINDOW_SIZE))


def process_question(question):
    return question.strip(" ").strip("?").lower() + "?"


def reformat_data(split_data, test_split=False):
    formatted_data = {}

    for video_datum in split_data["videos"]:
        for clip_datum in video_datum["clips"]:
            clip_uid = clip_datum["clip_uid"]

            clip_duration = (
                clip_datum["video_end_sec"] - clip_datum["video_start_sec"]
            )
            num_frames = get_nearest_frame(clip_duration, math.ceil)

            new_dict = {
                "fps": FEATURES_PER_SEC,
                "num_frames": num_frames,
                "timestamps": [],
                "exact_times": [],
                "sentences": [],
                "annotation_uids": [],
                "query_idx": [],
            }

            for ann_datum in clip_datum["annotations"]:
                for index, datum in enumerate(ann_datum["language_queries"]):

                    if "query" not in datum or not datum["query"]:
                        continue

                    if not test_split:
                        start_time = float(datum["clip_start_sec"])
                        end_time = float(datum["clip_end_sec"])
                    else:
                        start_time = 0.0
                        end_time = 0.0

                    new_dict["sentences"].append(
                        process_question(datum["query"])
                    )
                    new_dict["annotation_uids"].append(
                        ann_datum["annotation_uid"]
                    )
                    new_dict["query_idx"].append(index)
                    new_dict["exact_times"].append([start_time, end_time])

                    new_dict["timestamps"].append(
                        [
                            get_nearest_frame(start_time, math.floor),
                            get_nearest_frame(end_time, math.ceil),
                        ]
                    )

            formatted_data[clip_uid] = new_dict

    return formatted_data


def convert_ego4d_dataset(args):
    for split in ("train", "val", "test"):
        read_path = args[f"input_{split}_split"]
        print(f"Reading [{split}]: {read_path}")

        with open(read_path, "r") as file_id:
            raw_data = json.load(file_id)

        data_split = reformat_data(raw_data, split == "test")

        num_instances = sum(len(ii["sentences"]) for ii in data_split.values())
        print(f"# {split}: {num_instances}")
        print(f"# No. of queries in {split}: {num_instances}")
        os.makedirs(args["output_save_path"], exist_ok=True)
        save_path = os.path.join(args["output_save_path"], f"{split}.json")

        print(f"Writing [{split}]: {save_path}")
        with open(save_path, "w") as file_id:
            json.dump(data_split, file_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--input_train_split", required=True, help="Path to Ego4d train split"
    )
    parser.add_argument(
        "--input_val_split", required=True, help="Path to Ego4d val split"
    )
    parser.add_argument(
        "--input_test_split", required=True, help="Path to Ego4d test split"
    )
    parser.add_argument(
        "--output_save_path", required=True, help="Path to save output jsons"
    )

    args = vars(parser.parse_args())

    convert_ego4d_dataset(args)