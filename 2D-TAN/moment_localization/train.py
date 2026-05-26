from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import pprint
import argparse
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
import datasets
import models
from core.config import config, update_config
from core.engine import Engine
from core.utils import AverageMeter
from core import eval
from core.utils import create_logger
import models.loss as loss
import math
import json
import matplotlib
matplotlib.use('Agg')  # non-interactive backend, safe for servers
import matplotlib.pyplot as plt

########### fix everything ###########
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

torch.autograd.set_detect_anomaly(True)
def parse_args():
    parser = argparse.ArgumentParser(description='Train localization network')

    # general
    parser.add_argument('--cfg', help='experiment configure file name', required=True, type=str)
    args, rest = parser.parse_known_args()

    # update config
    update_config(args.cfg)

    # training
    parser.add_argument('--gpus', help='gpus', type=str)
    parser.add_argument('--workers', help='num of dataloader workers', type=int)
    parser.add_argument('--dataDir', help='data path', type=str)
    parser.add_argument('--modelDir', help='model path', type=str)
    parser.add_argument('--logDir', help='log path', type=str)
    parser.add_argument('--verbose', default=False, action="store_true", help='print progress bar')
    parser.add_argument('--tag', help='tags shown in log', type=str)
    parser.add_argument('--debug', help='debug mode', action='store_true')
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
        config.MODEL_DIR = args.modelDir
    if args.logDir:
        config.LOG_DIR = args.logDir
    if args.verbose:
        config.VERBOSE = args.verbose
    if args.tag:
        config.TAG = args.tag
    if args.debug:
        config.DEBUG = True
        print('=============== debug mode ==============')


if __name__ == '__main__':

    args = parse_args()
    reset_config(config, args)

    logger, final_output_dir = create_logger(config, args.cfg, config.TAG)
    logger.info('\n'+pprint.pformat(args))
    logger.info('\n'+pprint.pformat(config))

    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = config.CUDNN.ENABLED

    dataset_name = config.DATASET.NAME
    model_name = config.MODEL.NAME

    train_dataset = getattr(datasets, dataset_name)('train')
    if config.TEST.EVAL_TRAIN:
        eval_train_dataset = getattr(datasets, dataset_name)('train')
    if not config.DATASET.NO_VAL:
        val_dataset = getattr(datasets, dataset_name)('val')
    test_dataset = getattr(datasets, dataset_name)('test')

    model = getattr(models, model_name)()
    if config.MODEL.CHECKPOINT and config.TRAIN.CONTINUE:
        model_checkpoint = torch.load(config.MODEL.CHECKPOINT)
        model.load_state_dict(model_checkpoint)
    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs")
        model = torch.nn.DataParallel(model)
    device = ("cuda" if torch.cuda.is_available() else "cpu" )
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(),lr=config.TRAIN.LR, betas=(0.9, 0.999), weight_decay=config.TRAIN.WEIGHT_DECAY)
    # optimizer = optim.SGD(model.parameters(), lr=config.TRAIN.LR, momentum=0.9, weight_decay=config.TRAIN.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    factor=config.TRAIN.FACTOR,
    patience=config.TRAIN.PATIENCE
)


    def iterator(split):
        if split == 'train':
            dataloader = DataLoader(train_dataset,
                                    batch_size=config.TRAIN.BATCH_SIZE,
                                    shuffle=config.TRAIN.SHUFFLE,
                                    num_workers=config.WORKERS,
                                    pin_memory=False,
                                    collate_fn=datasets.collate_fn)
        elif split == 'val':
            dataloader = DataLoader(val_dataset,
                                    batch_size=config.TEST.BATCH_SIZE,
                                    shuffle=False,
                                    num_workers=config.WORKERS,
                                    pin_memory=False,
                                    collate_fn=datasets.collate_fn)
        elif split == 'test':
            dataloader = DataLoader(test_dataset,
                                    batch_size=config.TEST.BATCH_SIZE,
                                    shuffle=False,
                                    num_workers=config.WORKERS,
                                    pin_memory=False,
                                    collate_fn=datasets.collate_fn)
        elif split == 'train_no_shuffle':
            dataloader = DataLoader(eval_train_dataset,
                                    batch_size=config.TEST.BATCH_SIZE,
                                    shuffle=False,
                                    num_workers=config.WORKERS,
                                    pin_memory=False,
                                    collate_fn=datasets.collate_fn)
        else:
            raise NotImplementedError

        return dataloader

    def network(sample):
        anno_idxs = sample['batch_anno_idxs']
        textual_input = sample['batch_word_vectors'].cuda()
        textual_mask = sample['batch_txt_mask'].cuda()
        visual_input = sample['batch_vis_input'].cuda()
        map_gt = sample['batch_map_gt'].cuda()
        duration = sample['batch_duration']

        prediction, map_mask = model(textual_input, textual_mask, visual_input)
        loss_value, joint_prob = getattr(loss, config.LOSS.NAME)(prediction, map_mask, map_gt, config.LOSS.PARAMS)

        sorted_times = None if model.training else get_proposal_results(joint_prob, duration)

        return loss_value, sorted_times

    def get_proposal_results(scores, durations):
        # assume all valid scores are larger than one
        out_sorted_times = []
        for score, duration in zip(scores, durations):
            T = score.shape[-1]
            score_cpu = score.cpu().detach().numpy()
            sorted_indexs = np.dstack(np.unravel_index(np.argsort(score_cpu.ravel())[::-1], (T, T))).tolist()
            sorted_indexs = np.array([item for item in sorted_indexs[0] if item[0] <= item[1]]).astype(float)
            sorted_scores = np.array([score_cpu[0, int(x[0]),int(x[1])] for x in sorted_indexs])

            sorted_indexs[:,1] = sorted_indexs[:,1] + 1
            sorted_indexs = torch.from_numpy(sorted_indexs).cuda()
            target_size = config.DATASET.NUM_SAMPLE_CLIPS // config.DATASET.TARGET_STRIDE
            sorted_time = (sorted_indexs.float() / target_size * duration).tolist()
            out_sorted_times.append([[t[0], t[1], s] for t, s in zip(sorted_time, sorted_scores)])

        return out_sorted_times

    metrics_history = {
        'iters': [],                       # x-axis: iteration number
        # losses
        'train_loss': [],
        'val_loss':   [],
        'test_loss':  [],
        # training set metrics
        'train_R1_IoU03': [], 'train_R1_IoU05': [],
        'train_R5_IoU03': [], 'train_R5_IoU05': [],
        'train_mIoU':     [],
        # validation set metrics
        'val_R1_IoU03': [], 'val_R1_IoU05': [],
        'val_R5_IoU03': [], 'val_R5_IoU05': [],
        'val_mIoU':     [],
    }

    def record_metrics(state, train_state, val_state, test_state):
        """Called at every eval step to append current numbers to history."""
        metrics_history['iters'].append(state['t'])
        metrics_history['train_loss'].append(state['loss_meter'].avg)
        metrics_history['val_loss'].append(val_state['loss_meter'].avg if val_state else None)
        metrics_history['test_loss'].append(test_state['loss_meter'].avg)
 
        if train_state:
            m = train_state['Rank@N,mIoU@M']   # shape [n_tious, n_recalls]
            metrics_history['train_R1_IoU03'].append(float(m[0, 0]) * 100)
            metrics_history['train_R1_IoU05'].append(float(m[1, 0]) * 100)
            metrics_history['train_R5_IoU03'].append(float(m[0, 1]) * 100)
            metrics_history['train_R5_IoU05'].append(float(m[1, 1]) * 100)
            metrics_history['train_mIoU'].append(float(train_state['miou']) * 100)
 
        if val_state:
            m = val_state['Rank@N,mIoU@M']
            metrics_history['val_R1_IoU03'].append(float(m[0, 0]) * 100)
            metrics_history['val_R1_IoU05'].append(float(m[1, 0]) * 100)
            metrics_history['val_R5_IoU03'].append(float(m[0, 1]) * 100)
            metrics_history['val_R5_IoU05'].append(float(m[1, 1]) * 100)
            metrics_history['val_mIoU'].append(float(val_state['miou']) * 100)

    def save_metrics_json(save_dir):
        """Dump raw history to JSON so you can reload it anytime."""
        path = os.path.join(save_dir, 'metrics_history.json')
        with open(path, 'w') as f:
            json.dump(metrics_history, f, indent=2)
        print(f'[Metrics] Saved history → {path}')

    def plot_metrics(save_dir):
        """
        Plot all 5 eval metrics (Rank@1/5 × IoU@0.3/0.5, mIoU) for train and val,
        plus the three losses, and save to PNG files.
        """
        iters = metrics_history['iters']
        if not iters:
            return
 
        os.makedirs(save_dir, exist_ok=True)
 
        # ── 1. Loss curves ────────────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(iters, metrics_history['train_loss'], marker='o', label='Train Loss')
        ax.plot(iters, metrics_history['test_loss'],  marker='s', label='Test Loss')
        if any(v is not None for v in metrics_history['val_loss']):
            ax.plot(iters, metrics_history['val_loss'], marker='^', label='Val Loss')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Loss')
        ax.set_title('Loss Curves')
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        loss_path = os.path.join(save_dir, 'loss_curves.png')
        fig.savefig(loss_path, dpi=150)
        plt.close(fig)
        print(f'[Metrics] Saved → {loss_path}')

        metric_pairs = [
            ('train_R1_IoU03', 'val_R1_IoU03', 'Rank@1, mIoU@0.3'),
            ('train_R1_IoU05', 'val_R1_IoU05', 'Rank@1, mIoU@0.5'),
            ('train_R5_IoU03', 'val_R5_IoU03', 'Rank@5, mIoU@0.3'),
            ('train_R5_IoU05', 'val_R5_IoU05', 'Rank@5, mIoU@0.5'),
            ('train_mIoU',     'val_mIoU',     'mIoU'),
        ]
 
        fig, axes = plt.subplots(2, 3, figsize=(16, 9))
        axes = axes.flatten()
 
        for idx, (train_key, val_key, title) in enumerate(metric_pairs):
            ax = axes[idx]
            if metrics_history[train_key]:
                ax.plot(iters, metrics_history[train_key],
                        marker='o', color='steelblue', label='Train')
            if metrics_history[val_key]:
                ax.plot(iters, metrics_history[val_key],
                        marker='^', color='tomato',    label='Val')
            ax.set_title(title, fontsize=11, fontweight='bold')
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Score (%)')
            ax.legend()
            ax.grid(True, alpha=0.3)
 
        # hide unused 6th subplot
        axes[-1].set_visible(False)
 
        fig.suptitle('2D-TAN — Train vs Validation Metrics', fontsize=13, fontweight='bold')
        fig.tight_layout()
        metrics_path = os.path.join(save_dir, 'metrics_curves.png')
        fig.savefig(metrics_path, dpi=150)
        plt.close(fig)
        print(f'[Metrics] Saved → {metrics_path}')

    def on_start(state):
        state['loss_meter'] = AverageMeter()
        state['test_interval'] = int(len(train_dataset)/config.TRAIN.BATCH_SIZE*config.TEST.INTERVAL)
        state['t'] = 1
        model.train()
        if config.VERBOSE:
            state['progress_bar'] = tqdm(total=state['test_interval'])

    def on_forward(state):
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
        state['loss_meter'].update(state['loss'].item(), 1)

    def on_update(state):# Save All
        if config.VERBOSE:
            state['progress_bar'].update(1)

        if state['t'] % state['test_interval'] == 0:
            model.eval()
            if config.VERBOSE:
                state['progress_bar'].close()

            loss_message = '\niter: {} train loss {:.4f}'.format(state['t'], state['loss_meter'].avg)
            table_message = ''
            if config.TEST.EVAL_TRAIN:
                train_state = engine.test(network, iterator('train_no_shuffle'), 'train')
                train_table = eval.display_results(train_state['Rank@N,mIoU@M'], train_state['miou'],
                                                   'performance on training set')
                table_message += '\n'+ train_table
            if not config.DATASET.NO_VAL:
                val_state = engine.test(network, iterator('val'), 'val')
                state['scheduler'].step(-val_state['loss_meter'].avg)
                loss_message += ' val loss {:.4f}'.format(val_state['loss_meter'].avg)
                val_state['loss_meter'].reset()
                val_table = eval.display_results(val_state['Rank@N,mIoU@M'], val_state['miou'],
                                                 'performance on validation set')
                table_message += '\n'+ val_table

            test_state = engine.test(network, iterator('test'), 'test')
            loss_message += ' test loss {:.4f}'.format(test_state['loss_meter'].avg)
            test_state['loss_meter'].reset()
            test_table = eval.display_results(test_state['Rank@N,mIoU@M'], test_state['miou'],
                                              'performance on testing set')
            table_message += '\n' + test_table

            message = loss_message+table_message+'\n'
            logger.info(message)

            _train_state = train_state if config.TEST.EVAL_TRAIN else None
            _val_state   = val_state   if not config.DATASET.NO_VAL else None
            record_metrics(state, _train_state, _val_state, test_state)
            save_metrics_json(final_output_dir)
            plot_metrics(final_output_dir)

            saved_model_filename = os.path.join(config.MODEL_DIR,'{}/{}/iter{:06d}-{:.4f}-{:.4f}.pkl'.format(
                dataset_name, model_name+'_'+config.DATASET.VIS_INPUT_TYPE,
                state['t'], test_state['Rank@N,mIoU@M'][0,0], test_state['Rank@N,mIoU@M'][0,1]))

            rootfolder1 = os.path.dirname(saved_model_filename)
            rootfolder2 = os.path.dirname(rootfolder1)
            rootfolder3 = os.path.dirname(rootfolder2)
            if not os.path.exists(rootfolder3):
                print('Make directory %s ...' % rootfolder3)
                os.mkdir(rootfolder3)
            if not os.path.exists(rootfolder2):
                print('Make directory %s ...' % rootfolder2)
                os.mkdir(rootfolder2)
            if not os.path.exists(rootfolder1):
                print('Make directory %s ...' % rootfolder1)
                os.mkdir(rootfolder1)

            if torch.cuda.device_count() > 1:
                torch.save(model.module.state_dict(), saved_model_filename)
            else:
                torch.save(model.state_dict(), saved_model_filename)


            if config.VERBOSE:
                state['progress_bar'] = tqdm(total=state['test_interval'])
            model.train()
            state['loss_meter'].reset()

    def on_end(state):
        if config.VERBOSE:
            state['progress_bar'].close()

        plot_metrics(final_output_dir)
        save_metrics_json(final_output_dir)
        print(f'[Metrics] All plots saved to {final_output_dir}')


    def on_test_start(state):
        state['loss_meter'] = AverageMeter()
        state['sorted_segments_list'] = []
        if config.VERBOSE:
            if state['split'] == 'train':
                state['progress_bar'] = tqdm(total=math.ceil(len(train_dataset)/config.TEST.BATCH_SIZE))
            elif state['split'] == 'val':
                state['progress_bar'] = tqdm(total=math.ceil(len(val_dataset)/config.TEST.BATCH_SIZE))
            elif state['split'] == 'test':
                state['progress_bar'] = tqdm(total=math.ceil(len(test_dataset)/config.TEST.BATCH_SIZE))
            else:
                raise NotImplementedError

    def on_test_forward(state):
        if config.VERBOSE:
            state['progress_bar'].update(1)
        state['loss_meter'].update(state['loss'].item(), 1)

        min_idx = min(state['sample']['batch_anno_idxs'])
        batch_indexs = [idx - min_idx for idx in state['sample']['batch_anno_idxs']]
        sorted_segments = [state['output'][i] for i in batch_indexs]
        state['sorted_segments_list'].extend(sorted_segments)

    def on_test_end(state):
        annotations = state['iterator'].dataset.annotations
        merge = (state['split'] != 'train')
        state['Rank@N,mIoU@M'], state['miou'] = eval.eval_predictions(state['sorted_segments_list'], annotations, verbose=False, merge_window=merge)
        if config.VERBOSE:
            state['progress_bar'].close()

    engine = Engine()
    engine.hooks['on_start'] = on_start
    engine.hooks['on_forward'] = on_forward
    engine.hooks['on_update'] = on_update
    engine.hooks['on_end'] = on_end
    engine.hooks['on_test_start'] = on_test_start
    engine.hooks['on_test_forward'] = on_test_forward
    engine.hooks['on_test_end'] = on_test_end
    engine.train(network,
                 iterator('train'),
                 maxepoch=config.TRAIN.MAX_EPOCH,
                 optimizer=optimizer,
                 scheduler=scheduler)
