"""
Utilities

Fred Zhang <frederic.zhang@anu.edu.au>

The Australian National University
Australian Centre for Robotic Vision
"""

import os
import torch
import numpy as np
import scipy.io as sio
#import wandb
from tqdm import tqdm
from collections import defaultdict

from utils.hico_text_label import hico_unseen_index
import utils.ddp as ddp
import pocket
from pocket.core import DistributedLearningEngine
from pocket.utils import DetectionAPMeter, BoxPairAssociation
import datetime
import time

class CacheTemplate(defaultdict):
    """A template for VCOCO cached results """
    def __init__(self, **kwargs):
        super().__init__()
        for k, v in kwargs.items():
            self[k] = v
    def __missing__(self, k):
        seg = k.split('_')
        # Assign zero score to missing actions
        if seg[-1] == 'agent':
            return 0.
        # Assign zero score and a tiny box to missing <action,role> pairs
        else:
            return [0., 0., .1, .1, 0.]

from torch.cuda import amp
from pocket.ops import relocate_to_cuda

class CustomisedDLE(DistributedLearningEngine):
    def __init__(self, net, dataloader, max_norm=0, num_classes=117,test_loader=None,args=None, **kwargs):
        super().__init__(net, None, dataloader, **kwargs)
        self.net = net
        self.max_norm = max_norm
        self.num_classes = num_classes
        self.train_loader = dataloader
        self.test_loader = test_loader
        self.best_unseen = -1
        self.best_seen = -1
        self.args = args
        if self.args.amp:
            self.scaler = amp.GradScaler(enabled=True)

        self.epoch_start_time = None
        self.last_logged_epoch = 0

    def _on_end_iteration(self):
        # Print stats in the master process
        if self._verbal and self._state.iteration % self._print_interval == 0:
            self._print_statistics()

    def _on_start_iteration(self):
        if self._state.epoch != self.last_logged_epoch:
            self.epoch_start_time = time.time()
            self.last_logged_epoch = self._state.epoch
        self._state.iteration += 1
        self._state.inputs = relocate_to_cuda(self._state.inputs,ignore=True, non_blocking=True)
        self._state.targets = relocate_to_cuda(self._state.targets,ignore=True, non_blocking=True)

    def _print_statistics(self):
        running_loss = self._state.running_loss.mean()
        t_data = self._state.t_data.sum() / self._world_size
        t_iter = self._state.t_iteration.sum() / self._world_size

        t_iter_mean = self._state.t_iteration.mean()
        t_data_mean = self._state.t_data.mean()

        it_sec = t_iter_mean + t_data_mean

        # Print stats in the master process
        if self._rank == 0:
            num_iter = len(self._train_loader)
            n_d = len(str(num_iter))
            current_iter = self._state.iteration - num_iter * (self._state.epoch - 1)
            print(
                "Epoch [{}/{}], Iter. [{}/{}], "
                "Loss: {:.4f}, "
                "Time[Data/Iter./Remain.]: [{:.2f}s/{:.2f}s/{}]".format(
                self._state.epoch, self.epochs,
                str(current_iter).zfill(n_d),
                num_iter, running_loss, t_data, t_iter,  datetime.timedelta(seconds=(num_iter-current_iter)*it_sec)
            ))
        self._state.t_iteration.reset()
        self._state.t_data.reset()
        self._state.running_loss.reset()

    def _on_each_iteration(self):
        self._state.net.train()
        with amp.autocast(enabled=self.args.amp):
            loss_dict = self._state.net(
                *self._state.inputs, targets=self._state.targets)
        if loss_dict['interaction_loss'].isnan():
            raise ValueError(f"The HOI loss is NaN for rank {self._rank}")

        if self.args.amp:
            self._state.loss = sum(loss for loss in loss_dict.values())
            self._state.optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(self._state.loss).backward()
            self.scaler.step(self._state.optimizer)
            self.scaler.update()
        else:
            self._state.loss = sum(loss for loss in loss_dict.values())
            self._state.optimizer.zero_grad(set_to_none=True)
            self._state.loss.backward()
            if self.max_norm > 0:
                torch.nn.utils.clip_grad_norm_(self._state.net.parameters(), self.max_norm)
            self._state.optimizer.step()

    def _on_end_epoch(self):
        # if self._rank == 0:
        #     #self.save_checkpoint()
        if self._state.lr_scheduler is not None:
            self._state.lr_scheduler.step()
        self.net.object_class_to_target_class = self.test_loader.dataset.dataset.object_class_to_target_class
        if self.epoch_start_time is not None:
            epoch_duration = time.time() - self.epoch_start_time
            if self._rank == 0:  # only log on master
                print(f"Epoch {self._state.epoch} training time: {epoch_duration:.2f} seconds")
            self.epoch_start_time = None  # reset for next epoch

        if self.args.dataset == 'vcoco':
            ret = self.cache_vcoco(self.test_loader)
            vsrl_annot_file = 'vcoco/data/vcoco/vcoco_test.json'
            coco_file = 'vcoco/data/instances_vcoco_all_2014.json'
            split_file = 'vcoco/data/splits/vcoco_test.ids'
            vcocoeval = eval_vcoco.VCOCOeval(vsrl_annot_file, coco_file, split_file)
            det_file = 'vcoco_cache/cache.pkl'
            b= vcocoeval._do_eval(ret, ovr_thresh=0.5)
            mAPs = {
                'sc2': b[1]
            }

            #wandb.log(mAPs)
            return
            # raise NotImplementedError(f"Evaluation on V-COCO has not been implemented.")
        
        start_test_time = time.time()
        ap = self.test_hico(self.test_loader, self.args)
        test_duration = time.time() - start_test_time

        self.net.object_class_to_target_class = self.train_loader.dataset.dataset.object_class_to_target_class
        #self.net.tp = None

        # Fetch indices for rare and non-rare classes
        num_anno = torch.as_tensor(self.train_loader.dataset.dataset.anno_interaction)
        rare = torch.nonzero(num_anno < 10).squeeze(1)
        non_rare = torch.nonzero(num_anno >= 10).squeeze(1)
        if self._rank == 0:
            mAPs = {'mAP': ap.mean() * 100,
                    'rare': ap[rare].mean() * 100,
                    'non-rare': ap[non_rare].mean() * 100
                    }

            print(
                f"The mAP is {ap.mean() * 100:.2f},"
                f" rare: {ap[rare].mean() * 100:.2f},"
                f" none-rare: {ap[non_rare].mean() * 100:.2f},"
            )

            if self.args.zs:
                zs_hoi_idx = hico_unseen_index[self.args.zs_type]
                print(f'>>> zero-shot setting({self.args.zs_type}!!)')
                ap_unseen = []
                ap_seen = []
                for i, value in enumerate(ap):
                    if i in zs_hoi_idx:
                        ap_unseen.append(value)
                    else:
                        ap_seen.append(value)

                ap_unseen = torch.as_tensor(ap_unseen).mean()
                ap_seen = torch.as_tensor(ap_seen).mean()

                mAPs.update({"unseen": ap_unseen * 100, "seen": ap_seen * 100})
                print(
                    f"full mAP: {ap.mean() * 100:.2f}",
                    f"unseen: {ap_unseen * 100:.2f}",
                    f"seen: {ap_seen * 100:.2f}",
                )

            log_file_path = os.path.join(self.args.output_dir, "eval_log.txt")
            with open(log_file_path, "a") as f:
                f.write(f"Epoch {self._state.epoch} Evaluation Results:\n")
                f.write(f"Training time : {epoch_duration }\n")
                f.write(f"test time : {test_duration}\n")
                for k, v in mAPs.items():
                    f.write(f"{k}: {v:.2f}\n")
                f.write("\n")
        if self._rank == 0:
            if self._state.epoch % 5 == 0 or self._state.epoch == 1:
                self.save_checkpoint()
            #wandb.log(mAPs)

    @torch.no_grad()
    def test_hico(self, dataloader, args=None):
        net = self._state.net
        net.eval()
        dataset = dataloader.dataset.dataset
        interaction_to_verb = torch.as_tensor(dataset.interaction_to_verb)

        associate = BoxPairAssociation(min_iou=0.5)
        conversion = torch.from_numpy(np.asarray(
            dataset.object_n_verb_to_interaction, dtype=float
        ))
        tgt_num_classes = 600
        
        num_gt = dataset.anno_interaction if args.dataset == "hicodet" else None
        meter = DetectionAPMeter(
            tgt_num_classes, nproc=1,
            num_gt=num_gt,
            algorithm='11P'
        )

        gt_set = []
        pred_list = []
        count = 0
        for batch in tqdm(dataloader):
            inputs = pocket.ops.relocate_to_cuda(batch[0])
            outputs = net(inputs,batch[1])
            # Skip images without detections
            if outputs is None or len(outputs) == 0:
                continue
            # # Batch size is fixed as 1 for inference
            # assert len(output) == 1, f"Batch size is not 1 but {len(outputs)}."
           # import pdb; pdb.set_trace()
            for output, target in zip(outputs, batch[-1]):
                output = pocket.ops.relocate_to_cpu(output, ignore=True)
    
                gt_set.append(target['hoi'])

                # Format detections
                boxes = output['boxes']
                boxes_h, boxes_o = boxes[output['pairing']].unbind(0)
                objects = output['objects']
                scores = output['scores']
                verbs = output['labels']

                if net.module.num_classes==117 or net.module.num_classes==407:
                    interactions = conversion[objects, verbs]
                else:
                    interactions = verbs

                # Recover target box scale
                gt_bx_h = net.module.recover_boxes(target['boxes_h'], target['size'])
                gt_bx_o = net.module.recover_boxes(target['boxes_o'], target['size'])
                # Associate detected pairs with ground truth pairs

                labels = torch.zeros_like(scores)
                unique_hoi = interactions.unique()
                for hoi_idx in unique_hoi:
                    gt_idx = torch.nonzero(target['hoi'] == hoi_idx).squeeze(1)
                    det_idx = torch.nonzero(interactions == hoi_idx).squeeze(1)
                    if len(gt_idx):
                        labels[det_idx] = associate(
                            (gt_bx_h[gt_idx].view(-1, 4),
                            gt_bx_o[gt_idx].view(-1, 4)),
                            (boxes_h[det_idx].view(-1, 4),
                            boxes_o[det_idx].view(-1, 4)),
                            scores[det_idx].view(-1)
                        )
                        # all_det_idxs.append(det_idx)
                # meter.append(scores, interactions, labels)   # scores human*object*verb, interaction（600), labels
                #import pdb; pdb.set_trace()
                results = (scores, interactions, labels)
                pred_list.append(results)

        gathered_pred_list = []
        for preds in ddp.all_gather(pred_list):
            gathered_pred_list.extend(preds)
        for pred in gathered_pred_list:
            meter.append(*pred)

        # Compute score sharpness analysis
        sharpness_results = self.compute_score_sharpness(gathered_pred_list)
        if self._rank == 0 and sharpness_results:
            print("\n=== Score Calibration Analysis (Cross-Entropy) ===")
            if 'all' in sharpness_results:
                a = sharpness_results['all']
                print(f"All (n={a['count']}): "
                      f"mean_score={a['mean_score']:.4f} (±{a['std_score']:.4f}), "
                      f"mean_ce={a['mean_ce']:.4f} (±{a['std_ce']:.4f})")
            if 'tp' in sharpness_results:
                tp = sharpness_results['tp']
                print(f"True Positives (n={tp['count']}): "
                      f"mean_score={tp['mean_score']:.4f} (±{tp['std_score']:.4f}), "
                      f"mean_ce={tp['mean_ce']:.4f} (±{tp['std_ce']:.4f})")
            if 'fp' in sharpness_results:
                fp = sharpness_results['fp']
                print(f"False Positives (n={fp['count']}): "
                      f"mean_score={fp['mean_score']:.4f} (±{fp['std_score']:.4f}), "
                      f"mean_ce={fp['mean_ce']:.4f} (±{fp['std_ce']:.4f})")
            print("=================================================\n")

        ap = meter.eval()
        return ap

    def compute_score_sharpness(self, pred_list):
        """
        Analyze calibration of scores using cross-entropy.
        Compare true positive vs false positive detections.

        Cross-entropy CE(p, y) = -y*log(p) - (1-y)*log(1-p)
        - For TP (y=1): CE = -log(p) -> low when score is high (good)
        - For FP (y=0): CE = -log(1-p) -> low when score is low (good)
        - Lower CE overall means better calibrated predictions
        """
        all_scores_tp = []
        all_scores_fp = []

        for scores, interactions, labels in pred_list:
            tp_mask = labels > 0
            fp_mask = labels == 0

            if tp_mask.any():
                all_scores_tp.append(scores[tp_mask])
            if fp_mask.any():
                all_scores_fp.append(scores[fp_mask])

        scores_tp = torch.cat(all_scores_tp) if all_scores_tp else torch.tensor([])
        scores_fp = torch.cat(all_scores_fp) if all_scores_fp else torch.tensor([])

        def cross_entropy(p, y):
            """CE = -y*log(p) - (1-y)*log(1-p)"""
            p = torch.clamp(p, 1e-7, 1 - 1e-7)
            return -y * torch.log(p) - (1 - y) * torch.log(1 - p)

        results = {}

        # All scores combined with labels
        if len(scores_tp) > 0 or len(scores_fp) > 0:
            all_scores = []
            all_labels = []
            if len(scores_tp) > 0:
                all_scores.append(scores_tp)
                all_labels.append(torch.ones_like(scores_tp))
            if len(scores_fp) > 0:
                all_scores.append(scores_fp)
                all_labels.append(torch.zeros_like(scores_fp))
            scores_all = torch.cat(all_scores)
            labels_all = torch.cat(all_labels)
            ce_all = cross_entropy(scores_all, labels_all)
            results['all'] = {
                'count': len(scores_all),
                'mean_score': scores_all.mean().item(),
                'std_score': scores_all.std().item(),
                'mean_ce': ce_all.mean().item(),
                'std_ce': ce_all.std().item(),
            }

        if len(scores_tp) > 0:
            ce_tp = cross_entropy(scores_tp, torch.ones_like(scores_tp))  # y=1 for TP
            results['tp'] = {
                'count': len(scores_tp),
                'mean_score': scores_tp.mean().item(),
                'std_score': scores_tp.std().item(),
                'mean_ce': ce_tp.mean().item(),
                'std_ce': ce_tp.std().item(),
            }

        if len(scores_fp) > 0:
            ce_fp = cross_entropy(scores_fp, torch.zeros_like(scores_fp))  # y=0 for FP
            results['fp'] = {
                'count': len(scores_fp),
                'mean_score': scores_fp.mean().item(),
                'std_score': scores_fp.std().item(),
                'mean_ce': ce_fp.mean().item(),
                'std_ce': ce_fp.std().item(),
            }
        #import pdb; pdb.set_trace()
        return results

    def measure_fps_100(self, dataloader, args=None):
        net = self._state.net
        net.eval()
        total_images = 0
        count = 0
        max_images = 1000

        dataset = dataloader.dataset.dataset
        interaction_to_verb = torch.as_tensor(dataset.interaction_to_verb)

        associate = BoxPairAssociation(min_iou=0.5)
        conversion = torch.from_numpy(np.asarray(
            dataset.object_n_verb_to_interaction, dtype=float
        ))
        tgt_num_classes = 600
        
        num_gt = dataset.anno_interaction if args.dataset == "hicodet" else None
        meter = DetectionAPMeter(
            tgt_num_classes, nproc=1,
            num_gt=num_gt,
            algorithm='11P'
        )

        gt_set = []
        pred_list = []
        count = 0

        # Warm-up (1 batch to load weights on GPU, etc.)
        for batch in dataloader:
            inputs = pocket.ops.relocate_to_cuda(batch[0])
            _ = net(inputs, batch[1])
            break

        torch.cuda.synchronize()
        start_time = time.time()

        for batch in dataloader:
            inputs = pocket.ops.relocate_to_cuda(batch[0])
            _ = net(inputs, batch[1])
            
            
            total_images += 1
            count += 1
            if total_images >= max_images:
                break

        torch.cuda.synchronize()
        elapsed = time.time() - start_time
        fps = total_images / elapsed

        print(f"Processed {total_images} images in {elapsed:.2f} seconds")
        print(f"FPS: {fps:.2f}")
        return fps
    @torch.no_grad()
    def cache_hico(self, dataloader, cache_dir='matlab'):
        net = self._state.net
        net.eval()

        dataset = dataloader.dataset.dataset
        conversion = torch.from_numpy(np.asarray(
            dataset.object_n_verb_to_interaction, dtype=float
        ))
        object2int = dataset.object_to_interaction

        # Include empty images when counting
        nimages = len(dataset.annotations)
        all_results = np.empty((600, nimages), dtype=object)

        for i, batch in enumerate(tqdm(dataloader)):
            inputs = pocket.ops.relocate_to_cuda(batch[0])
            output = net(inputs)

            # Skip images without detections
            if output is None or len(output) == 0:
                continue
            # Batch size is fixed as 1 for inference
            assert len(output) == 1, f"Batch size is not 1 but {len(output)}."
            output = pocket.ops.relocate_to_cpu(output[0], ignore=True)
            # NOTE Index i is the intra-index amongst images excluding those
            # without ground truth box pairs
            image_idx = dataset._idx[i]
            # Format detections
            boxes = output['boxes']
            boxes_h, boxes_o = boxes[output['pairing']].unbind(0)
            objects = output['objects']
            scores = output['scores']
            verbs = output['labels']
            interactions = conversion[objects, verbs]
            # Rescale the boxes to original image size
            ow, oh = dataset.image_size(i)
            h, w = output['size']
            scale_fct = torch.as_tensor([
                ow / w, oh / h, ow / w, oh / h
            ]).unsqueeze(0)
            boxes_h *= scale_fct
            boxes_o *= scale_fct

            # Convert box representation to pixel indices
            boxes_h[:, 2:] -= 1
            boxes_o[:, 2:] -= 1

            # Group box pairs with the same predicted class
            permutation = interactions.argsort()
            boxes_h = boxes_h[permutation]
            boxes_o = boxes_o[permutation]
            interactions = interactions[permutation]
            scores = scores[permutation]

            # Store results
            unique_class, counts = interactions.unique(return_counts=True)
            n = 0
            for cls_id, cls_num in zip(unique_class, counts):
                all_results[cls_id.long(), image_idx] = torch.cat([
                    boxes_h[n: n + cls_num],
                    boxes_o[n: n + cls_num],
                    scores[n: n + cls_num, None]
                ], dim=1).numpy()
                n += cls_num
        
        # Replace None with size (0,0) arrays
        for i in range(600):
            for j in range(nimages):
                if all_results[i, j] is None:
                    all_results[i, j] = np.zeros((0, 0))
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        # Cache results
        for object_idx in range(80):
            interaction_idx = object2int[object_idx]
            sio.savemat(
                os.path.join(cache_dir, f'detections_{(object_idx + 1):02d}.mat'),
                dict(all_boxes=all_results[interaction_idx])
            )

    @torch.no_grad()
    def cache_vcoco(self, dataloader, cache_dir='vcoco_cache'):
        net = self._state.net
        net.eval()

        dataset = dataloader.dataset.dataset
        all_results = []
        for i, batch in enumerate(tqdm(dataloader)):
            inputs = pocket.ops.relocate_to_cuda(batch[0])
            output = net(inputs)

            # Skip images without detections
            if output is None or len(output) == 0:
                continue
            # Batch size is fixed as 1 for inference
            assert len(output) == 1, f"Batch size is not 1 but {len(output)}."
            output = pocket.ops.relocate_to_cpu(output[0], ignore=True)
            # NOTE Index i is the intra-index amongst images excluding those
            # without ground truth box pairs
            image_id = dataset.image_id(i)
            # Format detections
            boxes = output['boxes']
            boxes_h, boxes_o = boxes[output['pairing']].unbind(0)
            scores = output['scores']
            actions = output['labels']
            # Rescale the boxes to original image size
            ow, oh = dataset.image_size(i)
            h, w = output['size']
            scale_fct = torch.as_tensor([
                ow / w, oh / h, ow / w, oh / h
            ]).unsqueeze(0)
            boxes_h *= scale_fct
            boxes_o *= scale_fct

            for bh, bo, s, a in zip(boxes_h, boxes_o, scores, actions):
                a_name = dataset.actions[a].split()
                result = CacheTemplate(image_id=image_id, person_box=bh.tolist())
                result[a_name[0] + '_agent'] = s.item()
                result['_'.join(a_name)] = bo.tolist() + [s.item()]
                all_results.append(result)

        return all_results