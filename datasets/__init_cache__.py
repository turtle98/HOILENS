from torch.utils.data import Dataset
from datasets.vcoco import VCOCO
from datasets.hicodet import HICODet
import datasets.transform as T
import os
import pocket
from PIL import Image

import numpy as np
import torch
from utils.hico_text_label import hico_unseen_index
import pickle
import clip
import torch.nn.functional as F
def custom_collate(batch):
    images = []
    targets = []
    
    for im, tar in batch:
        images.append(im)
        targets.append(tar)

    return images, targets


class DataFactory(Dataset):
    def __init__(self, name, partition, data_root, clip_model_name, detr_backbone, zero_shot=False, zs_type='rare_first',
                 num_classes=600, args=None):  ##ViT-B/16, ViT-L/14@336px
        if name not in ['hicodet', 'vcoco']:
            raise ValueError("Unknown dataset ", name)
        self._load_features= False
        assert clip_model_name in ['ViT-L/14@336px', 'ViT-B/16']
        self.clip_model_name = clip_model_name
        if self.clip_model_name == 'ViT-B/16':
            self.clip_input_resolution = 224
        elif self.clip_model_name == 'ViT-L/14@336px':
            self.clip_input_resolution = 336
        self.llava_feature_dir = args.llava_feature_dir
        if name == 'hicodet':
            # self._text_features = pickle.load(open('inference_features_vit16.p','rb'))
            assert partition in ['train2015', 'test2015'], \
                "Unknown HICO-DET partition " + partition
            self.dataset = HICODet(
                root=os.path.join(data_root, 'hico_20160224_det/images', partition),
                anno_file=os.path.join(data_root, 'instances_{}.json'.format(partition)),
                target_transform=pocket.ops.ToTensor(input_format='dict'), args= args
            )
            if partition == 'train2015':
                import pickle
                feat_dir = os.path.join(self.llava_feature_dir,'bbox_files',f"{name}_train_bbox_{detr_backbone}.p")
                self.anno_bbox = pickle.load(open(feat_dir, 'rb'))
            else:
                import pickle
                feat_dir = os.path.join(self.llava_feature_dir,'bbox_files',f"{name}_test_bbox_{detr_backbone}.p")
                self.anno_bbox = pickle.load(open(feat_dir, 'rb'))
            # pdb.set_trace()
        else:
            assert partition in ['train', 'val', 'trainval', 'test'], \
                "Unknown V-COCO partition " + partition
            image_dir = dict(
                train='mscoco2014/train2014',
                val='mscoco2014/train2014',
                trainval='mscoco2014/train2014',
                test='mscoco2014/val2014'
            )
            self.dataset = VCOCO(
                root=os.path.join(data_root, image_dir[partition]),
                anno_file=os.path.join(data_root, 'instances_vcoco_{}.json'.format(partition)
                ), target_transform=pocket.ops.ToTensor(input_format='dict')
            )
            if partition == 'trainval':
                feat_dir = os.path.join(self.llava_feature_dir,'bbox_files',f"{name}_train_bbox_{detr_backbone}.p")
                self.anno_bbox = pickle.load(open(feat_dir, 'rb'))
            elif partition == 'test':
                feat_dir = os.path.join(self.llava_feature_dir,'bbox_files',f"{name}_test_bbox_{detr_backbone}.p")
                self.anno_bbox = pickle.load(open(feat_dir, 'rb'))

        # add clip normalization 
        normalize = T.Compose([
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        normalize_clip = T.Compose([
            T.ToTensor(),
            T.Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711])
        ])
        normalize_clip_1 = T.ToTensor()
        normalize_clip_2 = T.Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711])
        scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
        if partition.startswith('train'):
            self.transforms = [T.Compose([
                T.RandomHorizontalFlip(),
                T.ColorJitter(.4, .4, .4),
                T.RandomSelect(
                    T.RandomResize(scales, max_size=1333),
                    T.Compose([
                        T.RandomResize([400, 500, 600]),
                        T.RandomSizeCrop(384, 600),
                        T.RandomResize(scales, max_size=1333),
                    ]))
                    ]),
        normalize, normalize_clip,
        T.Compose([
                 T.IResize([self.clip_input_resolution,self.clip_input_resolution])
            ])
        ]
        else:   
            self.transforms = [T.Compose([
                T.RandomResize([800], max_size=1333),
            ]),
            normalize, normalize_clip,
            T.Compose([
                 T.IResize([self.clip_input_resolution,self.clip_input_resolution])
            ]),
            normalize_clip_1,
            normalize_clip_2
            ]

        self.partition = partition
        self.name = name
        self.count=0

        device = "cuda"
        _, self.process = clip.load(self.clip_model_name, device=device)

        self.partition = partition
        self.name = name
        self.count=0
        self.zero_shot = zero_shot
        if self.name == 'hicodet' and self.zero_shot and self.partition == 'train2015':
            self.zs_type = zs_type
            self.filtered_hoi_idx = hico_unseen_index[self.zs_type]

        self.keep = [i for i in range(len(self.dataset))]

        if self.name == 'hicodet' and self.zero_shot and self.partition == 'train2015':
            self.zs_keep = []
            self.remain_hoi_idx = [i for i in np.arange(600) if i not in self.filtered_hoi_idx]
            if os.path.exists(f'datasets/zs_{args.zs_type}_{self.partition}_idx.pkl'):
                import pickle
                self.zs_keep = pickle.load(open(f'datasets/zs_{args.zs_type}_{self.partition}_idx.pkl', 'rb'))
                print(f'datasets/zs_{args.zs_type}_{self.partition}_idx.pkl is loaded')
            else:
                for i in self.keep:
                    (image, target), filename = self.dataset[i]
                    # if 1 in target['hoi']:
                    #     pdb.set_trace()
                    mutual_hoi = set(self.remain_hoi_idx) & set([_h.item() for _h in target['hoi']])
                    if len(mutual_hoi) != 0:
                        self.zs_keep.append(i)
                import pickle
                pickle.dump(self.zs_keep, open(f'datasets/zs_{args.zs_type}_{self.partition}_idx.pkl', 'wb'))
            self.keep = self.zs_keep

            self.dataset.zs_object_to_target = [[] for _ in range(self.dataset.num_object_cls)]
            if num_classes == 600:
                for corr in self.dataset.class_corr:
                    if corr[0] not in self.filtered_hoi_idx:
                        self.dataset.zs_object_to_target[corr[1]].append(corr[0])
            else:
                for corr in self.dataset.class_corr:
                    if corr[0] not in self.filtered_hoi_idx:
                        self.dataset.zs_object_to_target[corr[1]].append(corr[2])

            #import pdb; pdb.set_trace()


    def __len__(self):
        return len(self.dataset)
 
    ##  padding zeros
    def __getitem__(self, i):
        # pdb.set_trace()
        (image, target), filename = self.dataset[i]
        w,h = image.size
        target['orig_size'] = torch.tensor([h,w])
        target['filename'] = filename

        anno_bbox_list = self.anno_bbox[filename][0]
        target['ex_bbox'] = torch.as_tensor(anno_bbox_list['boxes'])
        target['ex_scores'] = torch.as_tensor(anno_bbox_list['scores'])
        target['ex_labels'] = torch.as_tensor(anno_bbox_list['labels'])
        target['ex_hidden_states'] = torch.as_tensor(anno_bbox_list['hidden_states'])
        # pdb.set_trace()
        if self.name == 'hicodet':
            target['labels'] = target['verb'] 
            # Convert ground truth boxes to zero-based index and the
            # representation from pixel indices to coordinates
            target['boxes_h'][:, :2] -= 1 ## why not [:,:4] -= 1?
            target['boxes_o'][:, :2] -= 1
        else:
            target['labels'] = target['actions']
            target['object'] = target.pop('objects')

        if self._load_features:
            raise NotImplementedError
            all_images = torch.as_tensor(self._text_features[filename])
        else:
            crop_size_human, crop_size_object, crop_size = self.get_region_proposals(target,image_h=image.size[1], image_w=image.size[0])
            crop_size_human, crop_size_object, crop_size = crop_size_human.numpy(), crop_size_object.numpy(), crop_size.numpy()
            all_images = []
            all_objects = []
            all_human = []
            
            for crop_s, crop_s_o, crop_s_h in zip(crop_size,crop_size_object,crop_size_human):
                new_img = image.crop(crop_s)
                new_img = self.expand2square(new_img,(0,0,0)) #
                all_images.append(self.process(new_img))
                new_img = image.crop(crop_s_o)
                new_img = self.expand2square(new_img,(0,0,0)) #
                all_objects.append(self.process(new_img))
                new_img = image.crop(crop_s_h)
                new_img = self.expand2square(new_img,(0,0,0)) #
                all_human.append(self.process(new_img))
            
            all_images = torch.stack(all_images)
            # all_images_object = torch.stack(all_objects)
            # all_images_human = torch.stack(all_human)
            # all_images = torch.cat([all_images_human,all_images_object,all_images],dim=0)
        
        image_0, target_0 = self.transforms[3](image, target)
        image_clip, target = self.transforms[2](image_0, target_0)
        if image_0.size[-1] >self.clip_input_resolution or image_0.size[-2] >self.clip_input_resolution:
            print(image_0.size)

        mask = torch.zeros((len(target['ex_bbox']), 336, 336), dtype=torch.bool)
        for i in range(len(target['ex_bbox'])):
            t = target['ex_bbox'][i].clamp(0,336).int()
            mask[i, t[1]:t[3], t[0]:t[2]] = 1
        # pdb.set_trace()
        assert mask.shape[0] != 0
        mask = F.interpolate(mask[None].float(), size=(7,7)).to(torch.bool)[0]
        target['ex_mask'] = mask

        return all_images, target
    

    def expand2square(self, pil_img, background_color):
        width, height = pil_img.size
        if width == height:
            return pil_img
        elif width > height:
            result = Image.new(pil_img.mode, (width, width), background_color)
            result.paste(pil_img, (0, (width - height) // 2))
            return result
        else:
            result = Image.new(pil_img.mode, (height, height), background_color)
            result.paste(pil_img, ((height - width) // 2, 0))
            return result
            
    def get_region_proposals(self, results,image_h, image_w):
        human_idx = 0
        min_instances = 3
        max_instances = 15
        bx = results['ex_bbox']
        sc = results['ex_scores']
        lb = results['ex_labels'] ## object-category labels(0~80)
        hs = results['ex_hidden_states']
        is_human = lb == human_idx
        hum = torch.nonzero(is_human).squeeze(1)
        obj = torch.nonzero(is_human == 0).squeeze(1)
        n_human = is_human.sum(); n_object = len(lb) - n_human
        # Keep the number of human and object instances in a specified interval
        device = torch.device('cpu')
        if n_human < min_instances:
            keep_h = sc[hum].argsort(descending=True)[:min_instances]
            keep_h = hum[keep_h]
        elif n_human > max_instances:
            keep_h = sc[hum].argsort(descending=True)[:max_instances]
            keep_h = hum[keep_h]
        else:
            keep_h = hum

        if n_object < min_instances:
            keep_o = sc[obj].argsort(descending=True)[:min_instances]
            keep_o = obj[keep_o]
        elif n_object > max_instances:
            keep_o = sc[obj].argsort(descending=True)[:max_instances]
            keep_o = obj[keep_o]
        else:
            keep_o = obj

        keep = torch.cat([keep_h, keep_o])

        boxes=bx[keep]
        scores=sc[keep]
        labels=lb[keep]
        hidden_states=hs[keep]
        is_human = labels == human_idx
            
        n_h = torch.sum(is_human); n = len(boxes)
        # Permute human instances to the top
        if not torch.all(labels[:n_h]==human_idx):
            h_idx = torch.nonzero(is_human).squeeze(1)
            o_idx = torch.nonzero(is_human == 0).squeeze(1)
            perm = torch.cat([h_idx, o_idx])
            boxes = boxes[perm]; scores = scores[perm]
            labels = labels[perm]; unary_tokens = unary_tokens[perm]
        # Skip image when there are no valid human-object pairs
        if n_h == 0 or n <= 1:
            print(n_h, n)

        # Get the pairwise indices
        x, y = torch.meshgrid(
            torch.arange(n, device=device),
            torch.arange(n, device=device)
        )
        # Valid human-object pairs
        x_keep, y_keep = torch.nonzero(torch.logical_and(x != y, x < n_h)).unbind(1)
        sub_boxes = boxes[x_keep]
        obj_boxes = boxes[y_keep]
        lt = torch.min(sub_boxes[..., :2], obj_boxes[..., :2]) # left point
        rb = torch.max(sub_boxes[..., 2:], obj_boxes[..., 2:]) # right point
        union_boxes = torch.cat([lt,rb],dim=-1)
        sub_boxes[:,0].clamp_(0, image_w)
        sub_boxes[:,1].clamp_(0, image_h)
        sub_boxes[:,2].clamp_(0, image_w)
        sub_boxes[:,3].clamp_(0, image_h)

        obj_boxes[:,0].clamp_(0, image_w)
        obj_boxes[:,1].clamp_(0, image_h)
        obj_boxes[:,2].clamp_(0, image_w)
        obj_boxes[:,3].clamp_(0, image_h)

        union_boxes[:,0].clamp_(0, image_w)
        union_boxes[:,1].clamp_(0, image_h)
        union_boxes[:,2].clamp_(0, image_w)
        union_boxes[:,3].clamp_(0, image_h)

        return sub_boxes, obj_boxes, union_boxes

