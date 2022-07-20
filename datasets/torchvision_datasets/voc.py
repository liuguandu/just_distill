import os

import torch
import torch.utils.data
from PIL import Image
import sys
import scipy.io as scio
import cv2
import numpy
import xml.etree.ElementTree as ET
import datasets.transforms as T
# from bounding_box import BoxList
class PascalVOCDataset(torch.utils.data.Dataset):
    """
    CLASSES = ("__background__ ", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
               "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor")
    """
    CLASSES = ("__background__ ", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
               "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor")

    def __init__(self, data_dir, split, use_difficult=False, transforms=None, old_classes=[],
                 new_classes=[], excluded_classes=[], is_train=True):
        self.root = data_dir
        self.image_set = split  # train, validation, test
        self.keep_difficult = use_difficult
        self.transforms = transforms

        self._annopath = os.path.join(self.root, "Annotations", "%s.xml")
        self._imgpath = os.path.join(self.root, "JPEGImages", "%s.jpg")
        self._imgsetpath = os.path.join(self.root, "ImageSets", "Main", "%s.txt")
        # self._proposalpath = os.path.join(self.root, "EdgeBoxesProposals", "%s.mat")


        self.old_classes = old_classes
        self.new_classes = new_classes
        self.exclude_classes = excluded_classes
        self.is_train = is_train

        # load data from all categories
        # self._normally_load_voc()

        # do not use old data
        if self.is_train:  # training mode
            print('voc.py | in training mode')
            self._load_img_from_NEW_cls_without_old_data()
        else:
            print('voc.py | in test mode')
            self._load_img_from_NEW_and_OLD_cls_without_old_data()
    def _normally_load_voc(self):
        print("voc.py | normally_load_voc | load data from all 20 categories")
        with open(self._imgsetpath % self.image_set) as f:
            self.ids = f.readlines()
        self.ids = [x.strip("\n") for x in self.ids]
        self.final_ids = self.ids
        self.id_to_img_map = {k: v for k, v in enumerate(self.ids)}  # image_index : image_id

        cls = PascalVOCDataset.CLASSES
        self.class_to_ind = dict(zip(cls, range(len(cls))))  # class_name : class_id
    def _load_img_from_NEW_and_OLD_cls_without_old_data(self):
        self.ids = []
        total_classes = self.new_classes + self.old_classes
        for w in range(len(total_classes)):
            category = total_classes[w]
            img_per_categories = []
            with open(self._imgsetpath % "{0}_{1}".format(category, self.image_set)) as f:
                buff = f.readlines()
            buff = [x.strip("\n") for x in buff]
            for i in range(len(buff)):
                img = buff[i]
                img = img.split(' ')
                if img[1] == "-1":
                    pass
                else:
                    img_per_categories.append(img[0])
                    self.ids.append(img[0])
            print('voc.py | load_img_from_NEW_and_OLD_cls_without_old_data | number of images in {0}_{1}: {2}'.format(category, self.image_set, len(img_per_categories)))
        self.final_ids = list(set(self.ids))
        print('voc.py | load_img_from_NEW_and_OLD_cls_without_old_data | total used number of images in {0}: {1}'.format(self.image_set, len(self.final_ids)))
        self.id_to_img_map = {k: v for k, v in enumerate(self.final_ids)}
        cls = PascalVOCDataset.CLASSES
        self.class_to_ind = dict(zip(cls, range(len(cls))))
    def _load_img_from_NEW_cls_without_old_data(self):
        self.ids = []
        for incremental in self.new_classes:
            img_ids_per_category = []
            with open(self._imgsetpath % "{0}_{1}".format(incremental, self.image_set)) as f:
                buff = f.readlines()
                buff = [x.strip("\n") for x in buff]
            for i in range(len(buff)):
                img = buff[i]
                img = img.split(' ')
                if img[1] == "-1": # do not contain the category object
                    pass
                else:
                    img_ids_per_category.append(img[0])
                    self.ids.append(img[0])
            print('voc.py | load_img_from_NEW_cls_without_old_data | number of images in {0}_{1} set: {2}'.format(incremental, self.image_set, len(img_ids_per_category)))
        self.final_ids = list(set(self.ids))
        print('voc.py | load_img_from_NEW_and_OLD_cls_without_old_data | total used number of images in {0}: {1}'.format(self.image_set, len(self.final_ids)))

        self.id_to_img_map = {k: v for k, v in enumerate(self.final_ids)}
        clas = PascalVOCDataset.CLASSES
        self.class_to_ind = dict(zip(clas, range(len(clas))))
    def clip_to_image(self, remove_empty=True):
        pass
    def map_class_id_to_class_name(self, class_id):
        return PascalVOCDataset.CLASSES[class_id]
    def _preprocess_annotation(self, target):
        boxes = []
        gt_classes = []
        difficult_boxes = []
        for obj in target.iter("object"):
            difficult = int(obj.find("difficult").text) == 1
            if not self.keep_difficult and difficult:
                continue
            name = obj.find("name").text.lower().strip()
            
            # if name in self.old_classes:
            bb = obj.find("bndbox")
            box = [bb.find("xmin").text, bb.find("ymin").text, bb.find("xmax").text, bb.find("ymax").text]
            # print("box:", box)
            bndbox = tuple(map(lambda x: x, list(map(int, box))))
            if self.is_train and name in self.old_classes:
                # print('voc.py | incremental train | object category belongs to old categoires: {0}'.format(name))
                pass
            else:
                boxes.append(bndbox)
                gt_classes.append(self.class_to_ind[name])
                difficult_boxes.append(difficult)
        size = target.find("size")
        im_info = tuple(map(int, (size.find("height").text, size.find("width").text)))
        res = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(gt_classes),
            "difficult": torch.tensor(difficult_boxes),
            "im_info": im_info,
        }
        return res
    def get_img_info(self, index):
        anno = ET.parse(self._annopath % index).getroot()
        size = anno.find("size")
        im_info = tuple(map(int, (size.find("height").text, size.find("width").text)))
        return {"height": im_info[0], "width": im_info[1]}
    def get_groundtruth(self, index):
        img_id = self.final_ids[index]
        anno = ET.parse(self._annopath % img_id).getroot()
        anno = self._preprocess_annotation(anno)

        height, width = anno["im_info"]
        boxes = anno["boxes"]
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 0::2].clamp_(min=0, max=width)
        boxes[:, 1::2].clamp_(min=0, max=height)
        area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        target = {}
        labels = torch.tensor(anno["labels"], dtype=torch.int64)
        difficult = torch.tensor(anno["difficult"], dtype=torch.int64)
        target["boxes"] = boxes[keep]
        target["labels"] = labels[keep]
        target["area"] = area[keep]
        target["orig_size"] = torch.as_tensor([int(height), int(width)])
        target["size"] = torch.as_tensor([int(height), int(width)])
        target["difficult"] = difficult[keep]
        iscrowd = [0 for obj in anno["labels"]]
        iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)
        target["iscrowd"] = iscrowd[keep]
        # print('img_id:', img_id)
        target["image_id"] = torch.tensor([int(img_id)])

        # target = BoxList(anno["boxes"], (width, height), mode="xyxy")
        # target.add_field("labels", anno["labels"])
        # target.add_field("difficult", anno["difficult"])
        return target
        
    def __getitem__(self, index):
        img_id = self.final_ids[index]
        img = Image.open(self._imgpath % img_id).convert("RGB")

        target = self.get_groundtruth(index)
        # target = target.clip_to_image(remove_empty=True)
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        return img, target
    def __len__(self):
        return len(self.final_ids)
def make_voc_transforms(image_set):
    normalize = T.Compose([
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    if image_set == 'train':
        return T.Compose([
                T.RandomHorizontalFlip(),
                T.RandomSelect(
                    T.RandomResize(scales, max_size=1333),
                    T.Compose([
                        T.RandomResize([400, 500, 600]),
                        T.RandomSizeCrop(384, 600),
                        T.RandomResize(scales, max_size=1333),
                    ])
                ),
                normalize,
            ])

    else:
        return T.Compose([
                T.RandomResize([800], max_size=1333),
                normalize,
            ])

    raise ValueError(f'unknown {image_set}')

def build(image_set, old_classes, new_classes):
    is_train = False
    if image_set == "train" or image_set == 'trainval':
        is_train = True
    voc_folder = './data/VOC2007'
    dataset = PascalVOCDataset(voc_folder, image_set, transforms=make_voc_transforms(image_set), old_classes=old_classes,
                    new_classes=new_classes, is_train=is_train)
    return dataset
