# import json
# pathes = ['/home/liuguandu/lldetr/lifelongdetr/Deformable-DETR/data/VOC2COCO/VOC2007_test.json',
#         '/home/liuguandu/lldetr/lifelongdetr/Deformable-DETR/data/VOC2COCO/VOC2007_trainval_lat10.json',
#         '/home/liuguandu/lldetr/lifelongdetr/Deformable-DETR/data/VOC2COCO/VOC2007_trainval_pre10.json',
#         '/home/liuguandu/lldetr/lifelongdetr/Deformable-DETR/data/VOC2COCO/VOC2007_trainval.json']
# for path in pathes:
#     with open(path) as fp:
#         json_data = json.load(fp)
#     for i in range(len(json_data['categories'])):
#         json_data['categories'][i]["id"] += 1
#     for i in range(len(json_data['annotations'])):
#         json_data['annotations'][i]["category_id"] += 1
#     with open(path, 'w') as fp:
#         json.dump(json_data, fp)

import numpy as np
import torch
import json
from pycocotools.coco import COCO
old_class = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow"]
new_class = ["diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
with open('/home/liuguandu/lldetr/lifelongdetr/Deformable-DETR/data/VOC2COCO/VOC2007_trainval.json') as fp:
    json_data_all = json.load(fp)
with open('/home/liuguandu/lldetr/lifelongdetr/Deformable-DETR/data/VOC2COCO/VOC2007_trainval_pre10.json') as fp:
    json_data_pre10 = json.load(fp)
coco = COCO('/home/liuguandu/lldetr/lifelongdetr/Deformable-DETR/data/VOC2COCO/VOC2007_trainval.json')
for i in range(len(json_data_pre10['images'])):
    annIds = coco.getAnnIds(json_data_all['images'][i]['id'], iscrowd=None)
    anns = coco.loadAnns(annIds)
    old_flag = False
    new_flag = False
    for j in range(len(anns)):
        if anns[j]['category_id'] <= 10:
            old_flag = True
        if anns[j]['category_id'] > 10:
            new_flag = True
    if new_flag == True and old_flag == False:
        json_data_pre10['images'][i]['have_old'] = 0
    else:
        json_data_pre10['images'][i]['have_old'] = 1
with open('/home/liuguandu/lldetr/lifelongdetr/Deformable-DETR/data/VOC2COCO/VOC2007_trainval_pre10_haveOld.json', 'w') as fp:
    json.dump(json_data_pre10, fp)
