import json
pathes = ['/home/liuguandu/lldetr/lifelongdetr/Deformable-DETR/data/VOC2COCO/VOC2007_test.json',
        '/home/liuguandu/lldetr/lifelongdetr/Deformable-DETR/data/VOC2COCO/VOC2007_trainval_lat10.json',
        '/home/liuguandu/lldetr/lifelongdetr/Deformable-DETR/data/VOC2COCO/VOC2007_trainval_pre10.json',
        '/home/liuguandu/lldetr/lifelongdetr/Deformable-DETR/data/VOC2COCO/VOC2007_trainval.json']

for path in pathes:
    with open(path) as fp:
        json_data = json.load(fp)
    for i in range(len(json_data['categories'])):
        json_data['categories'][i]["id"] -= 1
    for i in range(len(json_data['annotations'])):
        json_data['annotations'][i]["category_id"] -= 1
    with open(path, 'w') as fp:
        json.dump(json_data, fp)