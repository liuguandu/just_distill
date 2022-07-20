import os
import xml.etree.ElementTree as ET
import json
# img_path = os.pagth.join('../VOC2007', 'JPEGImages', '%s.jpg')
# img = Image.open(img_path % img_id).convert("RGB")
root = '../VOC2007'
img_setpath = os.path.join(root,'ImageSets', 'Main', '%s.txt')
img_path = os.path.join(root, 'JPEGImages', '%s.jpg')
anno_path = os.path.join(root, 'Annotations', '%s.xml')

##### transform trainval data ########
CLASS = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
         "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
dataset = {'categories': [], 'images': [], 'annotations': []}
class_to_ind = dict(zip(CLASS, range(len(CLASS))))
# print(class_to_ind["aeroplane"])
def voc2coco(imageset):
    with open(img_setpath % imageset) as f:
        ids = f.readlines()
    ids = [x.strip('\n') for x in ids]
    # s_img_path = img_path % ids[i]
    print(len(ids))
    for i, name in enumerate(CLASS):
        cat = {"supercategory": "none", "id": i, "name": name}
        dataset['categories'].append(cat)
    print(dataset)
    anno_id = 1
    for i in range(len(ids)):
        file_name = img_path % ids[i]
        xml_file = ET.parse(anno_path % ids[i]).getroot()
        size = xml_file.find("size")
        im_info = tuple(map(int, (size.find('height').text, size.find('width').text)))
        height, width = im_info[0], im_info[1]
        im_id = i + 1
        image = {'file_name': file_name,
                'height': height,
                'width': width,
                'id': im_id,
                }
        dataset['images'].append(image)
        for obj in xml_file.iter('object'):
            name = obj.find('name').text.lower().strip()
            cat_id = class_to_ind[name]
            bb = obj.find('bndbox')
            box = [int(bb.find('xmin').text), int(bb.find('ymin').text), int(bb.find('xmax').text), int(bb.find('ymax').text)]
            o_width = abs(box[2] - box[0])
            o_height = abs(box[3] - box[1])
            anno_box = {
                'area': o_width * o_height,
                'iscrowd': 0,
                'image_id': im_id,
                'bbox': [box[0], box[1], o_width, o_height],
                'category_id': cat_id,
                'id': anno_id,
            }
            dataset['annotations'].append(anno_box)
            anno_id += 1

    json_path = './VOC2007_' + imageset + '.json'
    with open(json_path, 'w') as fp:
        json.dump(dataset, fp)
def split_class(sp_class, out_file):
    with open('./VOC2007_trainval.json') as fp:
        json_data = json.load(fp)
    anno = []
    for i in json_data['annotations']:
        if CLASS[i['category_id']] not in sp_class:
            # print('yes')
            anno.append(i)
    json_data['annotations'] = anno
    with open(out_file, 'w') as fp:
        json.dump(json_data, fp)

sp_class_1 = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow"]
sp_class_2 = ["diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]  
out_file_1 = './VOC2007_trainval_pre10.json'
out_file_2 = './VOC2007_trainval_lat10.json'
split_class(sp_class_1, out_file_1)
split_class(sp_class_2, out_file_2)
# voc2coco('test')
with open(out_file_1) as fp:
    json_data1 = json.load(fp)
with open(out_file_2) as fp:
    json_data2 = json.load(fp)
print(len(json_data1['annotations']))
print(len(json_data2['annotations']))
