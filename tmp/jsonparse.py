import os
import json
import cv2
import argparse
from tqdm import tqdm
from  matplotlib import pyplot as plt


def getIds(dataset):
    annos = []
    for f in os.listdir( os.path.join(dataset, dataset, 'annotations', 'jsons')):
        annos.append(f.split('.')[0])
    imgs = []
    for f in os.listdir( os.path.join(dataset, dataset, 'images')):
        imgs.append(f.split('.')[0])

    assert set(annos) == set(imgs)
    return annos


def getTransformJson(dataset):
    '''
    max(image_id) for COCO is 581929
    max(annotation_id) for COCO is 2229600

    //desperate
    # set image_ids range to 600000~610000 for dive2217
    # set image_ids range to 610000~620000 for vault1328
    # set image_ids range to 620000~630000 for vault844

    set anno_id range to 3000000~3100000 for dive2217
    set anno_id range to 3100000~3200000 for vault1328
    set anno_id range to 3200000~3300000 for vault844
    set anno_id range to 3300000~3400000 for vault1112
    set anno_id range to 3400000~3500000 for gymnastics1143
    set anno_id range to 3500000~3600000 for gymnastics670
    '''
    if dataset == 'dive2217':
        anno_id = 3000000
    elif dataset == 'vault1328':
        anno_id = 3100000
    elif dataset == 'vault844':
        anno_id = 3200000
    elif dataset == 'vault1112':
        anno_id = 3300000
    elif dataset == 'gymnastics1143':
        anno_id = 3400000
    elif dataset == 'gymnastics670':
        anno_id = 3500000
    else:
        raise Exception("Uknown dataset", dataset + \
                            ": should be one of ['dive2217', 'vault844', 'vault1328', 'vault1112', 'gymnastics670', 'gymnastics1143']")

    if dataset in [ 'dive2217', 'vault1328', 'vault1112', 'gymnastics670']:
        transformJson = transformJson_v1                    # ['shapes']
    elif dataset in [ 'vault844', 'gymnastics1143']:
        transformJson = transformJson_v2                    # ['Public'][0]['LandMark']
    else:
        raise Exception("Uknown dataset", dataset + \
                            ": should be one of ['dive2217', 'vault844', 'vault1328', 'vault1112', 'gymnastics670', 'gymnastics1143']")

    return transformJson, anno_id


def generateJson(dataset, fid, transformJson, anno_id_begin):
    new_image = {
        'license': 3,
        'file_name': '',
        'coco_url': '',
        'height': 0,
        'width': 0,
        'date_captured': '',
        'flickr_url': '',
        'id': 0
    }
    new_annotation = {
        'segmentation': [],
        'num_keypoints': 0,
        'area': 0,
        'iscrowd': 0,
        'keypoints': [],
        'image_id': 0,
        'bbox': [],
        'category_id': 1,
        'id': 0
    }

    imagefile = os.path.join( dataset, dataset, 'images', '{}.jpg'.format(fid) )
    image = cv2.imread(imagefile)
    height, width = image.shape[:2]
    new_image['height'] = height
    new_image['width'] = width
    new_image['id'] = int(fid)
    new_image['file_name'] = str(fid).zfill(12) + '.jpg'

    jsonfile = os.path.join( dataset, dataset, 'annotations', 'jsons', '{}.json'.format(fid) )
    bboxfile = os.path.join( dataset, dataset, 'annotations', 'bboxes', '{}.txt'.format(fid) )
    with open(bboxfile) as f:
        bbox = [ round(float(x)) for x in f.readlines()[0].strip().split() ]
    with open(jsonfile) as f:
        old_annotation = json.load(f)

    
    num_kpts, kpts = transformJson(old_annotation)
    x1, y1, x2, y2 = bbox
    bbox_w, bbox_h = x2-x1, y2-y1

    new_annotation['num_keypoints'] = num_kpts
    new_annotation['area'] = bbox_w*bbox_h
    new_annotation['keypoints'] = kpts
    new_annotation['image_id'] = int(fid)
    new_annotation['bbox'] = [x1, y1, bbox_w, bbox_h]
    new_annotation['id'] = anno_id_begin
    anno_id_begin += 1

    return new_image, new_annotation, anno_id_begin


def transformJson_v1(old_annotation):
    shapes = old_annotation['shapes']
    d = dict()
    for label in [2,3,4,5,6,7,9,10,11,12,13,14,19,20,21,22,23,24]:
        d[label] = (0,0,0)
    num_kpts = 0
    for item in shapes:
        label = item['label'].split(',')[0]
        label = int(label)
        points = item['points'][0]
        points = (round(points[0]), round(points[1]), 2)
        d[label] = points
        num_kpts += 1
    kpts = []
    for label in [2,3,4,5,6,7,9,10,11,12,13,14,19,20,21,22,23,24]:
        kpts.extend(d[label])
    
    return num_kpts, kpts


def transformJson_v2(old_annotation):
    labelmap = {
        "RShoulder" : 2,  "RElbow" : 3,  "RWrist" : 4, 
        "LShoulder" : 5,  "LElbow" : 6,  "LWrist" : 7, 
        "RHip" : 9,       "RKnee" : 10,  "RAnkle" : 11,
        "LHip" : 12,      "LKnee" : 13,  "LAnkle" : 14,
        "LBigToe" : 19,   "LSmallToe" : 20,    "LHeel" : 21,
        "RBigToe" : 22,   "RSmallToe" : 23,    "RHeel" : 24,
    }

    shapes = old_annotation['Public'][0]['Landmark']
    d = dict()
    for label in [2,3,4,5,6,7,9,10,11,12,13,14,19,20,21,22,23,24]:
        d[label] = (0,0,0)
    num_kpts = 0
    xs, ys = [], []
    for item in shapes:
        label = labelmap.get(item['type'])
        label = int(label)
        points = item['Points'][0]["X"], item['Points'][0]["Y"]
        points = (round(points[0]), round(points[1]), 2)
        xs.append(points[0])
        ys.append(points[1])
        d[label] = points
        num_kpts += 1
    kpts = []
    for label in [2,3,4,5,6,7,9,10,11,12,13,14,19,20,21,22,23,24]:
        kpts.extend(d[label])

    return num_kpts, kpts


def verify(dataset, annotation, crop=False):
    vis_dir = os.path.join(dataset, 'pose_vis')
    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir)
    if crop:
        crop_dir = os.path.join(dataset, 'crop_vis')
        if not os.path.exists(crop_dir):
            os.makedirs(crop_dir)
    
    link_pairs1 = [
        [0,1,(0,0,128)], [1,2,(0,0,255)], [3,4,(0,128,0)], [4,5,(0,255,0)],     # arm
        [6,7,(0,0,128)], [7,8,(0,0,255)], [9,10,(0,128,0)], [10,11,(0,255,0)],  # leg
        [11,12,(255,0,0)], [11,13,(255,255,0)], [11,14,(255,0,255)], [8,15,(255,0,0)], [8,16,(255,255,0)], [8,17,(255,0,255)],  # foot
    ]
    point_colors1 = [
        (255,255,51),(254,153,41),(44,127,184),  (255,255,51),(254,153,41),(44,127,184),    # arm
        (228,26,28),(49,163,84),(252,176,243),   (228,26,28),(49,163,84),(252,176,243),     # leg
        (0,176,240),(255,255,0),(169, 209, 142), (0,176,240),(255,255,0),(169, 209, 142)    # foot
    ]

    image_id = annotation['image_id']
    imagefile = os.path.join( dataset, dataset, 'images', '{:012d}.jpg'.format(image_id) )
    assert os.path.exists(imagefile), imagefile + 'not found'
    image = cv2.imread(imagefile)

    keypoints = annotation['keypoints']
    for i in range(len(keypoints)//3):
        kpt = keypoints[i*3], keypoints[i*3+1]
        kpt_color = point_colors1[i]
        cv2.circle(image, kpt, 3, kpt_color, thickness=-1)
        cv2.putText(image, str(i), kpt, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 2)
        x,y,w,h = annotation['bbox']
        cv2.rectangle(image, (x,y), (x+w,y+h), (0,255,0), 2)
        # crop = image[y:y+h, x:x+w,:]
        # cropfile = os.path.join(crop_vis, '{}.jpg'.format(image_id))
        # cv2.imwrite(cropfile, crop)

    for *link_pair, pair_color in link_pairs1:
        p1 = keypoints[link_pair[0]*3], keypoints[link_pair[0]*3+1]
        p2 = keypoints[link_pair[1]*3], keypoints[link_pair[1]*3+1]
        if p1 == (0,0) or p2 == (0,0):
            continue
        cv2.line(image, p1, p2, pair_color, 1)

    imagefile = os.path.join(vis_dir, '{:012d}.jpg'.format(image_id))
    cv2.imwrite(imagefile, image)


def parse(dataset, vis=True, crop=False):
    annotations = {}
    annotations['info'] = dict()
    annotations['licenses'] = list()
    annotations['images'] = list()
    annotations['annotations'] = list()
    annotations['categories'] = [
        {
            'supercategory': 'person',
            'id': 1,
            'name': 'person',
            'keypoints': ['nose',
                        'left_eye',
                        'right_eye',
                        'left_ear',
                        'right_ear',
                        'left_shoulder',
                        'right_shoulder',
                        'left_elbow',
                        'right_elbow',
                        'left_wrist',
                        'right_wrist',
                        'left_hip',
                        'right_hip',
                        'left_knee',
                        'right_knee',
                        'left_ankle',
                        'right_ankle'],
            'skeleton': [[16, 14],
                        [14, 12],
                        [17, 15],
                        [15, 13],
                        [12, 13],
                        [6, 12],
                        [7, 13],
                        [6, 7],
                        [6, 8],
                        [7, 9],
                        [8, 10],
                        [9, 11],
                        [2, 3],
                        [1, 2],
                        [1, 3],
                        [2, 4],
                        [3, 5],
                        [4, 6],
                        [5, 7]]
        }
    ]

    transformJson, anno_id_begin = getTransformJson(dataset)
    for fid in tqdm(getIds(dataset)):
        new_images, new_annotations, anno_id_begin = generateJson(dataset, fid, transformJson, anno_id_begin)
        annotations['annotations'].append(new_annotations)
        annotations['images'].append(new_images)
        if vis:
            verify(dataset, new_annotations, crop)

    new_jsonfile = os.path.join( dataset, dataset, 'annotations', 'person_keypoints.json' )
    with open(new_jsonfile, 'w') as f:
        json.dump(annotations, f, indent=4, sort_keys=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='+')
    parser.add_argument('--vis', action='store_true', default=False)
    parser.add_argument('--crop', action='store_true', default=False)
    args = parser.parse_args()
    datasets = args.datasets

    for dataset in datasets:
        parse(dataset, args.vis, args.crop)


if __name__ == '__main__':
    main()
