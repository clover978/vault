import os
import json
import shutil
import argparse


def mergeJson(datasets, output_jsonfile):
    with open( os.path.join('..', 'annotations', 'person_keypoints.json' ) ) as f:
        tmpl = json.load(f)

    for dataset in datasets:
        anno_file = os.path.join( dataset, dataset, 'annotations', 'person_keypoints.json')
        with open(anno_file) as f:
            annotations = json.load(f)
        tmpl['images'].extend(annotations['images'])
        tmpl['annotations'].extend(annotations['annotations'])

    with open(output_jsonfile, 'w') as f:
        json.dump(tmpl, f, indent=4, sort_keys=True)


def copyImage(datasets, output_imagedir):
    for dataset in datasets:
        src_dir = os.path.join(dataset, dataset, 'images')
        dst_dir = output_imagedir
        for filename in os.listdir(src_dir):
            shutil.copy( os.path.join(src_dir, filename), dst_dir )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', type=str)
    parser.add_argument('--datasets', nargs='+')
    args = parser.parse_args()
    datasets = args.datasets

    output_jsonfile = os.path.join( '..', 'annotations', 'person_keypoints_{}2017.json'.format(args.phase) )
    mergeJson(datasets, output_jsonfile)

    output_imagedir = os.path.join( '..', 'images', '{}2017/'.format(args.phase) )
    if not os.path.exists(output_imagedir):
        os.makedirs(output_imagedir)
    copyImage(datasets, output_imagedir)


if __name__ == '__main__':
    main()
