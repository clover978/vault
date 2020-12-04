'''
Format filenames for images and jsons to match with COCO dataset.
images' filename should be "%08d.jpg" 
'''
import os
import shutil
import argparse

def translate(dataset_dir, filename, ext):
    '''
    '01':   'dive2217':         '20201105-135203-6000-13_0_182.jpg/json'
    '02':   'vault844':         'n4_0_001303.jpg/.json'
    '02':   'vault1328':        'n1_0_001156.jpg/.json'
    '03':   'vault1112':        '031507000310.jpg/.json'
    '04':   'gymnastics670':    'Rk 001 W C3 130 湖北 商春松 FX Sub-02 Rot-04 2019_05_13_跟拍_1137.jpg/.json'
    '04':   'gymnastics1143':   'Rk 001 M C3 253 八一 肖若腾 FX Sub-01 Rot-01 2019_05_12_跟拍_644.jpg/.json'
    '''

    if dataset_dir in ['dive2217',]:
        dataset_id = 1
        person_id = int(filename[11:15])
        camera_id = int(filename[24])
        frame_id = int(filename[26:].split('.')[0])
    elif dataset_dir in ['vault844', 'vault1328']:
        dataset_id = 2
        person_id = int(filename[1])
        camera_id = int(filename[3])
        frame_id = int(filename[7:11])
    elif dataset_dir in ['vault1112',]:
        dataset_id = 3
        person_id = int(filename[2:6])
        camera_id = int(filename[6:8])
        frame_id = int(filename[8:12])
    elif dataset_dir in ['gymnastics670', 'gymnastics1143']:
        dataset_id = 4
        person_id = int(filename[12:15])
        camera_id = 0
        frame_id = int(filename.split('_')[-1].split('.')[0])
    else:
        raise Exception("Uknown dataset", dataset_dir + \
                            ": should be one of ['dive2217', 'vault844', 'vault1328', 'vault1112', 'gymnastics670', 'gymnastics1143']")

    return f'{dataset_id:02d}{person_id:04d}{camera_id:02d}{frame_id:04d}.{ext}'


def rename(dataset_dir, ext):
    if ext == 'jpg':
        src_dir = os.path.join(dataset_dir, 'archive', 'images')
        dst_dir = os.path.join(dataset_dir, dataset_dir, 'images')
    elif ext == 'json':
        src_dir = os.path.join(dataset_dir, 'archive', 'jsons')
        dst_dir = os.path.join(dataset_dir, dataset_dir, 'annotations', 'jsons')
    else:
        raise Exception("Invaild file extension", ext + ": should be one of [jpg, json]")

    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    fn_map = {}
    for _root, _dirs, files in os.walk(src_dir):
        for filename in files:
            assert filename.endswith(ext), filename
            new_filename = translate(dataset_dir, filename, ext)
            src = os.path.join(src_dir, filename)
            dst = os.path.join(dst_dir, new_filename)
            if new_filename not in fn_map:
                fn_map[new_filename] = filename
            else:
                raise Exception("Conflict filename.", f"{filename} vs {fn_map[new_filename]}")
            shutil.copyfile(src, dst)

    print(f'rename {len(fn_map)} {ext} files.')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='+')
    args = parser.parse_args()
    datasets = args.datasets

    for dataset in datasets:
        rename(dataset, 'jpg')
        rename(dataset, 'json')


if __name__ == "__main__":
    main()