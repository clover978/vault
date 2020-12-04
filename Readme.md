## 概述  

Vault 数据集是QDZT公司为3D动捕项目,专门标注的体育动作数据集，标注了人体的 18 个关键点。本仓库基于成都、郑州两家标注公司的共 6 个批次的标注数据，实现了数据解析、验证、格式转换标准化流程，最终生成与 [COCO](https://cocodataset.org/) 数据集格式一致的图片和标注文件，用于 [HRNET](https://github.com/leoxiaobin/deep-high-resolution-net.pytorch) 训练。
  
## 目录组织

按照 COCO2017 数据集格式，数据集图片为 JPG 格式，存储在 `images/train2017, images/val2017, images/test2017` 目录下; 标注文件为单个 JSON 文件，存储为 `annotations/person_keypoints_train2017.json, annotations/person_keypoints_val.json`. Vault 数据集与 COCO2017 保持一致，目录结构如下：

```bash
+ annotations/
  - person_keypoints.json
+ images/
+ tmp/
  + dive2217/
    + annotation_byhand/
    + archive/
      + images/
      + jsons/
  + gymnastics1143/
  + gymnastics670/
  + vault1112/
  + vault1328/
  + vault844/
  - jsonparse.py
  - mergedataset.py
  - rename.py
  - detect.py
```

`tmp/{dataset_id}/archive/` 目录存放标注公司标注的图片文件和标注文件，一张图片对应一个 JSON 文件。本项目中共包含 6 个 datasets.

## 数据转换流程
#### 1. 图片重命名

COCO 数据集中，图片的文件名均为 `"%012d.jpg"`，为保持一致，首先将所有图片和标注文件重命名。  
`rename.py` 将 `archive` 文件夹下的所有 `.jpg` 和 `.json` 文件重命名，并存储到 `tmp/{dataset_id}/{dataset_id}/` 目录下。

```bash
  + dive2217/
    + annotation_byhand/
    + archive/
      + images/
      + jsons/
    + dive2217/
      + images/             # generate by rename.py
      + annotations/
        + jsons/            # generate by rename.py
```

```bash
$ tmp/
python rename.py --datasets dive2217 gymnastics1143 gymnastics670 vault1112 vault1328 vault844
```

#### 2. 包围框检测

数据标注过程中，为节省成本，标注公司只负责标注人体关键点，因此，我们需要生成训练时需要用到的人体包围框。  
`detect.py` 检测数据集图片中的运动员，并记录**图片中最大**的一个人体包围框，人体包围框的 `[x1,y1,x2,y2]` 坐标保存到 `tmp/{dataset_id}/{dataset_id}/annotations/bboxes` 目录下，一张图片对应一个 `.txt` 文件。  
同时生成 **可视化结果**，存储在 `tmp/{dataset_id}/det_vis/` 目录下。  
  注意尽量设置相对显眼的可视化颜色，方便后续包围框校正。

```bash
$ tmp/
python detect.py --datasets dive2217 gymnastics1143 gymnastics670 vault1112 vault1328 vault844 --vis
```

##### 2.1 包围框校正

由于目标检测算法的局限性，需要对检测的结果进行校正。

  1. 使用 [`labelme`](https://github.com/wkentaro/labelme) 打开 `det_vis` 目录，逐张检查检测结果，在需要校正的图片上标注正确的包围框，并保存为同名 JSON 文件。

  2. 将人工标注的 JSON 文件存放在 `tmp/{dataset_id}/annotation_byhand/` 目录下。

  3. 执行 `detect.py` 重新生成 `.txt` 文件，使用 `update`参数可以跳过非人工标注的图片。

```bash
$ tmp/
python detect.py --datasets dive2217 gymnastics1143 gymnastics670 vault1112 vault1328 vault844 --vis --update
```

#### 3. JSON 转换

将标注公司提供的标注文件转换为 COCO 格式的标注。生成新的标注文件 `tmp/{dataset_id}/{dataset_id}/annotations/person_keypoints.json`。  
由于两家公司的标注格式不一样，所以 `jsonparse.py` 中实现了两种解析JSON方法，`transformJson_v1` 和 `transformJson_v2`，前者为 成都淘金客 公司标注，后者为 郑州 公司标注。

```bash
$ tmp/
python detect.py --datasets dive2217 gymnastics1143 gymnastics670 vault1112 vault1328 vault844 --vis
```

最终的 **关键点标注结果** 和 **人体包围框检测结果** 都保存在 `tmp/{dataset_id}/pose_vis/` 目录下，可以人工检查结果。
  
#### 4. 数据集合并

最终训练时可以将各个批次的标注数据合并为一整个数据集。

```bash
+ annotations/
  - person_keypoints.json
  - person_keypoints_train2017.json     # generate by merge_dataset.py
  - person_keypoints_val2017.json       # generate by merge_dataset.py
+ images/
  + train2017
      - 000000xxxxxx.jpg                    # generate by merge_dataset.py
      - ...
  + val2017
      - 000000xxxxxx.jpg                    # generate by merge_dataset.py
      - ...
```

```bash
$tmp
python merge_dataset.py --phase train --datasets dive2217 gymnastics1143 vault1112 vault1328 vault844
python merge_dataset.py --phase val --datasets gymnastics670
```

## TODO

目前最耗时部分为**包围框校正**步骤，为了提高目标检测的准确性，后续可做下列改进：

- [ ] 设置较低的阈值 --threshold。在场景中只有运动员一个人时可以减少漏检

- [ ] 根据关键点的标注辅助选择包围框。
  
- [ ] 使用目标跟踪的方法得到包围框。
  
- [ ] 进行少量验证后，使用验证后的数据快速 finetune 出一个新的目标检测模型。
