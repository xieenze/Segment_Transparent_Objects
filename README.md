# Segment_Transparent_Objects
## Introduce
This repository contains the data and code for ECCV2020 paper [Segmenting Transparent Objects in the Wild](https://arxiv.org/abs/2003.13948).

For downloading the data, you can refer to [Trans10K Website](https://xieenze.github.io/projects/TransLAB/TransLAB.html).


## Environments

- python 3
- torch = 1.1.0 (>1.1.0 with cause performance drop, we can't find the reason)
- torchvision
- pyyaml
- Pillow
- numpy

## INSTALL

```
python setup.py develop
```
## Pretrained Models and Logs
We provide the trained models and logs for TransLab.
[Google Drive](https://drive.google.com/drive/folders/1yJMEB4rNKIZt5IWL13Nn-YwckrvAPNuz?usp=sharing)

## Demo
1. put the images in './demo/imgs'
2. download the trained model from [Google Drive](https://drive.google.com/drive/folders/1yJMEB4rNKIZt5IWL13Nn-YwckrvAPNuz?usp=sharing)
, and put it in './demo/16.pth'
3. run this script
```
CUDA_VISIBLE_DEVICES=0 python -u ./tools/test_demo.py --config-file configs/trans10K/translab.yaml TEST.TEST_MODEL_PATH ./demo/16.pth  DEMO_DIR ./demo/imgs
```
4. the results are generated in './demo/results'


## Data Preparation
1. create dirs './datasets/Trans10K'
2. download the data from [Trans10K Website](https://xieenze.github.io/projects/TransLAB/TransLAB.html).
3. put the train/validation/test data under './datasets/Trans10K'. Data Structure is shown below.
```
Trans10K/
├── test
│   ├── easy
│   └── hard
├── train
│   ├── images
│   └── masks
└── validation
    ├── easy
    └── hard
```
## Pretrained backbone models 

pretrained backbone models will be download automatically in pytorch default directory(```~/.cache/torch/checkpoints/```).

## Train
Our experiments are based on one machine with 8 V100 GPUs(32g memory), if you face memory error, you can try the 'batchsize=4' version.
### Train with batchsize=8(cost 15G memory)
```
bash tools/dist_train.sh configs/trans10K/translab.yaml 8 TRAIN.MODEL_SAVE_DIR workdirs/translab_bs8
```
### Train with batchsize=4(cost 8G memory)
```
bash tools/dist_train.sh configs/trans10K/translab_bs4.yaml 8 TRAIN.MODEL_SAVE_DIR workdirs/translab_bs4
```

## Eval
for example (batchsize=8)
```
CUDA_VISIBLE_DEVICES=0 python -u ./tools/test_translab.py --config-file configs/trans10K/translab.yaml  TEST.TEST_MODEL_PATH workdirs/translab_bs8/16.pth
```

## License

For academic use, this project is licensed under the Apache License - see the LICENSE file for details. For commercial use, please contact the authors. 

## Citations
Please consider citing our paper in your publications if the project helps your research. BibTeX reference is as follows.

```
@article{xie2020segmenting,
  title={Segmenting Transparent Objects in the Wild},
  author={Xie, Enze and Wang, Wenjia and Wang, Wenhai and Ding, Mingyu and Shen, Chunhua and Luo, Ping},
  journal={arXiv preprint arXiv:2003.13948},
  year={2020}
}
```
