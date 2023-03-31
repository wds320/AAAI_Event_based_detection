# Dual Memory Aggregation Network for Event-based Object Detection with Learnable Representation

<div align=center><img src="https://github.com/wds320/AAAI_Event_based_detection/blob/main/demo2.gif" width="80%" height="80%" /></div>

## Abstract
Event-based cameras are bio-inspired sensors that capture brightness change of every pixel in an asynchronous manner. Compared with frame-based sensors, event cameras have microsecond-level latency and high dynamic range, hence showing great potential for object detection under high-speed motion and poor illumination conditions. Due to sparsity and asynchronism nature with event streams, most of existing approaches resort to hand-crafted methods to convert event data into 2D grid representation. However, they are sub-optimal in aggregating information from event stream for object detection. In this work, we propose to learn an event representation optimized for event-based object detection. Specifically, event streams are divided into grids in the x-y-t coordinates for both positive and negative polarity, producing a set of pillars as 3D tensor representation. To fully exploit information with event streams to detect objects, a dual-memory aggregation network (DMANet) is proposed to leverage both long and short memory along event streams to aggregate effective information for object detection. Long memory is encoded in the hidden state of adaptive convLSTMs while short memory is modeled by computing spatial-temporal correlation between event pillars at neighboring time intervals. Extensive experiments on the recently released event-based automotive detection dataset demonstrate the effectiveness of the proposed method.


## Installation
- Step1. Install DMANet
```
git clone https://github.com/wds320/AAAI_Event_based_detection.git
cd DMANet
conda create -n dmanet python=3.6
conda activate dmanet
conda install pytorch==1.4.0 torchvision==0.5.0 cudatoolkit=10.1 -c pytorch
pip install -r requirements.txt
```
- Step2. Install Apex (Mixed Precision Training)

Method a: official build
```
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```
Method b: If you failed to install apex with method a, you can also build with the following commands.
```
cd apex
python setup.py install
```


## Data preparation
- 1 Mpx Auto-Detection Sub Dataset 
 
Since the groundtruth of the [Prophesee dataset](https://www.prophesee.ai/2020/11/24/automotive-megapixel-event-based-dataset/) is obtained automatically by applying an RGB-based detector in the stereo recording setting, there are some geometric errros such as misalignment and semantic errors caused by wrong detection of the detector with the ground truth. We also found that there are some *mosaic* events in this dataset which might be caused by flickering as explained by authors of that dataset. Therefore, we remove theses events and incorrect labels manually by visualizing the whole dataset. The challenging scenario in this work is defined as the case that movement-to-still happens, where only few events are produced and little information could be used by the model. Precise statistics of the 1 Mpx Auto-
Detection Sub Dataset is shown in Tab. 2.
<div align=center><img src="https://github.com/wds320/AAAI_Event_based_detection/blob/main/subdataset.png" width="50%" /></div>
  
- Download 1 Mpx Auto-Detection Sub Dataset. (Total 268GB)

Links: [https://pan.baidu.com/s/1YawxZFJhQWVgLye9zZtysA](https://pan.baidu.com/s/1YawxZFJhQWVgLye9zZtysA)

Password: c6j9 

- Dataset structure
```
prophesee_dlut   
├── test
│   ├── testfilelist00
│   ├── testfilelist01
│   └── testfilelist02
├── train
│   ├── trainfilelist00
│   ├── trainfilelist01
│   ├── trainfilelist02
│   ├── trainfilelist03
│   ├── trainfilelist04
│   ├── trainfilelist05
│   ├── trainfilelist06
│   ├── trainfilelist07
│   ├── trainfilelist08
│   ├── trainfilelist09
│   ├── trainfilelist10
│   ├── trainfilelist11
│   ├── trainfilelist12
│   ├── trainfilelist13
│   └── trainfilelist14
└── val
    ├── valfilelist00
    └── valfilelist01
```

- Dataset Visualization
```
python data_check_npz.py
```


## Training & Testing
Change settings.yaml, including *dataset_path* and *save_dir*.  
- 1. Training
```
python train_DMANet.py --settings_file=$YOUR_YAML_PATH
```
- 2. Testing
```
python test.py --weight=$YOUR_MODEL_PATH
```
We provided a trained model [here](https://pan.baidu.com/s/1kqfk1gxxqtNHeg75z-EWEg). (Password：6r06)


## Visualization results on 1 Mpx Auto-Detection Sub Dataset
![图片](https://github.com/wds320/AAAI_Event_based_detection/blob/main/case.png)


## Citation
Please cite the following paper if you use this repo in your research:

```bibtex
@inproceedings{dmanet,
  title={Dual Memory Aggregation Network for Event-Based Object Detection with Learnable Representation},
  author={Wang, Dongsheng and Jia, Xu and Zhang, Yang and Zhang, Xinyu and Wang, Yaoyuan and Zhang, Ziyang and Wang, Dong and Lu, Huchuan},  
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2023}
}
```


## Related Repos
- RetinaNet implementation: [https://github.com/yhenon/pytorch-retinanet](https://github.com/yhenon/pytorch-retinanet)
- PointPillars implementation: [https://github.com/SmallMunich/nutonomy_pointpillars](https://github.com/SmallMunich/nutonomy_pointpillars)
- Prophesee's Automotive Dataset Toolbox: [https://github.com/prophesee-ai/prophesee-automotive-dataset-toolbox](https://github.com/prophesee-ai/prophesee-automotive-dataset-toolbox)
- Event-based Asynchronous Sparse Convolutional Networks: [https://github.com/uzh-rpg/rpg_asynet](https://github.com/uzh-rpg/rpg_asynet)

