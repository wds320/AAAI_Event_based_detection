# Dual Memory Aggregation Network for Event-based Object Detection with Learnable Representation
The code will release soon!


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
 
Since the groundtruth of the [Prophesee dataset](https://www.prophesee.ai/2020/11/24/automotive-megapixel-event-based-dataset/) is obtained automatically by applying an RGB-based detector in the stereo recording setting, there are some geometric errros such as misalignment and semantic errors caused by wrong detection of the detector with the ground truth. We also found that there are some $mosaic$ events in this dataset which might be caused by flickering as explained by authors of that dataset. Therefore, we remove theses events and incorrect labels manually by visualizing the whole dataset. The challenging scenario in this work is defined as the case that movement-to-still happens, where only few events are produced and little information could be used by the model. Precise statistics of the 1 Mpx Auto-
Detection Sub Dataset is shown in Tab. 2.
<div align=center><img src="https://github.com/wds320/AAAI_Event_based_detection/blob/main/subdataset.png" width="50%" /></div>
  
- Download 1 Mpx Auto-Detection Sub Dataset. (Total 268GB)
```
Links: [https://pan.baidu.com/s/1YawxZFJhQWVgLye9zZtysA](https://pan.baidu.com/s/1YawxZFJhQWVgLye9zZtysA)
Password: c6j9 
```


## Visualization results on 1 Mpx Auto-Detection Sub Dataset
![图片](https://github.com/wds320/AAAI_Event_based_detection/blob/main/case.png)


## Citation



## Acknowledgements
