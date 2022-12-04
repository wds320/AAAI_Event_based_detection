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



## Visualization results on 1 Mpx Auto-Detection Sub Dataset
![图片](https://github.com/wds320/AAAI_Event_based_detection/blob/main/case.png)



## Citation



## Acknowledgements
