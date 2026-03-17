# Unitree Grasp

## Installation

### 1. Prepare Environment
```bash
git clone --recursive https://github.com/alter-c/unitree-grasp.git
```
```bash
conda create -n grasp python=3.8 casadi=3.6.5 pinocchio=3.2.0 -c conda-forge
conda activate grasp

cd third_party/unitree_sdk2_python
pip3 install -e .

cd ../..
pip3 install -r requirements.txt
```
### 2. Torch Package
[PyTorch for Jetson](https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048)
```bash
pip3 uninstall torch torchvision

# PyTorch for Jetson (https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048)
pip3 install torch-2.1.0a0+41361538.nv23.06-cp38-cp38-linux_aarch64.whl

# Torchvision 
sudo apt-get install libjpeg-dev zlib1g-dev libpython3-dev libopenblas-dev libavcodec-dev libavformat-dev libswscale-dev
git clone --branch v0.16.1 https://github.com/pytorch/vision torchvision
cd torchvision
export BUILD_VERSION=0.16.1  
python3 setup.py install --user
cd ../ # for test
```

## Usage
### Quick Start
In the terminal and execute:
```bash
python demo_api.py
```
Then open a new terminal and you can execute below commands:
```bash
curl '0.0.0.0:8080/api/unitree/grasp?target=bottle' # grasp action

curl '0.0.0.0:8080/api/unitree/handover' # handover object

curl '0.0.0.0:8080/api/unitree/stop' # stop current action and release arm to walk
```


## FAQ
+ cyclonedds bug：[FAQ](https://github.com/unitreerobotics/unitree_sdk2_python?tab=readme-ov-file#faq)



