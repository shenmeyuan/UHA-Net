# UHA-Net
A U-shaped Hybrid Attention system for landmark point detection in cephalometric analysis for Adults and Pediatric Phases

## Abstract

 **Landmark detection on lateral head X-ray im- ages plays a crucial role in understanding anatomical structures and advancing AI-assisted healthcare. In recent years, various deep learning methods have been devel- oped for automatic landmark detection, achieving signifi- cant accomplishments. However, as far as we know, these methods have several limitations, such as being primarily trained on datasets comprising adult head lateral images and suffering from poor expressiveness and generalization due to the heavy reliance on deep learning itself and the lack of diverse data. These issues result in unsatisfac- tory performance of these methods on the increasingly prevalent issue of pediatric cranial measurement analysis. We have innovatively developed a U-shaped hybrid atten- tion system for landmark detection in lateral head X-ray images. Our proposed model consists of two main com- ponents: the primary network and the global network. In the primary network, we extract feature information from multiple subspaces using the innovative U-shaped hybrid attention mechanism, significantly enhancing the model’s expressiveness. In the global network, we further mitigate feature loss by fusing global features through dilated con- volutions. In this research, We have constructed a dataset of lateral head X-ray images from pediatric patients to address the current issue of the data is too narrow in scope. We comprehensively evaluated our model on this dataset and a publicly available competition dataset primar- ily consisting of adult patients. The experiments not only demonstrated the effectiveness of the model we proposed but also showcased its superior generalization capability and performance across adults and pediatric phases.**

## Prerequisites

Linux

NVIDIA GPU

python 3.10.9

pytorch 2.0.0

pytorch-cuda 11.7()



## Getting Started

### install python packages

```
pip3 install -r requirements.txt
```

### Preparing Datasets

Prepare datasets in the following directory structure.

* data 
  * ceph 
    * junior
    * senior
    * raw
      * \*.bmp

* landmark\_detection  # working directory

Now , `cd landmark_detection`.

### Usage

```bash
usage: main.py [-h] [-C CONFIG] [-c CHECKPOINT] [-g CUDA_DEVICES] [-m MODEL]
               [-l LOCALNET] [-n NAME_LIST [NAME_LIST ...]] [-e EPOCHS]
               [-L LR] [-w WEIGHT_DECAY] [-s SIGMA] [-x MIX_STEP] [-u] -r
               RUN_NAME -d RUN_DIR -p {train,validate,test}

optional arguments:
  -h, --help            show this help message and exit
  -C CONFIG, --config CONFIG
  -c CHECKPOINT, --checkpoint CHECKPOINT
                        checkpoint path
  -g CUDA_DEVICES, --cuda_devices CUDA_DEVICES
  -m MODEL, --model MODEL
  -l LOCALNET, --localNet LOCALNET
  -n NAME_LIST [NAME_LIST ...], --name_list NAME_LIST [NAME_LIST ...]
  -e EPOCHS, --epochs EPOCHS
  -L LR, --lr LR
  -w WEIGHT_DECAY, --weight_decay WEIGHT_DECAY
  -s SIGMA, --sigma SIGMA
  -x MIX_STEP, --mix_step MIX_STEP
  -u, --use_background_channel
  -r RUN_NAME, --run_name RUN_NAME
  -d RUN_DIR, --run_dir RUN_DIR
  -p {train,validate,test}, --phase {train,validate,test}
```

### Train 

- Train our UHA-Net model

```bash
python3 main.py -d ../runs -r UHANet_runs -p train -m gln -l uhanet -e 100
```

- Loading checkpoint

```bash
python3 main.py -d ../runs -r UHANet_runs -p train -m gln -l uhanet -e 100 -c CHECKPOINT_PATH
```

This running results are in the following directory structure.

* ../runs 
  * UHANet\_runs
    * network_graph.txt
    * config_train.yaml
    * config_origin.yaml
    * learning_rate.png
    * loss.png
    * checkpoints
      * best\_UHANet\_runs\_\*.pt
      * results
        * train_epoch
        * test_epoch

### Test

After training, it will automatically run the tests.

Yet you could manually run the tests:

```bash
python3 main.py -d ../runs -r UHANet_runs -p test -m gln -l uhanet -c CHECKPOINT_PATH
```

### Evaluation

```bash
python3 evaluation.py -i ../runs/UHANet_runs/results/test_epochxxx
```

## Acknowledgments

The yamlConfig.py is modified from [adn](https://github.com/liaohaofu/adn) and uhanet.py is modified from [YOLO_Universal_Anatomical_Landmark_Detection](https://github.com/MIRACLE-Center/YOLO_Universal_Anatomical_Landmark_Detection)

If you need the dataset, you can contact the following email address: Feng Chen(fengchenmit@swu.edu.cn);Guangyuan Zhang(1696706849@qq.com)
