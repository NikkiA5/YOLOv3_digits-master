# YOLOv3 digits

A YOLOv3 based digits detector. Based on the amazing work of [qqwweee](https://github.com/qqwweee/keras-yolo3)

**! THIS IS A WORK IN PROGRESS, MODEL AVAILABLE SOON**

## Dataset

The [Street View House Numbers (SVHN)](http://ufldl.stanford.edu/housenumbers/) dataset is used for transfer learning.  

![SVHN example images](http://ufldl.stanford.edu/housenumbers/examples_new.png)

From the website we see that there are : "*73257 digits for training, 26032 digits for testing, and 531131 additional, somewhat less difficult samples, to use as extra training data*"

Annotations PASCAL format: https://github.com/penny4860/svhn-voc-annotation-format

Annotations PASCAL format AND txt format (ready for training) http://download939.mediafire.com/ouwinissq9ng/ydg1i5qbm27mdax/annotations.tar.xz

    cd <path_to_repo_root>/data/
    wget http://download939.mediafire.com/ouwinissq9ng/ydg1i5qbm27mdax/annotations.tar.xz
    tar -xf annotations.tar.xz
    rm annotations.tar.xz

## Introduction
 
---

## Quick Start

1. Download YOLOv3 weights from [YOLO website](http://pjreddie.com/darknet/yolo/).
2. Convert the Darknet YOLO model to a Keras model.
3. Run YOLO detection.

```
wget https://pjreddie.com/media/files/yolov3.weights
python convert.py yolov3.cfg yolov3.weights model_data/yolo.h5
python yolo_video.py [OPTIONS...] --image, for image detection mode, OR
python yolo_video.py [video_path] [output_path (optional)]
```

For Tiny YOLOv3, just do in a similar way, just specify model path and anchor path with `--model model_file` and `--anchors anchor_file`.

### Usage
Use --help to see usage of yolo_video.py:
```
usage: yolo_video.py [-h] [--model MODEL] [--anchors ANCHORS]
                     [--classes CLASSES] [--gpu_num GPU_NUM] [--image]
                     [--input] [--output]

positional arguments:
  --input        Video input path
  --output       Video output path

optional arguments:
  -h, --help         show this help message and exit
  --model MODEL      path to model weight file, default model_data/yolo.h5
  --anchors ANCHORS  path to anchor definitions, default
                     model_data/yolo_anchors.txt
  --classes CLASSES  path to class definitions, default
                     model_data/coco_classes.txt
  --gpu_num GPU_NUM  Number of GPU to use, default 1
  --image            Image detection mode, will ignore all positional arguments
```

---

4. MultiGPU usage: use `--gpu_num N` to use N GPUs. It is passed to the [Keras multi_gpu_model()](https://keras.io/utils/#multi_gpu_model).

## Training

1. Generate your own annotation file and class names file.  
    One row for one image;  
    Row format: `image_file_path box1 box2 ... boxN`;  
    Box format: `x_min,y_min,x_max,y_max,class_id` (no space).  
    For VOC dataset, try `python voc_annotation.py`  
    Here is an example:
    ```
    path/to/img1.jpg 50,100,150,200,0 30,50,200,120,3
    path/to/img2.jpg 120,300,250,600,2
    ...
    ```

2. Make sure you have run `python convert.py -w yolov3.cfg yolov3.weights model_data/yolo_weights.h5`  
    The file model_data/yolo_weights.h5 is used to load pretrained weights.

3. Modify train.py and start training.  
    `python train.py`  
    Use your trained weights or checkpoint weights with command line option `--model model_file` when using yolo_video.py
    Remember to modify class path or anchor path, with `--classes class_file` and `--anchors anchor_file`.

If you want to use original pretrained weights for YOLOv3:  
    1. `wget https://pjreddie.com/media/files/darknet53.conv.74`  
    2. rename it as darknet53.weights  
    3. `python convert.py -w darknet53.cfg darknet53.weights model_data/darknet53_weights.h5`  
    4. use model_data/darknet53_weights.h5 in train.py

---

## TODO

- [X] Convert .mat file to txt for YOLO
- [] Transfer learning on digits
- [] Add mAP to the projects for evaluation
- [] Prune the network
- [] Quantize the network
- [] Convert to tf