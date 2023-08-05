# Download Weights
1. Download weights (i.e. YOLOv8s, YOLOv8s-seg, ..) from [ultralytics](https://github.com/ultralytics/yolov5). <br />
2. ```cd yolov5``` and put weights into the weights folder. <br />
3. Run ```python gen_wts.py -w ./weights/yolov5s.pt -o ./weights/yolov5s.wts``` to get ```.wts``` weight file.

# Compile
```
mkdir build
cd build
cmake .. # you should modify the CMakeListst.txt to set dependencies (i.e. TensorRT, OpenCV ..)
make ..
```

# Test
## Detection
1. Serialize model to generate the engine file, run ```./yolov5_det -s ../yolov5/weights/yolov5s.wts ./yolov5s.engine s```<br />
2. Infer images, run ```./yolov5_det -d yolov5s.engine -i ../images```<br />
3. Infer video, run ```./yolov5_det -d yolov5s.engine -v ../videos/test1.avi```<br />
4. Infer camera, run ```./yolov5_det -d yolov5s.engine -c 0```<br />
![image](https://github.com/liujf69/TensorRT-Demo/blob/master/TRT_YoloV5/build/det_test1.jpg)

## Segmentation
1. Serialize model to generate the engine file, run ```./yolov5_seg -s ../yolov5/weights/yolov5s-seg.wts ./yolov5s-seg.engine s```<br />
2. Run ```wget -O coco.txt https://raw.githubusercontent.com/amikelive/coco-labels/master/coco-labels-2014_2017.txt``` <br />
3. Infer images, run ```./yolov5_seg -d yolov5s-seg.engine -i ../images coco.txt```<br />
4. Infer video, run ```./yolov5_seg -d yolov5s-seg.engine -v ../videos/test1.avi coco.txt```<br />
5. Infer camera, run ```./yolov5_seg -d yolov5s-seg.engine -c 0 coco.txt```<br />
![image](https://github.com/liujf69/TensorRT-Demo/blob/master/TRT_YoloV5/build/seg_test1.jpg)

# Reference
Our project is based on [YoloV5](https://github.com/ultralytics/yolov5) and [tensorrtx](https://github.com/wang-xinyu/tensorrtx). <br />
