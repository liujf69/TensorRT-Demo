# Download Weight
1. Download weights (i.e. YOLOv8n, YOLOv8s, ..) from [ultralytics](https://github.com/ultralytics/ultralytics)<br />
2. ```cd yolov8``` and put weights into the weights folder <br />
3. Run ```python gen_wts.py``` to get ```.wts``` weight file

# Compile
```
mkdir build
cd build
cmake .. # you should modify the CMakeListst.txt to set dependencies (i.e. TensorRT, OpenCV ..)
make ..
```

# Test
1. Serialize model to generate the engine file, run ```./yolov8 -s ../yolov8/weights/yolov8s.wts ./yolov8s.engine s```<br />
2. Infer images, run ```./yolov8 -d ./yolov8s.engine -i ../images g```<br />
3. Infer video, run ```./yolov8 -d ./yolov8s.engine -v ../videos/test1.avi g```<br />
![image](https://github.com/liujf69/TensorRT-Demo/blob/master/TRT_YoloV8/build/_test1.jpg)

# Reference
Our project is based on [YoloV8](https://github.com/ultralytics/ultralytics) and [tensorrtx](https://github.com/wang-xinyu/tensorrtx) <br />
