# Description
A demo about how to serialize the onnx model and deserialize the engine.
# Build and Run
## 1. Export Onnx model
```
cd model
python Export_linear.py
```
## 2. Test python demo
```
cd ..
python Infer_Onnx.py
```
## 3. Test CPP demo
```
mkdir build
cd build
cmake ..
make
./Serialize_onnx
./Deserialize
```
# Test Result
## Export_linear.py
<div align=center>
<img src="https://github.com/liujf69/TensorRT-Demo/blob/master/TRT_Serialize_and_Deserialize/export_linear.png"/>
</div>

## Infer_Onnx.py
<div align=center>
<img src="https://github.com/liujf69/TensorRT-Demo/blob/master/TRT_Serialize_and_Deserialize/infer_Onnx.png"/>
</div>

## Deserialize.cpp
<div align=center>
<img src="https://github.com/liujf69/TensorRT-Demo/blob/master/TRT_Serialize_and_Deserialize/Deserialize.png"/>
</div>

