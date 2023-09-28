# Description
A demo about showing TensorRT inference process.

# Export Onnx Model
```
cd model
python Export_Onnx.py --d Dynamics_InputNet.onnx --s Static_InputNet.onnx --b 8 --c 3 --h 256 --w 256
```
# Infer Dynamic Input
## Python Demo
```python Infer_Onnx.py --b 2 --c 3 --h 256 --w 256 --onnx ./model/Dynamics_InputNet.onnx``` <br />

For more introduction about the project, please refer to the personal [study notes](https://blog.csdn.net/weixin_43863869/article/details/128651343?csdn_share_tail=%7B%22type%22%3A%22blog%22%2C%22rType%22%3A%22article%22%2C%22rId%22%3A%22128651343%22%2C%22source%22%3A%22weixin_43863869%22%7D)

## CPP Demo
1. Build project, run
```
mkdir build
cd build
cmake ..
make
```
2. Serialize onnx model, run
```
./Infer_Onnx -s ../model/Dynamics_InputNet.onnx ./Dynamics_InputNet.engine
```
3. Deserialize and infer, run
```
./Infer_Onnx -d ./Dynamics_InputNet.engine
```
# Comparison


