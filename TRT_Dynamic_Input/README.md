# Export Onnx Model
```
cd model
python Export_Onnx.py --d Dynamics_InputNet.onnx --s Static_InputNet.onnx --b 8 --c 3 --h 256 --w 256
```
# Infer Dynamic Input
```python Infer_Onnx.py --b 4 --c 3 --h 256 --w 256 --onnx ./model/Dynamics_InputNet.onnx``` <br />

For more introduction about the project, please refer to the personal [study notes](https://blog.csdn.net/weixin_43863869/article/details/128651343?csdn_share_tail=%7B%22type%22%3A%22blog%22%2C%22rType%22%3A%22article%22%2C%22rId%22%3A%22128651343%22%2C%22source%22%3A%22weixin_43863869%22%7D)

