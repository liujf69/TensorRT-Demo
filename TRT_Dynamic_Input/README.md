# Export Onnx Model
```python Export_Onnx.py --d Dynamics_InputNet.onnx --s Static_InputNet.onnx --b 8 --c 3 --h 256 --w 256```  <br />
# Infer Dynamic Input
```python Infer_Onnx.py --b 4 --c 3 --h 256 --w 256 --onnx ./model/Dynamics_InputNet.onnx```
