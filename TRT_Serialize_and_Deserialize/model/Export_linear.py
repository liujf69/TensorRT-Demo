import torch
import argparse
import torch.nn as nn

class Model_Net(nn.Module):
    def __init__(self):
        super(Model_Net, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(1, 1)
        )

    def forward(self, data):
        data = self.layer1(data)
        return data
    
if __name__ == "__main__":
    
    # 实例化模型
    model = Model_Net()
    
    input_data = torch.ones((3, 1))
    output_data = model(input_data)
    print(output_data)
    
    # 导出为动态输入
    input_name = 'input'
    output_name = 'output'
    torch.onnx.export(model,
                      input_data,
                      './Dynamics_linear.onnx',
                      opset_version=11,
                      input_names=[input_name],
                      output_names=[output_name],
                      dynamic_axes={
                          input_name: {0: 'batch_size', },
                          output_name: {0: 'batch_size',}})