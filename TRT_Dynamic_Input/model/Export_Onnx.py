#python Export_Onnx.py --d Dynamics_InputNet.onnx --s Static_InputNet.onnx --b 8 --c 3 --h 256 --w 256

import torch
import argparse
import torch.nn as nn

def get_parser():
    parser = argparse.ArgumentParser(
        description='Parameters of Export Onnx Demo')
    parser.add_argument(
        '--d',
        type=str,
        default='./Dynamics_InputNet.onnx')
    parser.add_argument(
        '--s',
        type=str,
        default='./Static_InputNet.onnx')
    parser.add_argument(
        '--b',
        type=int,
        default=8)
    parser.add_argument(
        '--c',
        type=int,
        default=3)
    parser.add_argument(
        '--h',
        type=int,
        default=256)
    parser.add_argument(
        '--w',
        type=int,
        default=256)
    return parser


class Model_Net(nn.Module):
    def __init__(self):
        super(Model_Net, self).__init__()
        self.layer1 = nn.Sequential(

            nn.Conv2d(in_channels=3, out_channels=64,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=64, out_channels=256,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

    def forward(self, data):
        data = self.layer1(data)
        return data


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    # 设置输入参数
    Batch_size = args.b
    Channel = args.c
    Height = args.h
    Width = args.w
    input_data = torch.rand((Batch_size, Channel, Height, Width))

    # 实例化模型
    model = Model_Net()

    # 导出为静态输入
    input_name = 'input'
    output_name = 'output'
    torch.onnx.export(model,
                      input_data,
                      args.s,
                      verbose=True,
                      input_names=[input_name],
                      output_names=[output_name])

    # 导出为动态输入
    torch.onnx.export(model,
                      input_data,
                      args.d,
                      opset_version=11,
                      input_names=[input_name],
                      output_names=[output_name],
                      dynamic_axes={
                          input_name: {0: 'batch_size', 2: 'input_height', 3: 'input_width'},
                          output_name: {0: 'batch_size', 2: 'output_height', 3: 'output_width'}})
