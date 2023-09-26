'''
    python MLP_Demo.py -s --weight_path ./mlp.wts --engine_path ./mlp.engine
    python MLP_Demo.py -d --weight_path ./mlp.wts --engine_path ./mlp.engine
'''

import os
import struct
import argparse
import numpy as np
import pycuda.autoinit
import tensorrt as trt
import pycuda.driver as cuda

# 日志
gLogger = trt.Logger(trt.Logger.INFO)

class MLP_Demo():
    def __init__(self, weight_path, engine_path):
        super(MLP_Demo, self).__init__()
        self.weight_path = weight_path # weight路径
        self.engine_path = engine_path # engine路径
        self.input_blob_name = 'data' # 输入名称
        self.out_blob_name = 'out' # 输出名称
        self.input_size = 1
        self.output_size = 1
        
    # 序列化模型
    def serialize_model(self, max_batch_size):
        builder = trt.Builder(gLogger) # 创建builder
        config = builder.create_builder_config() # 创建config
        engine = self.create_engine(max_batch_size, builder, config, trt.float32) # 序列化模型
        assert engine

        with open(self.engine_path, "wb") as f: # 保存序列化好的 engine
            f.write(engine.serialize())
            
        # 释放内存
        del engine
        del builder

    def create_engine(self, max_batch_size, builder, config, dt):
        weight_map = self.load_weights(self.weight_path) # 加载模型
        network = builder.create_network() # 先使用 builder 创建一个空的network 
        data = network.add_input(self.input_blob_name, dt, (1, 1, self.input_size))  # 添加输入操作数
        assert data

        # 使用模型的权重来添加线性层
        linear = network.add_fully_connected(input = data, num_outputs = self.output_size,
                                            kernel = weight_map['linear.weight'],
                                            bias = weight_map['linear.bias'])
        assert linear
        
        linear.get_output(0).name = self.out_blob_name # 设置线性层的输出名称
        network.mark_output(linear.get_output(0)) # 将线性层的输出标记为整个网络的输出
        builder.max_batch_size = max_batch_size # 设置最大 batchsize
        engine = builder.build_engine(network, config) # 使用空的network和相应的配置config来创建engine

        # 释放内存
        del network
        del weight_map

        return engine # 返回创建好的engine
    
    # 加载权重数据，以 dict 的形式存放
    def load_weights(self, file_path):
        assert os.path.exists(file_path), '[ERROR]: Unable to load weight file.'

        weight_map = {}
        with open(file_path, "r") as f:
            lines = [line.strip() for line in f]

        count = int(lines[0])
        assert count == len(lines) - 1

        # 循环遍历处理权重数据
        for i in range(1, count + 1):
            splits = lines[i].split(" ")
            name = splits[0]
            cur_count = int(splits[1])
            assert cur_count + 2 == len(splits)

            values = []
            for j in range(2, len(splits)): # 循环遍历处理权重数据
                # bytes.fromhex 将16个字节的字符串转化为字节对象
                # struct.unpack 对字节对象进行解包
                values.append(struct.unpack(">f", bytes.fromhex(splits[j])))

            # 保存格式为 { 'weight.name': [weights_val0, weight_val1, ..] }
            weight_map[name] = np.array(values, dtype=np.float32)
        return weight_map
    
    
    def inference(self, inf_context, inf_host_in, inf_host_out):
        inference_engine = inf_context.engine # 推理引擎
        assert inference_engine.num_bindings == 2 # 输入和输出两个bindings
        device_in = cuda.mem_alloc(inf_host_in.nbytes) # 在GPU中分配输入的内存
        device_out = cuda.mem_alloc(inf_host_out.nbytes) # 在GPU中分配输出的内存

        bindings = [int(device_in), int(device_out)] # 创建输入和输出两个 bindings
        stream = cuda.Stream() # 创建cuda流
        cuda.memcpy_htod_async(device_in, inf_host_in, stream) # 将输入数据从cpu拷贝到gpu中

        # 执行推理
        inf_context.execute_async(bindings=bindings, stream_handle=stream.handle)

        cuda.memcpy_dtoh_async(inf_host_out, device_out, stream) # 将推理结果从gpu拷贝到cpu中
        stream.synchronize() # 同步流，确保所有操作完成
        
    # 反序列化加载序列化的engine，并进行模型推理
    def deserialize_inference(self, input_val):
        runtime = trt.Runtime(gLogger) # 创建runtime
        assert runtime
        
        # 反序列化加载模型，生成engine
        with open(self.engine_path, "rb") as f:
            engine = runtime.deserialize_cuda_engine(f.read())
        assert engine

        # 使用 engine 生成上下文
        context = engine.create_execution_context()
        assert context

        data = np.array([input_val], dtype=np.float32) # 创建输入数据
        host_in = cuda.pagelocked_empty((self.input_size), dtype=np.float32) # 在GPU申请输入的内存缓冲区
        np.copyto(host_in, data.ravel()) # 将输入数据拷贝到内存缓冲区
        host_out = cuda.pagelocked_empty(self.output_size, dtype=np.float32) # 在GPU申请输出的内存缓冲区

        # 执行推理
        self.inference(context, host_in, host_out) # 传入 context 和对应的输入输出缓冲区

        print(f'Input:\t{input_val}\nOutput:\t{host_out[0]:.4f}')

def get_parser():
    arg_parser = argparse.ArgumentParser(description = 'Parameters of MLP Demo')
    arg_parser.add_argument('--weight_path', # 模型的权重数据
                            type = str,
                            default = "./mlp.wts") # engine的保存地址
    arg_parser.add_argument('--engine_path', 
                            type = str,
                            default = "./mlp.engine")
    arg_parser.add_argument('-s', action = 'store_true') # 序列化模型
    arg_parser.add_argument('-d', action = 'store_true') # 推理模型
    return arg_parser

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    mlp_demo = MLP_Demo(args.weight_path, args.engine_path) # 实例化demo对象
    if args.s: 
        mlp_demo.serialize_model(max_batch_size = 1) # 序列化模型
    else: # 反序列化模型进行推理
        mlp_demo.deserialize_inference(input_val = 4.0)

