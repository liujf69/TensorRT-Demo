import torch
import struct
import torchvision
from torchsummary import summary

if __name__ == '__main__':
    net = torchvision.models.vgg11(pretrained=True)
    net = net.eval()
    print(net)
    net = net.to('cuda:0')
    tmp = torch.ones(1, 3, 224, 224).to('cuda:0')
    out = net(tmp)
    print('out_shape:', out.shape)
    for i in range(10):
        print('out_data:', out[0,i])
    
    summary(net, (3, 224, 224))
    f = open("./vgg.wts", 'w')
    f.write("{}\n".format(len(net.state_dict().keys()))) # 权重的个数
    for k,v in net.state_dict().items():
        print('key: ', k)
        print('value: ', v.shape)
        vr = v.reshape(-1).cpu().numpy() # 展开为一维
        f.write("{} {}".format(k, len(vr)))
        for vv in vr:
            f.write(" ")
            f.write(struct.pack(">f", float(vv)).hex())
        f.write("\n")