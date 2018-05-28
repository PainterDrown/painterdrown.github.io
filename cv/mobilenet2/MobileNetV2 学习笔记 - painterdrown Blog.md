[painterdrown Blog](https://painterdrown.github.io) - [painterdrown CV](https://painterdrown.github.io/cv)

# MobileNetV2 - Notes

## 1. Introduction

意思大概是，目前最前沿的用于目标检测的神经网络模型，都需要大量的计算资源，因此不适用于移动设备和嵌入式设备。MobileNets 应运而生，有三个特点：

1. 无需大量的计算
2. 无需大量的内存开销
3. 保持着良好的检测精度
MobileNetV2 的主要创新之处是在神经网络层：the inverted residual with linear bottleneck（线性瓶颈的反向残差结构）。我在 [CSDN 找到一篇论文](https://blog.csdn.net/u011995719/article/details/79135818)是这样解释的：

> 通常的 residuals block 是先经过一个 1\*1 的 Conv layer，把 feature map 的通道数“压”下来，再经过 3\*3 Conv layer，最后经过一个 1\*1 的 Conv layer，将 feature map 通道数再“扩张”回去。即先“压缩”，最后“扩张”回去。而 inverted residuals就是 先“扩张”，最后“压缩”。

> Linear bottlenecks，为了避免 relu 对特征的破坏（我们知道 relu 函数只取输入的非负值，且认为这个行为导致了信息损失），在 residual block 的 Eltwise sum 之前的那个 1\*1 Conv 不再采用 relu。
