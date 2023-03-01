import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.optim as optim
import imageio
from layer import MergedConv2d
from layer import MaxPooling2d
from layer import FC
from layer import ReLULayer
from imageStar import ImageStar
import plot


if __name__ == '__main__':
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 6, 5)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(6, 16, 5)
            self.fc1 = nn.Linear(16 * 5 * 5, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 10)

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = torch.flatten(x, 1)  # flatten all dimensions except batch
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    result_path1 = './cifar_net3.pth'
    result_path2 = './cifar_net4.pth'
    # net = Net()
    # net.load_state_dict(torch.load(result_path))
    net1 = torch.load(result_path1)
    net2 = torch.load(result_path2)

    conv1_weight1 = net1['conv1.weight'].to(torch.float64)
    conv1_bias1 = net1['conv1.bias'].to(torch.float64)
    conv2_weight1 = net1['conv2.weight'].to(torch.float64)
    conv2_bias1 = net1['conv2.bias'].to(torch.float64)
    fc1_weight1 = net1['fc1.weight'].to(torch.float64)
    fc1_bias1 = net1['fc1.bias'].to(torch.float64)
    fc2_weight1 = net1['fc2.weight'].to(torch.float64)
    fc2_bias1 = net1['fc2.bias'].to(torch.float64)
    fc3_weight1 = net1['fc3.weight'].to(torch.float64)
    fc3_bias1 = net1['fc3.bias'].to(torch.float64)

    conv1_weight2 = net2['conv1.weight'].to(torch.float64)
    conv1_bias2 = net2['conv1.bias'].to(torch.float64)
    conv2_weight2 = net2['conv2.weight'].to(torch.float64)
    conv2_bias2 = net2['conv2.bias'].to(torch.float64)
    fc1_weight2 = net2['fc1.weight'].to(torch.float64)
    fc1_bias2 = net2['fc1.bias'].to(torch.float64)
    fc2_weight2 = net2['fc2.weight'].to(torch.float64)
    fc2_bias2 = net2['fc2.bias'].to(torch.float64)
    fc3_weight2 = net2['fc3.weight'].to(torch.float64)
    fc3_bias2 = net2['fc3.bias'].to(torch.float64)

    IM = imageio.v2.imread('C:/Users/emiya/Desktop/PhD Research/Research 1'
                           '/CIFAR-10-images-master/test/airplane/0385.jpg')
    # IM = imageio.v2.imread('D:/Cody/phd/research1/CIFAR-10-images/test/airplane/0001.jpg')
    # print(IM.dtype)
    IM = IM/255
    IM = (IM - 0.5)/0.5
    LB = np.zeros((32, 32, 3))
    UB = np.zeros((32, 32, 3))
    # LB[0, 0, 0] = -0.05
    # UB[0, 0, 0] = 0.05
    # LB[10, 0, 0] = -0.03
    # UB[10, 0, 0] = 0.03
    IM_m = np.concatenate((IM, IM))
    LB_m = np.concatenate((LB, LB))
    UB_m = np.concatenate((UB, UB))
    #I1 = ImageStar(IM, LB, UB)
    I1_m = ImageStar(IM_m, LB_m, UB_m)

    # print(type(fc1_weight1))
    # print(isinstance(fc1_weight1, torch.Tensor))
    # print(fc1_weight1.shape)
    # print(fc1_bias1.shape)
    merge_layer = MergedConv2d(conv1_weight1, conv1_bias1, conv1_weight2, conv1_bias2)
    Istar_1 = merge_layer.reach(I1_m)

    relu_layer = ReLULayer()
    method = 'exact-star'
    method2 = 'approx-star'
    Istar_2 = relu_layer.reach(Istar_1, method)

    poolsize = np.array([2, 2])
    stride = np.array([2, 2])
    padding = np.array([0, 0])
    maxpooling_layer1 = MaxPooling2d(poolsize, stride, padding, "maxpool1")
    Istar_3 = maxpooling_layer1.reach(Istar_2, method)

    merge_layer2 = MergedConv2d(conv2_weight1, conv2_bias1, conv2_weight2, conv2_bias2)
    Istar_4 = merge_layer2.reach(Istar_3)

    Istar_5 = relu_layer.reach(Istar_4, method)

    maxpooling_layer2 = MaxPooling2d(poolsize, stride, padding, "maxpool2")
    Istar_6 = maxpooling_layer2.reach(Istar_5, method)

    fc1 = FC(fc1_weight1, fc1_bias1, fc1_weight2, fc1_bias2)
    Istar_7 = fc1.reach(Istar_6)

    Istar_8 = relu_layer.reach(Istar_7, method)

    fc2 = FC(fc2_weight1, fc2_bias1, fc2_weight2, fc2_bias2)
    Istar_9 = fc2.reach(Istar_8)

    Istar_10 = relu_layer.reach(Istar_9, method)

    fc3 = FC(fc3_weight1, fc3_bias1, fc3_weight2, fc3_bias2)
    Istar_11 = fc3.reach(Istar_10)

    fc4_weight1 = np.eye(10)
    fc4_weight2 = -np.eye(10)
    b1 = np.array([0.5, 0.2])
    b2 = np.array([-0.5, 0.1])
    fc4 = FC(fc4_weight1, b1, fc4_weight2, b2, 'last')
    Istar_12 = fc4.reach(Istar_11)
    # print(Istar_10.C)
    # print(Istar_10.d)

    print(Istar_11[0].V[0,0,0:10,0])
    print(Istar_11[0].V[0,0,10:,0])
    print(Istar_12[0].V[0,0,:,0])

    # n = len(Istar_10)
    # for i in range(n):
    #     Star1 = Istar_10[i].toStar()
    #     vertices = Star1.toPolyhedron()
    #     print("Set %d: \n" % i)
    #     print(vertices)
    #     print("\n")
    #     max_dis = 0
    #     for j in range(vertices.shape[0]):
    #         dis = np.linalg.norm(vertices[j,:])
    #         if dis > max_dis:
    #             max_dis = dis
    #     print("Max distance in Set %d is %.6f" % (i, max_dis))

