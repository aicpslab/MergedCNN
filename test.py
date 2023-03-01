import numpy as np
from sklearn import preprocessing
from layer import MergedConv2d
from layer import MaxPooling2d
from layer import FC
from layer import ReLULayer
from imageStar import ImageStar
import torch
import torch.nn as nn
import torch.nn.functional as F
import plot
import matplotlib.pyplot as plt

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 2, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(2, 4, 2)
        self.fc1 = nn.Linear(4, 2)
        # self.fc2 = nn.Linear(120, 84)
        # self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(self.conv1(x))
        x = self.conv2(x)
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        # x = self.fc3(x)
        return x

if __name__ == '__main__':
    IM = np.array([[1, 1, 0, 1, 0, 0], [0, 1, 0, 0, 1, 0], [1, 0, 1, 0, 1, 0], [0, 1, 1, 1, 0, 0], [1, 0, 0, 0, 1, 0],
                   [0, 0, 0, 0, 0, 0]])
    # np.random.seed(203)
    # IM = np.random.random((6, 6))
    # LB = np.array([[-0.1, 0, 0, 0, 0, 0], [0, -0.05, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, -0.2, 0], [0, 0, 0, 0, 0, 0],
    #                [0, 0, 0, 0, 0, 0]])
    # UB = np.array([[0.1, 0, 0, 0, 0, 0], [0, 0.2, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0.05, 0], [0, 0, 0, 0, 0, 0],
    #                [0, 0, 0, 0, 0, 0]])

    LB = np.zeros((6,6))
    UB = np.zeros((6,6))
    IM_m = np.concatenate((IM, IM))
    LB_m = np.concatenate((LB, LB))
    UB_m = np.concatenate((UB, UB))
    I1 = ImageStar(IM, LB, UB)
    I1_m = ImageStar(IM_m, LB_m, UB_m)

    k1 = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
    k2 = np.array([[0., 0., 1.], [0., 1., 0.], [1., 0., 0.]])
    filter1 = np.dstack((k1, k1))
    filter2 = np.dstack((k2, k2))
    filter11 = filter1[:, :, np.newaxis, :]
    filter22 = filter2[:, :, np.newaxis, :]
    bias1 = np.array([[0.1, 0.2]])
    bias2 = np.array([[-0.2, 0.1]])

    merge_layer = MergedConv2d(filter11, bias1, filter22, bias2)
    Istar_1 = merge_layer.reach(I1_m)

    poolsize = np.array([2, 2])
    stride = np.array([2, 2])
    padding = np.array([0, 0])
    maxpooling_layer = MaxPooling2d(poolsize, stride, padding, 'maxp1')

    method = 'exact-star'
    method2 = 'approx-star'
    Istar_2 = maxpooling_layer.reach(Istar_1, method)

    print(Istar_2[0].V[:,:,0,0])
    print(Istar_2[0].V[:,:,1,0])

    k3 = np.array([[1., 0.], [0., 1.]])
    k4 = np.array([[0., 1.], [1., 0.]])
    filter3 = np.dstack((k3, k3, k3, k3))
    filter4 = np.dstack((k4, k4, k4, k4))
    filter33 = np.concatenate((filter3[:, :, np.newaxis, :], filter3[:, :, np.newaxis, :]), axis=2)
    filter44 = np.concatenate((filter4[:, :, np.newaxis, :], filter4[:, :, np.newaxis, :]), axis=2)
    bias3 = np.array([[0.1, 0.2, 0.3, -0.1]])
    bias4 = np.array([[-0.1, -0.2, 0.1, -0.3]])

    merge_layer2 = MergedConv2d(filter33, bias3, filter44, bias4)
    Istar_3 = merge_layer2.reach(Istar_2)

    print(Istar_3[0].V[:,:,1,0])

    f1 = np.array([[0.1, 0.2, 0.3, 0.4], [0.2, 0.1, 0.2, 0.1]])
    f2 = np.array([[-0.1, 0.2, -0.3, 0.4], [0.1, 0.1, 0.2, 0.2]])
    b1 = np.array([0.5, 0.2])
    b2 = np.array([-0.5, 0.1])
    fc1 = FC(f1, b1, f2, b2)
    Istar_4 = fc1.reach(Istar_3)

    print(Istar_4[0].V[0,0,:,0])

    relu_layer = ReLULayer('ReLU')
    method = 'exact-star'
    method2 = 'approx-star'
    Istar_5 = relu_layer.reach(Istar_4, method)

    print(Istar_5[0].V[0,0,:,0])

    f3 = np.eye(2)
    f4 = -np.eye(2)
    fc2 = FC(f3, b1, f4, b2, 'last')
    Istar_6 = fc2.reach(Istar_5)

    # print(Istar_6[0].V[0,0,:,0])

    # Star_1 = Istar_6.toStar()
    # len_x = Star_1.V.shape[0]
    # fig = plt.figure(1)
    # for i in range(len_x):
    #     for j in range(len_x):
    #         plot.plot_2d_box(Star_1, i, j)
    # plt.show()

    net1 = Net()
    net1.conv1.weight = nn.Parameter(torch.from_numpy(filter11.transpose(3,2,0,1)), requires_grad=False)
    net1.conv1.bias = nn.Parameter(torch.from_numpy(bias1.flatten()), requires_grad=False)
    net1.conv2.weight = nn.Parameter(torch.from_numpy(filter33.transpose(3,2,0,1)), requires_grad=False)
    net1.conv2.bias = nn.Parameter(torch.from_numpy(bias3.flatten()), requires_grad=False)
    net1.fc1.weight = nn.Parameter(torch.from_numpy(f1), requires_grad=False)
    net1.fc1.bias = nn.Parameter(torch.from_numpy(b1), requires_grad=False)

    IM_sample = IM[:,:,np.newaxis]
    output = net1(torch.from_numpy(IM_sample.transpose(2, 0, 1)).unsqueeze_(0).to(torch.float64))
    print(output)

    net2 = Net()
    net2.conv1.weight = nn.Parameter(torch.from_numpy(filter22.transpose(3, 2, 0, 1)), requires_grad=False)
    net2.conv1.bias = nn.Parameter(torch.from_numpy(bias2.flatten()), requires_grad=False)
    net2.conv2.weight = nn.Parameter(torch.from_numpy(filter44.transpose(3, 2, 0, 1)), requires_grad=False)
    net2.conv2.bias = nn.Parameter(torch.from_numpy(bias4.flatten()), requires_grad=False)
    net2.fc1.weight = nn.Parameter(torch.from_numpy(f2), requires_grad=False)
    net2.fc1.bias = nn.Parameter(torch.from_numpy(b2), requires_grad=False)
    output1 = net2(torch.from_numpy(IM_sample.transpose(2, 0, 1)).unsqueeze_(0).to(torch.float64))
    print(output1)