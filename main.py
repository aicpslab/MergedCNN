import numpy as np
from sklearn import preprocessing
from layer import MergedConv2d
from layer import MaxPooling2d
from layer import FC
from layer import ReLULayer
from imageStar import ImageStar
import torch
import torch.nn as nn


# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    IM = np.array([[1, 1, 0, 1], [0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 1, 1]])
    LB = np.array([[-0.1, -0.2, 0, 0], [0, 0, 0, -0.05], [0, 0, 0, 0], [0, 0, 0, 0]])
    UB = np.array([[0.1, 0.2, 0, 0], [0, 0, 0, 0.05], [0, 0, 0, 0], [0, 0, 0, 0]])
    # print(IM.reshape((16,1)))

    IM_m = np.concatenate((IM, IM))
    LB_m = np.concatenate((LB, LB))
    UB_m = np.concatenate((UB, UB))
    # a = [[0.1],[0.2],[0],[0],[0],[0],[0],[0.05],[0],[0],[0],[0],[0],[0],[0],[0]]
    # print(np.linalg.norm(a))
    I1 = ImageStar(IM, LB, UB)

    I1_m = ImageStar(IM_m, LB_m, UB_m)
    # print(I1.V[:, :, 0, 0])
    # print(I1.C)
    # print(I1.pred_lb)
    # print(I1.pred_ub)
    # print(I1.numPred)
    # print(I1.im_lb)
    # print(I1.im_ub)

    filter1 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    filter11 = filter1[:, :, np.newaxis, np.newaxis]
    filter2 = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])
    filter22 = filter2[:, :, np.newaxis, np.newaxis]
    bias1 = np.array([[0.1]])
    bias2 = np.array([[-0.1]])

    # startPoints = [[] for i in range(2)]
    # startPoints[0].append(np.zeros(2))
    # print(startPoints[0])
    # print(startPoints[0][0])
    # print(startPoints[0][0].shape)

    # conv1 = nn.Conv2d(1, 1, 3)
    # conv1.requires_grad = False
    # print(torch.from_numpy(filter1.transpose(2, 3, 0, 1)).dtype)
    # conv1.weight.requires_grad = False
    # conv1.weight = nn.Parameter(torch.from_numpy(filter1.transpose(2, 3, 0, 1)),requires_grad=False)
    # conv1.bias = nn.Parameter(torch.from_numpy(bias1).squeeze(), requires_grad=False)
    # a1 = torch.from_numpy(bias1.flatten())
    # print(a1.shape)
    # print(conv1.weight.dtype)

    merge_layer = MergedConv2d(filter11, bias1, filter22, bias2)
    Istar_1 = merge_layer.merged_reach(I1_m)

    # Istar1_1, Istar1_2 = merge_layer.merged_reach_test(I1, I1)
    #
    # pred1 = 3
    # pred2 = 6
    # print(Istar_1.V[:,:,0,pred2])
    # print(Istar1_1.V[:,:,0,pred1])
    # print(Istar_1.V[:,:,0,pred2])
    # print(Istar1_2.V[:,:,0,pred1])
    #
    #
    # print(Istar1_2.getRanges())

    poolsize = np.array([2, 2])
    stride = np.array([2,2])
    padding = np.array([0,0])
    maxpooling_layer = MaxPooling2d(poolsize, stride, padding)

    method = 'approx-star'
    Istar_2 = maxpooling_layer.reach(Istar_1, method)
    # Istar1_2 = maxpooling_layer.reach(Istar1_1, method)
    # Istar2_2 = maxpooling_layer.reach(Istar2_1, method)

    f1 = np.array([[1.0]])
    f2 = np.array([[-1.0]])
    b1 = np.array([0.])
    b2 = np.array([0.])
    fc1 = FC(f1, b1, f2, b2, option='last')
    Istar_3 = fc1.merge_reach(Istar_2)
    print(Istar_1.V[:, :, 0, 0])
    print(Istar_2.V[:, :, 0, 0])
    print(Istar_3.V[:,:,0,0])
    # f1 = 0.1*np.ones((1,16))
    # f2 = -0.1*np.ones((1,16))
    # b1 = np.array([[0.5]])
    # b2 = np.array([[-0.5]])
    # fc1 = FC(f1,b1,f2,b2)
    # image1, image2 = fc1.merge_reach(I1,I1)
    # print(I1.V[:,:,0,0])
    # print(image1.V)
    # print(image2.V)

    # a = np.array([[1,2,3,-4],[0,1,-2,3]])
    # b = a.copy()
    # b[0,0] = 0
    # print(b)
    # print(a)
    # b[b<0] = 0
    # print(b)
    # print(type(b))
    # c = np.argwhere(b<0)
    # print(type(c))
    # print(c.dtype)
    # a[c,:] =0
    # print(a)
    # d = np.empty([0,2],dtype=c.dtype)
    # print(d.dtype)
    # a[d,:]=0
    # print(a)

    # a = np.array([[3], [2], [1]])
    # b = np.array([[6], [4], [2]])
    # t = []
    # for i in range(a.shape[0]):
    #     t.append((a[i,0],b[i,0]))
    # print(t)

    # m1 = np.array([2.0, 1.0, 3.0, 1.0])
    # m2 = np.array([1., 3., 3., 2.])
    # f1 = np.array([[0.1, 0.2, 0.2, 0.1], [0.2, 0.1, 0.1, 0.2]])
    # f2 = np.array([[-0.1, 0.2, 0.3, 0.1], [-0.2, 0.1, 0.1, 0.3]])
    # b1 = np.array([0.1, 0])
    # b2 = np.array([-0.1, 0.1])
    # m_f1 = np.concatenate((f1, np.zeros((2,4))))
    # m_f2 = np.concatenate((np.zeros((2,4)), f2))
    # merge_w = np.concatenate((m_f1, m_f2), axis=1)
    # merge_b = np.concatenate((b1, b2))
    # fc = nn.Linear(8, 4)
    # fc.weight = nn.Parameter(torch.from_numpy(merge_w), requires_grad=False)
    # fc.bias = nn.Parameter(torch.from_numpy(merge_b), requires_grad=False)
    # m_input = np.concatenate((m1, m2))
    # print(m_input.dtype)
    # print(fc.weight.dtype)
    # print(fc.bias.dtype)
    # output1 = fc(torch.from_numpy(m_input))
    # print(output1)


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
