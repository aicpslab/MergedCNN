import numpy as np
import torchvision.datasets
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
from layer import MergedConv2d
from layer import MaxPooling2d
from layer import FC
from layer import ReLULayer
from imageStar import ImageStar
import matplotlib.pyplot as plt
import torch.optim as optim
import torchvision.models as models
import imageio

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
        # self.temp = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv2(x)))
        # self.temp = torch.flatten(x, 1)
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        # self.temp = F.relu(self.fc1(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def compute_net_diff(result_path1, result_path2, IM, lb, ub):
    # result_path1 = './cifar_net3.pth'
    # result_path2 = './cifar_net4.pth'
    # net1 = Net()
    # net1.load_state_dict(torch.load(result_path1))
    # net2 = Net()
    # net2.load_state_dict(torch.load(result_path2))
    net1 = torch.load(result_path1)         # only load for parameter reading
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

    # IM = imageio.v2.imread('C:/Users/emiya/Desktop/PhD Research/Research 1'
    #                        '/CIFAR-10-images-master/test/airplane/0001.jpg')
    # IM = imageio.v2.imread('D:/Cody/phd/research1/CIFAR-10-images/test/airplane/0001.jpg')
    # IM = IM / 255
    # IM = (IM - 0.5) / 0.5
    LB = np.zeros((32, 32, 3))
    UB = np.zeros((32, 32, 3))
    LB[0, 0, 0] = lb
    UB[0, 0, 0] = ub
    IM_m = np.concatenate((IM, IM))
    LB_m = np.concatenate((LB, LB))
    UB_m = np.concatenate((UB, UB))
    I1_m = ImageStar(IM_m, LB_m, UB_m)

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

    n = len(Istar_12)
    max_dis = 0
    for i in range(n):
        Star1 = Istar_12[i].toStar()
        vertices = Star1.toPolyhedron()
        # print(vertices)
        for i in range(vertices.shape[0]):
            dis = np.linalg.norm(vertices[i,:])
            if dis > max_dis:
                max_dis = dis
    print(max_dis)
    print('\n')
    return max_dis
    # return Istar_12

if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    root_path = 'C:/Users/emiya/Desktop/PhD Research/Research 1'
    # result_path = './cifar_net3.pth'
    # root_path = 'D:/Cody/phd/research1/cifar10'
    # result_path = 'D:/Cody/phd/research1/cifar10/cifar_net3.pth'
    result_path1 = './cifar_net3.pth'
    result_path2 = './cifar_net4.pth'

    # transform = transforms.Compose(
    #     [transforms.ToTensor(),
    #      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    #
    # batch_size = 4
    #
    # # trainset = torchvision.datasets.CIFAR10(root=root_path, train=True,
    # #                                         download=True, transform=transform)
    # # trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
    # #                                           shuffle=True, num_workers=2)
    # testset = torchvision.datasets.CIFAR10(root=root_path, train=False,
    #                                        download=True, transform=transform)
    # testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
    #                                          shuffle=False, num_workers=2)
    #
    # classes = ('plane', 'car', 'bird', 'cat',
    #            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # load model and test
    net1 = Net()
    net1.load_state_dict(torch.load(result_path1))      # load whole network
    net2 = Net()
    net2.load_state_dict(torch.load(result_path2))

    for iter in range(10):
        index = np.random.randint(0, 1000)
        index_s = str(index).zfill(4)
        # index_s = '0385'
        image_path = 'C:/Users/emiya/Desktop/PhD Research/Research 1/CIFAR-10-images-master/test/airplane/'\
                     + index_s +'.jpg'
        # test on one image set
        IM = imageio.v2.imread(image_path)
        # IM = imageio.v2.imread('D:/Cody/phd/research1/CIFAR-10-images/test/airplane/0081.jpg')
        IM = IM / 255
        IM = (IM - 0.5)/0.5
        lb = -0.05
        ub = 0.05
        # lb = 0
        # ub = 0
        max_dis = compute_net_diff(result_path1, result_path2, IM, lb, ub)
        # temp = compute_net_diff(result_path1, result_path2, IM, lb, ub)
        # if max_dis < 5:
        #     print("%s" % index_s)
        max_diff_total = 0
        for i in range(100):
            noise = lb + (ub - lb)*np.random.random()
            IM_sample = IM.copy()
            IM_sample[0, 0, 0] += noise
            outputs1 = net1(torch.from_numpy(IM_sample.transpose(2, 0, 1)).unsqueeze_(0).to(torch.float32))
            outputs2 = net2(torch.from_numpy(IM_sample.transpose(2, 0, 1)).unsqueeze_(0).to(torch.float32))
            out_diff = outputs1 - outputs2
            # print(outputs1)
            # print(outputs2)
            # print(out_diff)
            max_diff = torch.max(abs(out_diff), 1)[0]
            if max_diff > max_diff_total:
                max_diff_total = max_diff
            # print(torch.max(abs(out_diff), 1)[0])
            # if max_diff > max_dis:
            #     print(False)
        print(max_diff_total)

    # print(net1.temp.shape)
    # print(net1.temp.dtype)
    # print(net1.temp)
    # a = torch.transpose(net1.temp,0,2)
    # print(a.dtype)
    # # print(a.shape)
    # a = torch.transpose(a, 1,3)
    # # print(a.shape)
    # a = torch.transpose(a, 2,3)
    # print(a.shape)
    # print(len(temp))
    # print(temp[0].V.shape)
    # for i in range(5):
    #     print(a[i,:,5,0])
    #     print(temp[0].V[i,:,5,0])
    #     print('\n')

    # print(temp[0].V[0,0,0:120,0])

    # a = net1.temp
    # print(a[0, 0:20])
    # print(temp[0].V[0,0,0:20,0])
    # for i in range(10):
    #     print(a[0, i*10:(i+1)*10])
    #     print(temp[0].V[0,0,i*10:(i+1)*10,0])

    # print(outputs1.shape)
    # print(outputs1)
    # print(temp[0].V.shape)
    # print(temp[0].V[0,0,0:10,0])
    #
    # print(outputs2)
    # print(temp[0].V[0,0,10:,0])

