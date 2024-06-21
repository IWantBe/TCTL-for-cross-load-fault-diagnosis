import CNN_C
import mmd
import torch.optim as optim
import torch.nn as nn
import ReadData_200 as RD
import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

np.random.seed(42)


def mmd_loss(x_src, x_tar):
    return mmd.mmd_rbf_noaccelerate(x_src, x_tar)


def train(model, optimizer, epoch, data_src, data_tar, sample_len):
    if epoch > 2:
        # beta = 1
        beta = 0.25
        # beta = 0
    else:
        beta = 0
    # print(beta)
    total_loss_train = 0
    criterion = nn.CrossEntropyLoss()
    correct = 0
    batch_j = 0
    list_src, list_tar = list(enumerate(data_src)), list(enumerate(data_tar))
    for batch_id, (data, target) in enumerate(data_src):
        _, (x_tar, y_target) = list_tar[batch_j]
        data, target = data.to(device), target.to(device)
        x_tar, y_target = x_tar.to(device), y_target.to(device)
        model.train()
        y_src, y_tar, x_src_mmd, x_tar_mmd = model(data, x_tar)
        y_tarpred = y_tar.data.max(1)[1]   # 伪标签
        loss_c = criterion(y_src, target.long())
        loss_pseudo = criterion(y_tar, y_tarpred.long())
        loss_mmd = mmd_loss(x_src = x_src_mmd, x_tar= x_tar_mmd)
        pred = y_src.data.max(1)[1]  # get the x_src_mmd, index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        # print('lossc=', loss_c)
        # print('loss_pseudo=', loss_pseudo)
        # print('loss_mmd=', loss_mmd)
        loss = loss_c + LAMBDA * loss_mmd + beta * loss_pseudo
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss_train += loss.data
        res_i = 'Epoch: [{}/{}], Batch: [{}/{}], loss: {:.6f}'.format(
            epoch, N_EPOCH, batch_id + 1, len(data_src), loss.data
        )
        batch_j += 1
        if batch_j >= len(list_tar)-1:
            batch_j = 0
    total_loss_train /= len(data_src)
    acc = correct * 100. / len(data_src.dataset)
    res_e = 'Epoch: [{}/{}], training loss: {:.6f}, correct: [{}/{}], training accuracy: {:.4f}%'.format(
        epoch, N_EPOCH, total_loss_train, correct, len(data_src.dataset), acc
    )
    # tqdm.write(res_e)
    RESULT_TRAIN.append([epoch, total_loss_train, acc])
    return model


def test(model, data_tar, e, sample_len):
    total_loss_test = 0
    correct = 0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for batch_id, (data, target) in enumerate(data_tar):
            data, target = data.to(device), target.to(device)
            model.eval()
            ypred, _, _, _ = model(data, data)
            loss = criterion(ypred, target.long())
            pred = ypred.data.max(1)[1]  # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
            total_loss_test += loss.data
        accuracy = correct * 100. / len(data_tar.dataset)
        res = 'Test: total loss: {:.6f}, correct: [{}/{}], testing accuracy: {:.4f}%'.format(
            total_loss_test, correct, len(data_tar.dataset), accuracy
        )
    RESULT_TEST.append([e, total_loss_test, accuracy])
    # tqdm.write(res)


for g in range(5):
    print('第几次实验', g)
    for i in range(4):
        sour = str(i)
        # sour = str(1)
        path_source = r'../data/' + sour + 'HP'
        path_target = 'null'
        for j in range(4):
            if i == j:
                pass
            else:
                tar = str(j)
                # tar = str(0)
                path_target = r'../data_cwru/' + tar + 'HP'
                path_source = r'../data_cwru/' + sour + 'HP'
                print('path_source=', path_source)
                print('path_target=', path_target)
s
                rate = [1.0, 0, 0]
                sample_len = 420
                # stride = sample_len
                stride = int(sample_len / 2)
                # LAMBDA = 1
                LAMBDA = 0.7
                # LAMBDA = 0
                RESULT_TRAIN = []
                RESULT_TEST = []
                N_EPOCH = 200  # 900
                x_train_source, y_train_source, x_validate, y_validate, x_test, y_test = \
                    RD.get_data(path_source, rate, stride, sample_len)
                x_train_target, y_train_target, x_validate, y_validate, x_test, y_test = \
                    RD.get_data(path_target, rate, stride, sample_len)

                # 增加一个维度以满足conv1D的输入要求[]
                x_train_source, x_train_target = x_train_source[:, np.newaxis, :], x_train_target[:, np.newaxis,
                                                                                   :]  # 后面训练模块增加通道数

                data_src = torch.tensor(x_train_source)
                data_src = data_src.to(torch.float32)
                data_tar = torch.tensor(x_train_target)
                data_tar = data_tar.to(torch.float32)
                # print(data_src.shape)
                # print(data_tar.shape)

                train_data_source = TensorDataset(data_src, torch.tensor(y_train_source))
                train_data_target = TensorDataset(data_tar, torch.tensor(y_train_target))
                # train_dataloader_source = DataLoader(train_data_source, batch_size=32, shuffle=True, drop_last=True)
                # train_dataloader_target = DataLoader(train_data_target, batch_size=32, shuffle=True, drop_last=True)
                # print(type(train_dataloader_source))
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                # print("Using {} device".format(device))
                model = CNN_C.CNN1()
                model = model.to(device)
                # optimizer = optim.SGD(
                #     model.parameters(),
                #     lr=0.001,
                #     momentum=0.05,
                #     weight_decay=0.003
                # )
                optimizer = optim.Adam(model.parameters(), lr=0.001)

                for e in range(1, N_EPOCH):
                    train_dataloader_source = DataLoader(train_data_source, batch_size=32, shuffle=True, drop_last=True)
                    train_dataloader_target = DataLoader(train_data_target, batch_size=32, shuffle=True, drop_last=True)
                    # print("epoch=%d", e)
                    model = train(model=model, optimizer=optimizer,
                                  epoch=e, data_src=train_dataloader_source, data_tar=train_dataloader_target,
                                  sample_len=sample_len)
                    test(model, train_dataloader_target, e, sample_len=sample_len)
                print(RESULT_TEST[-1])
                # 保存模型