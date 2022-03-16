# coding = utf-8

import torch
import pandas as pd
import numpy as np
import read_dataset
import MyNet
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import selector
import csv


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

times = 10
pre_training_epoch_num = 200
test_epoch_num = 400
support_value = 0
query_value = 0.8
# ______________________ pre ______________________________
# input federation data
path = 'members'
name_dict, data_dict = read_dataset.import_data(path)

# input target data
target_path = 'targets'
target_name_dict, target_data_dict = read_dataset.import_data(target_path)

avr_metric_list = np.zeros([len(target_data_dict), 4])
Select_list = np.zeros([len(target_data_dict), 3])

for num in range(len(target_data_dict)):
    print('target =', num)
    target_name = target_name_dict[num]
    target_num = num + 1
    target_data = target_data_dict[num]

    # _____________________ Selector ____________________
    # select_dict = selector.Eq_selector(data_dict, target_num)    # Equivalent Selector as A3C-Selector (RL)
    # select_dict = selector.RL_selector(target_type, target_poi_density, data_dict, federation_state)
    # select_dict = selector.type_selector(target_type, type_dict, data_dict)
    # select_dict = selector.transfer_selector(data_dict)
    select_dict, index_list = selector.random_selector(data_dict)   # random select
    Select_list[num, :] = index_list

    test_size = len(target_data) - int(len(target_data)*query_value) - 6*2
    occ_list = np.zeros([test_size, times])
    output_matrix = torch.zeros([test_epoch_num, times*6])
    net_dict = dict()

    for t in range(times):
        print('time =', t+1)
        # network instantiation
        net_dict[num, t] = MyNet.MyLSTMNet().to(device)
        net = net_dict[num, t]
        loss_function = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(net.parameters(), lr=0.02)

        # _______________ training ________________________
        # input data
        train_task_dict = dict()
        support_set_dict = dict()
        query_set_dict = dict()
        support_dataloader_dict = dict()
        query_dataloader_dict = dict()
        task_num = len(select_dict)
        for n in range(task_num):
            train_task_dict[n] = select_dict[n]
            support_size = int(len(train_task_dict[n]) * 0.6)
            query_size = int(len(train_task_dict[n]) * query_value)
            temp = train_task_dict[n][:support_size]
            support_set_dict[n] = read_dataset.MyData(data=temp, seq_length=6)
            support_dataloader_dict[n] = DataLoader(support_set_dict[n], batch_size=len(support_set_dict[n]), shuffle=False)
            temp = train_task_dict[n][support_size:query_size]
            query_set_dict[n] = read_dataset.MyData(data=temp, seq_length=6)
            query_dataloader_dict[n] = DataLoader(query_set_dict[n], batch_size=len(query_set_dict[n]), shuffle=False)

        # ori_ta matrix
        ori_ta = dict()
        temp_num = 0
        with torch.no_grad():
            for param in net.parameters():
                ori_ta[temp_num] = param.data
                temp_num += 1
        # outer loop
        for epoch in range(pre_training_epoch_num):
            if (epoch+1) % 10 == 0:
                print('pre_training_epoch =', epoch+1, '/', pre_training_epoch_num)
            # gradient matrix
            gradient = dict()
            temp_num = 0
            for param in net.parameters():
                if param.requires_grad:
                    gradient[temp_num] = ori_ta[temp_num] * 0
                    temp_num += 1
            # inner loop
            for n in range(task_num):
                # FOMAML
                # init params
                temp_num = 0
                with torch.no_grad():
                    for param in net.parameters():
                        if param.requires_grad:
                            param.data = ori_ta[temp_num]
                            temp_num += 1

                for (support, query) in zip(support_dataloader_dict[n], query_dataloader_dict[n]):
                    support_sample, support_label = support
                    query_sample, query_label = query
                    # ith task, Support_set, temporal parameter
                    optimizer.zero_grad()
                    output = net(support_sample)
                    output = torch.squeeze(output)
                    loss = loss_function(output, support_label)
                    loss.backward()
                    optimizer.step()
                    # ith task, Query_set, gradient
                    optimizer.zero_grad()
                    output = net(query_sample)
                    output = torch.squeeze(output)
                    loss = loss_function(output, query_label)
                    loss.backward()
                    optimizer.step()
                    # extract cumulative gradient
                    with torch.no_grad():
                        gradient_index = 0
                        for param in net.parameters():
                            if param.requires_grad:
                                gradient[gradient_index] = gradient[gradient_index] + param.grad
                                gradient_index += 1

            # update ori_ta
            for index in range(len(ori_ta)):
                temp_value = ori_ta[index] - 0.03 * (gradient[index] / task_num)
                ori_ta[index] = temp_value
            # print('pretraining_epoch=', epoch+1, '/', pre_training_epoch_num)

        # ______________ testing _____________________
        support_size = int(len(target_data)*support_value)    # 3d: 0.7
        query_size = int(len(target_data)*query_value)
        support_target = target_data[support_size:query_size]   # 1) Partial data: [support_size:query_size]; 2) Full data: [:, query_size]
        query_target = target_data[query_size:]
        support_set = read_dataset.MyData(support_target, seq_length=6)
        query_set = read_dataset.MyData(query_target, seq_length=6)
        support_loader = DataLoader(support_set, batch_size=len(support_set), shuffle=False)
        query_loader = DataLoader(query_set, batch_size=len(query_set), shuffle=False)

        fine_tuning_loss_list = torch.zeros([test_epoch_num])
        test_loss_list = torch.zeros([test_epoch_num])
        RMSE_list = torch.zeros([test_epoch_num])
        MAPE_list = torch.zeros([test_epoch_num])
        R2_list = torch.zeros([test_epoch_num])
        RAE_list = torch.zeros([test_epoch_num])

        # init params
        temp_num = 0
        with torch.no_grad():
            for param in net.parameters():
                if param.requires_grad:
                    param.data = ori_ta[temp_num]
                    temp_num += 1

        for epoch in range(test_epoch_num):
            # fine-tuning
            if (epoch+1) % 10 == 0:
                print('test_epoch =', epoch+1, '/', test_epoch_num)
            for i, support in enumerate(support_loader):
                support_sample, support_label = support
                optimizer.zero_grad()
                output = net(support_sample)
                output = torch.squeeze(output)
                loss = loss_function(output, support_label)
                loss.backward()
                optimizer.step()
                fine_tuning_loss_list[epoch] = loss.item()
                # print(epoch, '/', fine_tuning_epoch_num, 'Loss =', loss.item())

            # testing
            for i, query in enumerate(query_loader):
                query_sample, query_label = query
                output = net(query_sample)
                output = torch.squeeze(output)
                loss = loss_function(output, query_label)
                test_loss_list[epoch] = loss.item()

                # output metrics
                output = torch.reshape(output, [len(query_label)]).cpu()
                output = output.detach().numpy()
                query_label = torch.reshape(query_label, [len(query_label)]).cpu()
                query_label = query_label.detach().numpy()

                # calculate MAPE
                MAPE = np.mean(abs(output - query_label) / query_label) * 100
                # RMSE
                RMSE = np.sqrt(np.mean((output - query_label)*(output - query_label))) * 100
                # R2
                SSE = np.sum((output - query_label)*(output - query_label))
                SST =np.sum((np.mean(query_label) - query_label)*(np.mean(query_label) - query_label))
                R2_score = (1 - SSE / SST) * 100
                # RAE
                RAE = np.sum(abs(output - query_label)) / np.sum(abs(np.mean(query_label) - query_label)) * 100

                RMSE_list[epoch] = RMSE
                MAPE_list[epoch] = MAPE
                R2_list[epoch] = R2_score
                RAE_list[epoch] = RAE
        occ_list[:, t] = output
        output_matrix[:, 6*t+0] = RMSE_list
        output_matrix[:, 6*t+1] = MAPE_list
        output_matrix[:, 6*t+2] = R2_list
        output_matrix[:, 6*t+3] = RAE_list
        output_matrix[:, 6*t+4] = fine_tuning_loss_list
        output_matrix[:, 6*t+5] = test_loss_list


    # ________________ output ______________________________________
    output_matrix = output_matrix.detach().numpy()

    print(target_name, 'result')
    # print final metrics
    avr_RMSE = 0
    avr_MAPE = 0
    avr_R2 = 0
    avr_RAE = 0
    for t in range(times):
        avr_RMSE += output_matrix[output_matrix.shape[0]-1, 6 * t + 0]
        avr_MAPE += output_matrix[output_matrix.shape[0]-1, 6 * t + 1]
        avr_R2 += output_matrix[output_matrix.shape[0]-1, 6 * t + 2]
        avr_RAE += output_matrix[output_matrix.shape[0]-1, 6 * t + 3]
    avr_RMSE = avr_RMSE / times
    avr_MAPE = avr_MAPE / times
    avr_R2 = avr_R2 / times
    avr_RAE = avr_RAE / times
    print('RMSE =', avr_RMSE)
    print('MAPE =', avr_MAPE)
    print('R2 =', avr_R2)
    print('RAE =', avr_RAE)

    avr_metric_list[num, 0] = avr_RMSE
    avr_metric_list[num, 1] = avr_MAPE
    avr_metric_list[num, 2] = avr_R2
    avr_metric_list[num, 3] = avr_RAE

    # output metrics
    output = output_matrix
    f = open('result/Full data/transfer_LSTM/transfer_LSTM_%s_metrics.csv' % target_name, 'w', newline='')
    csv_writer = csv.writer(f)
    for l in output:
        csv_writer.writerow(l)
    f.close()

    # output occupancy
    output = occ_list
    f = open('result/Full data/transfer_LSTM/transfer_LSTM_%s_occupancy.csv' % target_name, 'w', newline='')
    csv_writer = csv.writer(f)
    for l in output:
        csv_writer.writerow(l)
    f.close()

# output avr_metrics
output = avr_metric_list
f = open('result/Full data/transfer_LSTM/transfer_LSTM_avr_metrics.csv', 'w', newline='')
csv_writer = csv.writer(f)
for l in output:
    csv_writer.writerow(l)
f.close()

print(Select_list)
