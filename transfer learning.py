# coding = utf-8
import random

import torch
import pandas as pd
import numpy as np
import read_dataset
import MyNet
from torch.utils.data import DataLoader, Dataset
import MyFunction
from torchvision import transforms
import selector
import csv


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# ______________________ pre ______________________________
# federation data
path = 'dataset\occupancies'
name_dict, type_dict, data_dict, poi_density_list = read_dataset.import_data(path)


# target data
target = pd.read_csv('dataset/target/target4.csv')
target_data = target['RATE'].values.astype('float64')
target_data = torch.as_tensor(target_data, device=device).float()
target_type = 'res'
target_poi_density = 12

# _____________________ Selector ____________________
beg = 0
end = 30-1
select = random.randint(beg, end)
selected = data_dict[select]

times = 10
pre_training_epoch_num = 200
epoch_num = 400

output_matrix = torch.zeros([epoch_num, times*6])

for t in range(times):
    # init network
    net = MyNet.MyLSTMNet(input_size=1, hidden_size=1, seq_len=6, output_size=1, num_layers=1).to(device)
    loss_function = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.02)

    support_size = int(len(target_data) * 0.6)
    query_size = int(len(target_data) * 0.8)
    transfer_data = selected[:query_size]
    transfer_set = read_dataset.MyData(transfer_data, seq_length=6)
    transfer_loader = DataLoader(transfer_set, batch_size=len(transfer_data), shuffle=False)
    for p in range(200):
        for i, data in enumerate(transfer_loader):
            sample, label = data
            optimizer.zero_grad()
            output = net(sample)
            output = torch.squeeze(output)
            loss = loss_function(output, label)
            loss.backward()
            optimizer.step()

    # input data
    support_size = int(len(target_data) * 0.6)
    query_size = int(len(target_data) * 0.8)
    support_target = target_data[support_size:query_size]
    query_target = target_data[query_size:]
    support_set = read_dataset.MyData(support_target, seq_length=6)
    query_set = read_dataset.MyData(query_target, seq_length=6)
    support_loader = DataLoader(support_set, batch_size=len(support_set), shuffle=False)
    query_loader = DataLoader(query_set, batch_size=len(query_set), shuffle=False)

    fine_tuning_loss_list = torch.zeros([epoch_num])
    test_loss_list = torch.zeros([epoch_num])
    RMSE_list = torch.zeros([epoch_num])
    MAPE_list = torch.zeros([epoch_num])
    R2_list = torch.zeros([epoch_num])
    RAE_list = torch.zeros([epoch_num])

    for epoch in range(epoch_num):
        # fine-tuning
        for i, support in enumerate(support_loader):
            support_sample, support_label = support
            optimizer.zero_grad()
            output = net(support_sample)
            output = torch.squeeze(output)
            loss = loss_function(output, support_label)
            loss.backward()
            optimizer.step()
            fine_tuning_loss_list[epoch] = loss.item()
        # testing
        for i, query in enumerate(query_loader):
            query_sample, query_label = query
            optimizer.zero_grad()
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

        output_matrix[:, 6 * t + 0] = RMSE_list
        output_matrix[:, 6 * t + 1] = MAPE_list
        output_matrix[:, 6 * t + 2] = R2_list
        output_matrix[:, 6 * t + 3] = RAE_list
        output_matrix[:, 6 * t + 4] = fine_tuning_loss_list
        output_matrix[:, 6 * t + 5] = test_loss_list

output_matrix = output_matrix.detach().numpy()

# output loss
output = output_matrix
f = open('result/Transfer_target4_LSTM_output_matrix.csv', 'w', newline='')
csv_writer = csv.writer(f)
for l in output:
    csv_writer.writerow(l)
f.close()
