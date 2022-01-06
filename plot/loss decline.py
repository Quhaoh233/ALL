import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# input
# target1
Target1 = pd.read_csv('loss/target1.csv')
L3F_RMSE_target1 = Target1['RMSE'].values.astype('float64')
L3F_MAPE_target1 = Target1['MAPE'].values.astype('float64')
L3F_R2_target1 = Target1['R2'].values.astype('float64')
L3F_RAE_target1 = Target1['RAE'].values.astype('float64')
L3F_TRAIN_target1 = Target1['TRAIN'].values.astype('float64')
L3F_TEST_target1 = Target1['TEST'].values.astype('float64')

Target1_baseline = pd.read_csv('loss/target1_baseline.csv')
L3F_RMSE_target1_baseline = Target1_baseline['RMSE'].values.astype('float64')
L3F_MAPE_target1_baseline = Target1_baseline['MAPE'].values.astype('float64')
L3F_R2_target1_baseline = Target1_baseline['R2'].values.astype('float64')
L3F_RAE_target1_baseline = Target1_baseline['RAE'].values.astype('float64')
L3F_TRAIN_target1_baseline = Target1_baseline['TRAIN'].values.astype('float64')
L3F_TEST_target1_baseline = Target1_baseline['TEST'].values.astype('float64')

# target2
Target2 = pd.read_csv('loss/target2.csv')
L3F_RMSE_target2 = Target2['RMSE'].values.astype('float64')
L3F_MAPE_target2 = Target2['MAPE'].values.astype('float64')
L3F_R2_target2 = Target2['R2'].values.astype('float64')
L3F_RAE_target2 = Target2['RAE'].values.astype('float64')
L3F_TRAIN_target2 = Target2['TRAIN'].values.astype('float64')
L3F_TEST_target2 = Target2['TEST'].values.astype('float64')

Target2_baseline = pd.read_csv('loss/target1_baseline.csv')
L3F_RMSE_target2_baseline = Target2_baseline['RMSE'].values.astype('float64')
L3F_MAPE_target2_baseline = Target2_baseline['MAPE'].values.astype('float64')
L3F_R2_target2_baseline = Target2_baseline['R2'].values.astype('float64')
L3F_RAE_target2_baseline = Target2_baseline['RAE'].values.astype('float64')
L3F_TRAIN_target2_baseline = Target2_baseline['TRAIN'].values.astype('float64')
L3F_TEST_target2_baseline = Target2_baseline['TEST'].values.astype('float64')

# target3
Target3 = pd.read_csv('loss/target3.csv')
L3F_RMSE_target3 = Target3['RMSE'].values.astype('float64')
L3F_MAPE_target3 = Target3['MAPE'].values.astype('float64')
L3F_R2_target3 = Target3['R2'].values.astype('float64')
L3F_RAE_target3 = Target3['RAE'].values.astype('float64')
L3F_TRAIN_target3 = Target3['TRAIN'].values.astype('float64')
L3F_TEST_target3 = Target3['TEST'].values.astype('float64')

Target3_baseline = pd.read_csv('loss/target1_baseline.csv')
L3F_RMSE_target3_baseline = Target3_baseline['RMSE'].values.astype('float64')
L3F_MAPE_target3_baseline = Target3_baseline['MAPE'].values.astype('float64')
L3F_R2_target3_baseline = Target3_baseline['R2'].values.astype('float64')
L3F_RAE_target3_baseline = Target3_baseline['RAE'].values.astype('float64')
L3F_TRAIN_target3_baseline = Target3_baseline['TRAIN'].values.astype('float64')
L3F_TEST_target3_baseline = Target3_baseline['TEST'].values.astype('float64')

# target4
Target4 = pd.read_csv('loss/target1.csv')
L3F_RMSE_target4 = Target4['RMSE'].values.astype('float64')
L3F_MAPE_target4 = Target4['MAPE'].values.astype('float64')
L3F_R2_target4 = Target4['R2'].values.astype('float64')
L3F_RAE_target4 = Target4['RAE'].values.astype('float64')
L3F_TRAIN_target4 = Target4['TRAIN'].values.astype('float64')
L3F_TEST_target4 = Target4['TEST'].values.astype('float64')

Target4_baseline = pd.read_csv('loss/target1_baseline.csv')
L3F_RMSE_target4_baseline = Target4_baseline['RMSE'].values.astype('float64')
L3F_MAPE_target4_baseline = Target4_baseline['MAPE'].values.astype('float64')
L3F_R2_target4_baseline = Target4_baseline['R2'].values.astype('float64')
L3F_RAE_target4_baseline = Target4_baseline['RAE'].values.astype('float64')
L3F_TRAIN_target4_baseline = Target4_baseline['TRAIN'].values.astype('float64')
L3F_TEST_target4_baseline = Target4_baseline['TEST'].values.astype('float64')

# 截断
loss_epoch = 100
x = np.arange(loss_epoch) + 1
# target1
L3F_RMSE_target1 = L3F_RMSE_target1[0:loss_epoch]
L3F_MAPE_target1 = L3F_MAPE_target1[0:loss_epoch]
L3F_R2_target1 = L3F_R2_target1[0:loss_epoch]
L3F_RAE_target1 = L3F_RAE_target1[0:loss_epoch]
L3F_TRAIN_target1 = L3F_TRAIN_target1[0:loss_epoch]
L3F_TEST_target1 = L3F_TEST_target1[0:loss_epoch]

L3F_RMSE_target1_baseline = L3F_RMSE_target1_baseline[0:loss_epoch]
L3F_MAPE_target1_baseline = L3F_MAPE_target1_baseline[0:loss_epoch]
L3F_R2_target1_baseline = L3F_R2_target1_baseline[0:loss_epoch]
L3F_RAE_target1_baseline = L3F_RAE_target1_baseline[0:loss_epoch]
L3F_TRAIN_target1_baseline = L3F_TRAIN_target1_baseline[0:loss_epoch]
L3F_TEST_target1_baseline = L3F_TEST_target1_baseline[0:loss_epoch]

# target2
L3F_RMSE_target2 = L3F_RMSE_target2[0:loss_epoch]
L3F_MAPE_target2 = L3F_MAPE_target2[0:loss_epoch]
L3F_R2_target2 = L3F_R2_target2[0:loss_epoch]
L3F_RAE_target2 = L3F_RAE_target2[0:loss_epoch]
L3F_TRAIN_target2 = L3F_TRAIN_target2[0:loss_epoch]
L3F_TEST_target2 = L3F_TEST_target2[0:loss_epoch]

L3F_RMSE_target2_baseline = L3F_RMSE_target2_baseline[0:loss_epoch]
L3F_MAPE_target2_baseline = L3F_MAPE_target2_baseline[0:loss_epoch]
L3F_R2_target2_baseline = L3F_R2_target2_baseline[0:loss_epoch]
L3F_RAE_target2_baseline = L3F_RAE_target2_baseline[0:loss_epoch]
L3F_TRAIN_target2_baseline = L3F_TRAIN_target2_baseline[0:loss_epoch]
L3F_TEST_target2_baseline = L3F_TEST_target2_baseline[0:loss_epoch]

# target3
L3F_RMSE_target3 = L3F_RMSE_target3[0:loss_epoch]
L3F_MAPE_target3 = L3F_MAPE_target3[0:loss_epoch]
L3F_R2_target3 = L3F_R2_target3[0:loss_epoch]
L3F_RAE_target3 = L3F_RAE_target3[0:loss_epoch]
L3F_TRAIN_target3 = L3F_TRAIN_target3[0:loss_epoch]
L3F_TEST_target3 = L3F_TEST_target3[0:loss_epoch]

L3F_RMSE_target3_baseline = L3F_RMSE_target3_baseline[0:loss_epoch]
L3F_MAPE_target3_baseline = L3F_MAPE_target3_baseline[0:loss_epoch]
L3F_R2_target3_baseline = L3F_R2_target3_baseline[0:loss_epoch]
L3F_RAE_target3_baseline = L3F_RAE_target3_baseline[0:loss_epoch]
L3F_TRAIN_target3_baseline = L3F_TRAIN_target3_baseline[0:loss_epoch]
L3F_TEST_target3_baseline = L3F_TEST_target3_baseline[0:loss_epoch]

# target4
L3F_RMSE_target4 = L3F_RMSE_target4[0:loss_epoch]
L3F_MAPE_target4 = L3F_MAPE_target4[0:loss_epoch]
L3F_R2_target4 = L3F_R2_target4[0:loss_epoch]
L3F_RAE_target4 = L3F_RAE_target4[0:loss_epoch]
L3F_TRAIN_target4 = L3F_TRAIN_target4[0:loss_epoch]
L3F_TEST_target4 = L3F_TEST_target4[0:loss_epoch]

L3F_RMSE_target4_baseline = L3F_RMSE_target4_baseline[0:loss_epoch]
L3F_MAPE_target4_baseline = L3F_MAPE_target4_baseline[0:loss_epoch]
L3F_R2_target4_baseline = L3F_R2_target4_baseline[0:loss_epoch]
L3F_RAE_target4_baseline = L3F_RAE_target4_baseline[0:loss_epoch]
L3F_TRAIN_target4_baseline = L3F_TRAIN_target4_baseline[0:loss_epoch]
L3F_TEST_target4_baseline = L3F_TEST_target4_baseline[0:loss_epoch]


fig = plt.figure()

# ________________ target1 __________________
# TRAIN and TEST
ax1 = plt.subplot2grid((5, 4), (0, 0))
ax1.plot(x, L3F_TRAIN_target1,  c='tab:orange', label='Test:L3F')
ax1.plot(x, L3F_TEST_target1, '--', c='tab:orange', label='Train:L3F')
plt.ylabel('Loss: L3F')
# plt.xlabel('Epoch')
ax1.legend(loc='upper left')
ax = ax1.twinx()
ax.plot(L3F_TRAIN_target1_baseline,  c='tab:blue', label='Test:Baseline')
ax.plot(L3F_TEST_target1_baseline, '--', c='tab:blue', label='Train:Baseline')
ax.legend(loc='upper right')
plt.tick_params(axis='both', )

# RMSE
ax1 = plt.subplot2grid((5, 4), (1, 0))
ax1.plot(x, L3F_RMSE_target1, c='tab:orange', label='L3F')
plt.ylabel('RMSE (10^-2): L3F')
# plt.xlabel('Epoch')
ax1.legend(loc='upper left')
ax = ax1.twinx()
ax.plot(L3F_RMSE_target1_baseline, c='tab:blue', label='Baseline')
ax.legend(loc='upper right')
plt.tick_params(axis='both', )

# MAPE
ax1 = plt.subplot2grid((5, 4), (2, 0))
ax1.plot(x, L3F_MAPE_target1, c='tab:orange', label='L3F')
plt.ylabel('MAPE (%): L3F')
# plt.xlabel('Epoch')
ax1.legend(loc='upper left')
ax = ax1.twinx()
ax.plot(L3F_MAPE_target1_baseline, c='tab:blue', label='Baseline')
ax.legend(loc='upper right')
plt.tick_params(axis='both', )

# R2
ax1 = plt.subplot2grid((5, 4), (3, 0))
ax1.plot(x, L3F_R2_target1, c='tab:orange', label='L3F')
plt.ylabel('R2 (%): L3F')
# plt.xlabel('Epoch')
ax1.legend(loc='upper left')
ax = ax1.twinx()
ax.plot(L3F_R2_target1_baseline, c='tab:blue', label='Baseline')
ax.legend(loc='upper right')
plt.tick_params(axis='both', )

# RAE
ax1 = plt.subplot2grid((5, 4), (4, 0))
ax1.plot(x, L3F_RAE_target1, c='tab:orange', label='L3F')
plt.ylabel('RAE (%): L3F')
plt.xlabel('Target 1')
ax1.legend(loc='upper left')
ax = ax1.twinx()
ax.plot(L3F_RAE_target1_baseline, c='tab:blue', label='Baseline')
ax.legend(loc='upper right')
plt.tick_params(axis='both', )

# ________________ target2 __________________
# TRAIN and TEST
ax1 = plt.subplot2grid((5, 4), (0, 1))
ax1.plot(x, L3F_TEST_target2, c='tab:orange', label='Test:L3F')
ax1.plot(x, L3F_TRAIN_target2, '--', c='tab:orange', label='Train:L3F')

# plt.xlabel('Epoch')
ax1.legend(loc='upper left')
ax = ax1.twinx()
ax.plot(L3F_TRAIN_target2_baseline, c='tab:blue', label='Test:Baseline')
ax.plot(L3F_TEST_target2_baseline, '--', c='tab:blue', label='Train:Baseline')
ax.legend(loc='upper right')
plt.tick_params(axis='both', )

# RMSE
ax1 = plt.subplot2grid((5, 4), (1, 1))
ax1.plot(x, L3F_RMSE_target2, c='tab:orange', label='L3F')
# plt.xlabel('Epoch')
ax1.legend(loc='upper left')
ax = ax1.twinx()
ax.plot(L3F_RMSE_target2_baseline, c='tab:blue', label='Baseline')
ax.legend(loc='upper right')
plt.tick_params(axis='both', )

# MAPE
ax1 = plt.subplot2grid((5, 4), (2, 1))
ax1.plot(x, L3F_MAPE_target2, c='tab:orange', label='L3F')
# plt.xlabel('Epoch')
ax1.legend(loc='upper left')
ax = ax1.twinx()
ax.plot(L3F_MAPE_target2_baseline, c='tab:blue', label='Baseline')
ax.legend(loc='upper right')
plt.tick_params(axis='both', )

# R2
ax1 = plt.subplot2grid((5, 4), (3, 1))
ax1.plot(x, L3F_R2_target2, c='tab:orange', label='L3F')
# plt.xlabel('Epoch')
ax1.legend(loc='upper left')
ax = ax1.twinx()
ax.plot(L3F_R2_target2_baseline, c='tab:blue', label='Baseline')
ax.legend(loc='upper right')
plt.tick_params(axis='both', )

# RAE
ax1 = plt.subplot2grid((5, 4), (4, 1))
ax1.plot(x, L3F_RAE_target2, c='tab:orange', label='L3F')
plt.xlabel('Target 2')
ax1.legend(loc='upper left')
ax = ax1.twinx()
ax.plot(L3F_RAE_target2_baseline, c='tab:blue', label='Baseline')
ax.legend(loc='upper right')
plt.tick_params(axis='both', )

# ________________ target3 __________________
# TRAIN and TEST
ax1 = plt.subplot2grid((5, 4), (0, 2))
ax1.plot(x, L3F_TEST_target3, c='tab:orange', label='Test:L3F')
ax1.plot(x, L3F_TRAIN_target3, '--', c='tab:orange', label='Train:L3F')
# plt.xlabel('Epoch')
ax1.legend(loc='upper left')
ax = ax1.twinx()
ax.plot(L3F_TRAIN_target3_baseline, c='tab:blue', label='Test:Baseline')
ax.plot(L3F_TEST_target3_baseline, '--', c='tab:blue', label='Train:Baseline')
ax.legend(loc='upper right')
plt.tick_params(axis='both', )

# RMSE
ax1 = plt.subplot2grid((5, 4), (1, 2))
ax1.plot(x, L3F_RMSE_target3, c='tab:orange', label='L3F')
# plt.xlabel('Epoch')
ax1.legend(loc='upper left')
ax = ax1.twinx()
ax.plot(L3F_RMSE_target3_baseline, c='tab:blue', label='Baseline')
ax.legend(loc='upper right')

plt.tick_params(axis='both', )

# MAPE
ax1 = plt.subplot2grid((5, 4), (2, 2))
ax1.plot(x, L3F_MAPE_target3, c='tab:orange', label='L3F')

# plt.xlabel('Epoch')
ax1.legend(loc='upper left')
ax = ax1.twinx()
ax.plot(L3F_MAPE_target3_baseline, c='tab:blue', label='Baseline')
ax.legend(loc='upper right')

plt.tick_params(axis='both', )

# R2
ax1 = plt.subplot2grid((5, 4), (3, 2))
ax1.plot(x, L3F_R2_target3, c='tab:orange', label='L3F')

# plt.xlabel('Epoch')
ax1.legend(loc='upper left')
ax = ax1.twinx()
ax.plot(L3F_R2_target3_baseline, c='tab:blue', label='Baseline')
ax.legend(loc='upper right')

plt.tick_params(axis='both', )

# RAE
ax1 = plt.subplot2grid((5, 4), (4, 2))
ax1.plot(x, L3F_RAE_target3, c='tab:orange', label='L3F')

plt.xlabel('Target 3')
ax1.legend(loc='upper left')
ax = ax1.twinx()
ax.plot(L3F_RAE_target3_baseline, c='tab:blue', label='Baseline')
ax.legend(loc='upper right')

plt.tick_params(axis='both', )

# ________________ target4 __________________
# TRAIN and TEST
ax1 = plt.subplot2grid((5, 4), (0, 3))
ax1.plot(x, L3F_TRAIN_target4,  c='tab:orange', label='Test:L3F')
ax1.plot(x, L3F_TEST_target4, '--', c='tab:orange', label='Train:L3F')

# plt.xlabel('Epoch')
ax1.legend(loc='upper left')
ax = ax1.twinx()
ax.plot(L3F_TRAIN_target4_baseline, c='tab:blue', label='Test:Baseline')
ax.plot(L3F_TEST_target4_baseline, '--', c='tab:blue', label='Train:Baseline')
ax.legend(loc='upper right')
plt.ylabel('Loss: Baseline')
plt.tick_params(axis='both', )

# RMSE
ax1 = plt.subplot2grid((5, 4), (1, 3))
ax1.plot(x, L3F_RMSE_target4, c='tab:orange', label='L3F')

# plt.xlabel('Epoch')
ax1.legend(loc='upper left')
ax = ax1.twinx()
ax.plot(L3F_RMSE_target4_baseline, c='tab:blue', label='Baseline')
ax.legend(loc='upper right')
plt.ylabel('RMSE (10^-2): Baseline')
plt.tick_params(axis='both', )

# MAPE
ax1 = plt.subplot2grid((5, 4), (2, 3))
ax1.plot(x, L3F_MAPE_target4, c='tab:orange', label='L3F')

# plt.xlabel('Epoch')
ax1.legend(loc='upper left')
ax = ax1.twinx()
ax.plot(L3F_MAPE_target4_baseline, c='tab:blue', label='Baseline')
ax.legend(loc='upper right')
plt.ylabel('MAPE (%): Baseline')
plt.tick_params(axis='both', )

# R2
ax1 = plt.subplot2grid((5, 4), (3, 3))
ax1.plot(x, L3F_R2_target4, c='tab:orange', label='L3F')

# plt.xlabel('Epoch')
ax1.legend(loc='upper left')
ax = ax1.twinx()
ax.plot(L3F_R2_target4_baseline, c='tab:blue', label='Baseline')
ax.legend(loc='upper right')
plt.ylabel('R2 (%): Baseline')
plt.tick_params(axis='both', )

# RAE
ax1 = plt.subplot2grid((5, 4), (4, 3))
ax1.plot(x, L3F_RAE_target4, c='tab:orange', label='L3F')

plt.xlabel('Target 4')
ax1.legend(loc='upper left')
ax = ax1.twinx()
ax.plot(L3F_RAE_target4_baseline, c='tab:blue', label='Baseline')
ax.legend(loc='upper right')
plt.ylabel('RAE (%): Baseline')
plt.tick_params(axis='both', )



plt.show()