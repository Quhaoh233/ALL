import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import MultipleLocator

# input
target1 = pd.read_csv('predict/target1_predict.csv')
X = target1['TIME'].values

target1_origin = target1['Origin'].values.astype('float64')
target1_L3F = target1['L3F'].values.astype('float64')
target1_LSTM = target1['LSTM'].values.astype('float64')
target1_SVR =target1['SVR'].values.astype('float64')

target2 = pd.read_csv('predict/target2_predict.csv')
target2_origin = target2['Origin'].values.astype('float64')
target2_L3F = target2['L3F'].values.astype('float64')
target2_LSTM = target2['LSTM'].values.astype('float64')
target2_SVR =target2['SVR'].values.astype('float64')

target3 = pd.read_csv('predict/target3_predict.csv')
target3_origin = target3['Origin'].values.astype('float64')
target3_L3F = target3['L3F'].values.astype('float64')
target3_LSTM = target3['LSTM'].values.astype('float64')
target3_SVR =target3['SVR'].values.astype('float64')

target4 = pd.read_csv('predict/target4_predict.csv')
target4_origin = target4['Origin'].values.astype('float64')
target4_L3F = target4['L3F'].values.astype('float64')
target4_LSTM = target4['LSTM'].values.astype('float64')
target4_SVR =target4['SVR'].values.astype('float64')

x = np.arange(len(target1_L3F))

fig = plt.figure()

ax1 = plt.subplot2grid((4, 4), (0, 0), colspan=4)
ax1.plot(X, target1_origin, '--', c='tab:gray', label='Origin')
ax1.plot(X, target1_L3F, c='tab:orange', label='L3F-LSTM')
ax1.plot(X, target1_LSTM, c='tab:blue', label='LSTM')
# ax1.boxplot(x, target1_SVR, c='tab:red', label='SVR')

# 设置横坐标间隔
x_major_locator = MultipleLocator(240)
ax = plt.gca()  # 实例化
ax.xaxis.set_major_locator(x_major_locator)

ax1.tick_params(labelbottom=True, labelleft=True)
ax1.set_xlabel('Target1')
ax1.set_ylabel('Parking Occupancy')
ax1.legend(loc='best')

ax2 = plt.subplot2grid((4, 4), (1, 0), colspan=4)
x_major_locator = MultipleLocator(240)
ax = plt.gca()
ax.xaxis.set_major_locator(x_major_locator)
ax2.plot(X, target2_origin, '--', c='tab:gray', label='Origin')
ax2.plot(X, target2_L3F, c='tab:orange', label='L3F-LSTM')
ax2.plot(X, target2_LSTM, c='tab:blue', label='LSTM')
ax2.tick_params(labelbottom=True, labelleft=True)
ax2.set_xlabel('Target2')
ax2.set_ylabel('Parking Occupancy')
ax2.legend(loc='best')

ax3 = plt.subplot2grid((4, 4), (2, 0), colspan=4)
x_major_locator = MultipleLocator(240)
ax = plt.gca()
ax.xaxis.set_major_locator(x_major_locator)
ax3.plot(X, target3_origin, '--', c='tab:gray', label='Origin')
ax3.plot(X, target3_L3F, c='tab:orange', label='L3F-LSTM')
ax3.plot(X, target3_LSTM, c='tab:blue', label='LSTM')
ax3.tick_params(labelleft=True)
ax3.set_xlabel('Target3')
ax3.set_ylabel('Parking Occupancy')
ax3.legend(loc='best')

ax4 = plt.subplot2grid((4, 4), (3, 0), colspan=4)
x_major_locator = MultipleLocator(240)
ax = plt.gca()
ax.xaxis.set_major_locator(x_major_locator)
ax4.plot(X, target4_origin, '--', c='tab:gray', label='Origin')
ax4.plot(X, target4_L3F, c='tab:orange', label='L3F-LSTM')
ax4.plot(X, target4_LSTM, c='tab:blue', label='LSTM')
ax4.tick_params(labelbottom=True, labelleft=True)
ax4.set_xlabel('Target4')
ax4.set_ylabel('Parking Occupancy')
ax4.legend(loc='best')

# plt.tight_layout()
plt.show()