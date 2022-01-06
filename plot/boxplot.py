import pandas as pd
import matplotlib.pyplot as plt

# ________________ import data _________________________
# MAPE
target1 = pd.read_csv('boxplot/target1_lstm.csv')
target1_RL = target1['RL'].values.astype('float32')
target1_Random = target1['Random'].values.astype('float32')
target1_Baseline = target1['Baseline'].values.astype('float32')

target2 = pd.read_csv('boxplot/target2_lstm.csv')
target2_RL = target2['RL'].values.astype('float32')
target2_Random = target2['Random'].values.astype('float32')
target2_Baseline = target2['Baseline'].values.astype('float32')

target3 = pd.read_csv('boxplot/target3_lstm.csv')
target3_RL = target3['RL'].values.astype('float32')
target3_Random = target3['Random'].values.astype('float32')
target3_Baseline = target3['Baseline'].values.astype('float32')

target4 = pd.read_csv('boxplot/target4_lstm.csv')
target4_RL = target4['RL'].values.astype('float32')
target4_Random = target4['Random'].values.astype('float32')
target4_Baseline = target4['Baseline'].values.astype('float32')

# RMSE
target1_RMSE = pd.read_csv('boxplot/target1_lstm_RMSE.csv')
target1_RL_RMSE = target1_RMSE['RL'].values.astype('float32')
target1_Random_RMSE = target1_RMSE['Random'].values.astype('float32')
target1_Baseline_RMSE = target1_RMSE['Baseline'].values.astype('float32')

target2_RMSE = pd.read_csv('boxplot/target2_lstm_RMSE.csv')
target2_RL_RMSE = target2_RMSE['RL'].values.astype('float32')
target2_Random_RMSE = target2_RMSE['Random'].values.astype('float32')
target2_Baseline_RMSE = target2_RMSE['Baseline'].values.astype('float32')

target3_RMSE = pd.read_csv('boxplot/target3_lstm_RMSE.csv')
target3_RL_RMSE = target3_RMSE['RL'].values.astype('float32')
target3_Random_RMSE = target3_RMSE['Random'].values.astype('float32')
target3_Baseline_RMSE = target3_RMSE['Baseline'].values.astype('float32')

target4_RMSE = pd.read_csv('boxplot/target4_lstm_RMSE.csv')
target4_RL_RMSE = target4_RMSE['RL'].values.astype('float32')
target4_Random_RMSE = target4_RMSE['Random'].values.astype('float32')
target4_Baseline_RMSE = target4_RMSE['Baseline'].values.astype('float32')

# R2
# MAPE
target1_R2 = pd.read_csv('boxplot/target1_lstm_R2.csv')
target1_RL_R2 = target1_R2['RL'].values.astype('float32')
target1_Random_R2 = target1_R2['Random'].values.astype('float32')
target1_Baseline_R2 = target1_R2['Baseline'].values.astype('float32')

target2_R2 = pd.read_csv('boxplot/target2_lstm_R2.csv')
target2_RL_R2 = target2_R2['RL'].values.astype('float32')
target2_Random_R2 = target2_R2['Random'].values.astype('float32')
target2_Baseline_R2 = target2_R2['Baseline'].values.astype('float32')

target3_R2 = pd.read_csv('boxplot/target3_lstm_R2.csv')
target3_RL_R2 = target3_R2['RL'].values.astype('float32')
target3_Random_R2 = target3_R2['Random'].values.astype('float32')
target3_Baseline_R2 = target3_R2['Baseline'].values.astype('float32')

target4_R2 = pd.read_csv('boxplot/target4_lstm_R2.csv')
target4_RL_R2 = target4_R2['RL'].values.astype('float32')
target4_Random_R2 = target4_R2['Random'].values.astype('float32')
target4_Baseline_R2 = target4_R2['Baseline'].values.astype('float32')

# RAE
target1_RAE = pd.read_csv('boxplot/target1_lstm_RAE.csv')
target1_RL_RAE = target1_RAE['RL'].values.astype('float32')
target1_Random_RAE = target1_RAE['Random'].values.astype('float32')
target1_Baseline_RAE = target1_RAE['Baseline'].values.astype('float32')

target2_RAE = pd.read_csv('boxplot/target2_lstm_RAE.csv')
target2_RL_RAE = target2_RAE['RL'].values.astype('float32')
target2_Random_RAE = target2_RAE['Random'].values.astype('float32')
target2_Baseline_RAE = target2_RAE['Baseline'].values.astype('float32')

target3_RAE = pd.read_csv('boxplot/target3_lstm_RAE.csv')
target3_RL_RAE = target3_RAE['RL'].values.astype('float32')
target3_Random_RAE = target3_RAE['Random'].values.astype('float32')
target3_Baseline_RAE = target3_RAE['Baseline'].values.astype('float32')

target4_RAE = pd.read_csv('boxplot/target4_lstm_RAE.csv')
target4_RL_RAE = target4_RAE['RL'].values.astype('float32')
target4_Random_RAE = target4_RAE['Random'].values.astype('float32')
target4_Baseline_RAE = target4_RAE['Baseline'].values.astype('float32')

# ___________________ define painting function __________________
def painting(input):
    # color
    for box, c in zip(input['boxes'], color):
        box.set(color=c)
    # median
    for median, c in zip(input['medians'], color):
        median.set(color=c, linewidth=1)
    # mean
    for mean, c in zip(input['means'], color):
        mean.set(markerfacecolor=c, markeredgecolor=c)
    # whisker
    for whisker, c in zip(input['whiskers'], color2):
        whisker.set(color=c, linewidth=1)
    # cap
    for cap, c in zip(input['caps'], color2):
        cap.set(color=c, linewidth=1)
    # flier
    for flier, c in zip(input['fliers'], color):
        flier.set(marker='x', markerfacecolor=c, markeredgecolor=c)

# ____________________ params _________________________
plt.figure(dpi=130)
font1 = {'family':'Times New Roman'}
labels = ' ', ' ', ' '
color = ['tab:orange', 'firebrick', 'tab:blue']
color2 = ['tab:orange', 'tab:orange', 'firebrick', 'firebrick', 'tab:blue', 'tab:blue']
plt.title('MAPE')
# _____________ Subfigure_1 _____________________
ax_1 = plt.subplot(441)
plt.ylabel('MAPE (%)')
plt.tick_params(axis='both', )
ax1 = plt.boxplot([target1_RL, target1_Random, target1_Baseline], showmeans=True, meanprops={'marker':'+'}, labels=labels)
painting(ax1)
ax_1.grid(ls='--')

# _____________ Subfigure_2 _____________________
labels = ' ', ' ', ' '
ax_2 = plt.subplot(442)
ax2 = plt.boxplot([target2_RL, target2_Random, target2_Baseline], showmeans=True, meanprops={'marker':'+'}, labels=labels)
painting(ax2)
ax_2.grid(ls='--')

# _____________ Subfigure_3 _____________________
labels = ' ', ' ', ' '
ax_3 = plt.subplot(443)
plt.xlabel('(c) target3', fontdict=font1)
ax3 = plt.boxplot([target3_RL, target3_Random, target3_Baseline], showmeans=True, meanprops={'marker':'+'}, labels=labels)
painting(ax3)
ax_3.grid(ls='--')

# _____________ Subfigure_4 _____________________
labels = ' ', ' ', ' '
ax_4 = plt.subplot(444)
plt.xlabel('(d) target4')
ax4 = plt.boxplot([target4_RL, target4_Random, target4_Baseline], showmeans=True, meanprops={'marker':'+'}, labels=labels)
painting(ax4)
ax_4.grid(ls='--')

# sub figure 5
labels = ' ', ' ', ' '
ax_5 = plt.subplot(445)
plt.ylabel('RMSE (10^-2)')
plt.tick_params(axis='both', )
print(target1_RL_RMSE)
ax5 = plt.boxplot([target1_RL_RMSE, target1_Random_RMSE, target1_Baseline_RMSE], showmeans=True, meanprops={'marker':'+'}, labels=labels)
painting(ax5)
ax_5.grid(ls='--')

# sub figure 6
labels = ' ', ' ', ' '
ax_6 = plt.subplot(446)
plt.tick_params(axis='both', )
ax6 = plt.boxplot([target2_RL_RMSE, target2_Random_RMSE, target2_Baseline_RMSE], showmeans=True, meanprops={'marker':'+'}, labels=labels)
painting(ax6)
ax_6.grid(ls='--')

# # sub figure 7
labels = ' ', ' ', ' '
ax_7 = plt.subplot(447)
plt.tick_params(axis='both', )
ax7 = plt.boxplot([target3_RL_RMSE, target3_Random_RMSE, target3_Baseline_RMSE], showmeans=True, meanprops={'marker':'+'}, labels=labels)
painting(ax7)
ax_7.grid(ls='--')

# sub figure 8
labels = ' ', ' ', ' '
ax_8 = plt.subplot(448)
plt.tick_params(axis='both', )
ax8 = plt.boxplot([target4_RL_RMSE, target4_Random_RMSE, target4_Baseline_RMSE], showmeans=True, meanprops={'marker':'+'}, labels=labels)
painting(ax8)
ax_8.grid(ls='--')

# sub figure 9
labels = ' ', ' ', ' '
ax_9 = plt.subplot(449)
plt.ylabel('RAE (%)')
plt.tick_params(axis='both', )
ax9 = plt.boxplot([target1_RL_RAE, target1_Random_RAE, target1_Baseline_RAE], showmeans=True, meanprops={'marker':'+'}, labels=labels)
painting(ax9)
ax_9.grid(ls='--')

# sub figure 10
labels = ' ', ' ', ' '
ax_10 = plt.subplot(4, 4, 10)
plt.tick_params(axis='both', )
ax10 = plt.boxplot([target2_RL_RAE, target2_Random_RAE, target2_Baseline_RAE], showmeans=True, meanprops={'marker':'+'}, labels=labels)
painting(ax10)
ax_10.grid(ls='--')

# # sub figure 11
labels = ' ', ' ', ' '
ax_11 = plt.subplot(4, 4, 11)
plt.tick_params(axis='both', )
ax11 = plt.boxplot([target3_RL_RAE, target3_Random_RAE, target3_Baseline_RAE], showmeans=True, meanprops={'marker':'+'}, labels=labels)
painting(ax11)
ax_11.grid(ls='--')

# sub figure 12
labels = ' ', ' ', ' '
ax_12 = plt.subplot(4, 4, 12)
plt.tick_params(axis='both', )
ax12 = plt.boxplot([target4_RL_RAE, target4_Random_RAE, target4_Baseline_RAE], showmeans=True, meanprops={'marker':'+'}, labels=labels)
painting(ax12)
ax_12.grid(ls='--')


# sub figure 13
labels = 'L3F', 'FML', 'Baseline'
ax_13 = plt.subplot(4, 4, 13)
plt.xlabel('(a) Target1')
plt.ylabel('R2 (%)')
plt.tick_params(axis='both', )
ax13 = plt.boxplot([target1_RL_R2, target1_Random_R2, target1_Baseline_R2], showmeans=True, meanprops={'marker':'+'}, labels=labels)
painting(ax13)
ax_13.grid(ls='--')

# sub figure 14
labels = 'L3F', 'FML', 'Baseline'
ax_14 = plt.subplot(4, 4, 14)
plt.xlabel('(b) Target2')
plt.tick_params(axis='both', )
ax14 = plt.boxplot([target2_RL_R2, target2_Random_R2, target2_Baseline_R2], showmeans=True, meanprops={'marker':'+'}, labels=labels)
painting(ax14)
ax_14.grid(ls='--')

# # sub figure 15
labels = 'L3F', 'FML', 'Baseline'
ax_15 = plt.subplot(4, 4, 15)
plt.xlabel('(c) Target3')
plt.tick_params(axis='both', )
ax15 = plt.boxplot([target3_RL_R2, target3_Random_R2, target3_Baseline_R2], showmeans=True, meanprops={'marker':'+'}, labels=labels)
painting(ax15)
ax_15.grid(ls='--')

# sub figure 16
labels = 'L3F', 'FML', 'Baseline'
ax_16 = plt.subplot(4, 4, 16)
plt.xlabel('(d) Target4')
plt.tick_params(axis='both', )
ax16 = plt.boxplot([target4_RL_R2, target4_Random_R2, target4_Baseline_R2], showmeans=True, meanprops={'marker':'+'}, labels=labels)
painting(ax16)
ax_16.grid(ls='--')

plt.show()
