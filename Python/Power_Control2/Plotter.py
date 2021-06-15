# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 10:49:44 2021

@author: Henke
"""
import matplotlib.pyplot as plt

#--Load acc_history from files
#filenames = ["0noise-10hidden", "0.1noise-10hidden-trunc", "0.1noise-10hidden-henrik"]
#filenames = ["0.1noise-100hidden-xiaowen", "0.1noise-100hidden-henrik"]
#filenames = ["0noise-10hidden-xiaowen", "1noise-10hidden-xiaowen", "1noise-10hidden-xiaowen-1rtx"]
#filenames = ["0.1noise-10hidden-xiaowen-10rtx", "0.1noise-10hidden-henrik-10rtx"]
#filenames = ["1noise-10hidden-henrik-0rtx-50budget", "1noise-10hidden-henrik-1rtx-50budget", "1noise-10hidden-henrik-3rtx-50budget", "1noise-10hidden-henrik-7rtx-50budget"]
#filenames = ["1noise-10hidden-henrik-0rtx-100budget", "1noise-10hidden-henrik-1rtx-100budget", "1noise-10hidden-henrik-3rtx-100budget", "1noise-10hidden-henrik-7rtx-100budget", "1noise-10hidden-henrik-15rtx-100budget", "1noise-10hidden-henrik-31rtx-100budget"]
#filenames = ["1noise-10hidden-henrik-0rtx-100budget", "1noise-10hidden-henrik-3rtx-100budget", "1noise-10hidden-henrik-31rtx-100budget"]
#filenames = ["1noise-10hidden-henrik-0rtx-200budget", "1noise-10hidden-henrik-1rtx-200budget", "1noise-10hidden-henrik-3rtx-200budget", "1noise-10hidden-henrik-0.1growth-200budget", "1noise-10hidden-henrik-0.2growth-200budget"]
#filenames = ["0.7071067811865475noise-10hidden-henrik-0rtx-200budget", "0.7071067811865475noise-10hidden-henrik-7rtx-200budget", "0.7071067811865475noise-10hidden-henrik-63rtx-200budget"]
#filenames = ["1noise-10hidden-henrik-3rtx-200budget", "1noise-10hidden-henrik-3rtx-200budget-2"]
#filenames = ["2noise-10hidden-henrik-0rtx-200budget", "2noise-10hidden-henrik-7rtx-200budget", "2noise-10hidden-henrik-15rtx-200budget", "2noise-10hidden-henrik-31rtx-200budget", "2noise-10hidden-henrik-0.1growth-200budget", "2noise-10hidden-henrik-0.2growth-200budget"]
#filenames = ["1.5noise-10hidden-henrik-0rtx-200budget", "1.5noise-10hidden-henrik-3rtx-200budget", "1.5noise-10hidden-henrik-7rtx-200budget", "1.5noise-10hidden-henrik-15rtx-200budget", "1.5noise-10hidden-henrik-31rtx-200budget"]
#filenames = ["0noise-10hidden-henrik-0rtx-50budget", "10noise-10hidden-henrik-0rtx-50budget", "10noise-10hidden-henrik-4rtx-50budget", "10noise-10hidden-henrik-1rtx-50budget"]
#filenames = ["0noise-10hidden-henrik-0rtx-50budget", "1noise-10hidden-henrik-0rtx-50budget", "2noise-10hidden-henrik-0rtx-50budget", "10noise-10hidden-henrik-0rtx-50budget", "20noise-10hidden-henrik-0rtx-50budget", "30noise-10hidden-henrik-0rtx-50budget", "40noise-10hidden-henrik-0rtx-50budget"]
#filenames = ["20noise-10hidden-henrik-0rtx-100budget", "20noise-10hidden-henrik-1rtx-100budget", "20noise-10hidden-henrik-3rtx-100budget"]
#--Load norm_history from files
filenames = ["1noise-10hidden-henrik-3rtx-101budget_norm", "1noise-10hidden-henrik-3rtx-101budget_error"]
#filenames = ["20noise-10hidden-henrik-1rtx-100budget_norm", "20noise-10hidden-henrik-1rtx-100budget_error", "20noise-10hidden-henrik-1rtx-100budget_rcv"]
#filenames = ["1noise-10hidden-henrik-9rtx-200budget_norm", "2noise-10hidden-henrik-0rtx-200budget_norm"]
#filenames = ["2noise-10hidden-henrik-0rtx-20budget_norm", "2noise-10hidden-henrik-0rtx-20budget_error"]
#filenames = ["4noise-10hidden-henrik-0rtx-20budget_norm", "4noise-10hidden-henrik-0rtx-20budget_error"]
#filenames = ["8noise-10hidden-henrik-0rtx-50budget_norm", "8noise-10hidden-henrik-0rtx-50budget_error"]
#filenames = ["10noise-10hidden-henrik-0rtx-50budget_norm", "10noise-10hidden-henrik-0rtx-50budget_error", "10noise-10hidden-henrik-0rtx-50budget_rcv"]
#filenames = ["0noise-10hidden-henrik-0rtx-50budget_norm", "0noise-10hidden-henrik-0rtx-50budget_error", "0noise-10hidden-henrik-0rtx-50budget_rcv"]
accuracies = False
ymin= 0
ymax = 0
if accuracies == True:
    ymin = 0
    ymax = 0.92
else:
    ymin = 0
    ymax = 10
acc_histories = []

for filename in filenames:
    acc_history = []
    with open("./data/" + filename + ".txt", "r") as filehandle:
        for line in filehandle:
            acc_history.append(float(line))
    acc_histories.append(acc_history)
    
#Find max length
max_len = 1
for acc_history in acc_histories:
    if len(acc_history) > max_len:
        max_len = len(acc_history)
        
#--Plot acc_histories
start = 0
end = -1
fixed_labels = True
if fixed_labels == False:
    labels = filenames
else:
    if accuracies == True:
        labels = ["M=1", "M=2", "M=4", "M=32", "M=0.1growth", "M=0.2growth", "M=0.3growth"]
    else:
        labels = ["norm", "error", "rcv_norm"]
    #labels = ["sigma=0", "sigma=1", "sigma=2", "sigma=10", "sigma=20", "sigma=30", "sigma=40"]
fig = plt.figure()
i = 0
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
for acc_history in acc_histories:
    plt.plot(range(len(acc_history)), acc_history, colors[i], label=labels[i])
    dashed_line_length = max_len-len(acc_history)
    plt.hlines(acc_history[len(acc_history)-1], len(acc_history)-1, max_len, colors[i], linestyles='dashed')
    plt.ylim((ymin,ymax))
    #plt.plot(range(dashed_line_length), [acc_history[len(acc_history)-1]]*dashed_line_length, colors[i] + "--", xmin=len(acc_history))
    i = i + 1
plt.xlabel("Communication Round")
plt.ylabel("Classification Accuracy")
plt.legend()
#compareplot_name = "200budget-1noise-comparison"
compareplot_name = "100budget-unnormalized-norm-comparison"
plt.savefig("./plots/"+compareplot_name+".png", format='png')