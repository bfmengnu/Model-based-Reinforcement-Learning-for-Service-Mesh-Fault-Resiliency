import torch                                    # 导入torch
import torch.nn as nn                           # 导入torch.nn
import torch.nn.functional as F                 # 导入torch.nn.functional
import numpy as np                              # 导入numpy
import gym                                      # 导入gym
import matplotlib.pyplot as plt


def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

def rollingupdate(x, mean, pwrSumAvg, n):
    if n < window_size:
        n += 1
    else:
        n = window_size
    mean += (x - mean) / n
    pwrSumAvg += (x*x - pwrSumAvg) / n
    stdDev = np.sqrt((pwrSumAvg*n-n*mean*mean)/(n-1))
    x = np.nan_to_num((x - mean)/stdDev)
    return x, mean, pwrSumAvg, n


def reward_function(k8s_output):
    #failure rate larger than 0.1(10#) could be regarded as an effective fault injection.
    if k8s_output[-1] > 0.1:
        new_r = (k8s_output[-1]*(k8s_output[-2]))/200
        # reward should be positive. It is possible that negitive value apprears due to simulation errors
        if new_r < 0:
            new_r = 0
    else:
            new_r = -5
    return new_r

def Interaction(rl_state, a, k8smean, k8sstd, label_std, label_mean, k8s_net, device):
    # Attach action with rl_state, k8s_state has one more action value
    k8s_state = np.append(rl_state, a+1)
    # k8s action starts from 1, so a = a+1
    k8s_state[-1], k8s_state[-2] = k8s_state[-2], k8s_state[-1]
    # Normalization on k8s state
    k8s_state = np.nan_to_num((k8s_state - k8smean)/k8sstd)
    k8s_output = k8s_net(torch.Tensor(k8s_state).to(device)).detach().numpy()
    k8s_output = (k8s_output * label_std) + label_mean
    return k8s_output



