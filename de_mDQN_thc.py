import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np                             
import matplotlib.pyplot as plt
from function import Interaction, rollingupdate, reward_function,moving_average

#The number of sample for each epoch
window_size = 500
BATCH_SIZE = 1
LR = 1000*1e-5                                      # learning rate
EPSILON = 0.9                                       # greedy policy
GAMMA = 0                                           # reward discount
TARGET_REPLACE_ITER = window_size/2                 # Q-network updating frequency
MEMORY_CAPACITY = window_size                       # Capacity of Memory Base
# 5 thread selections
N_ACTIONS1 = 5
# 15 calls selection
N_ACTIONS2 = 15
# the size of state vector
#Thread State
N_STATES1 = 7
#Call State
N_STATES2 = 8
#Well-trained simulation network of bookinfo
PATH = 'modelresult/NNpk8s'
device = 'cpu'
#Parameter for simulation network of bookinfo
dim_hidden = 512
dim_out = 2
dim_in = 9
#the number of total epoch
round = 500
#the number of hidden neurons for DQN
DQNhidden = 256
# Rolling window size
reward_window_size = 25
call_startp = 435 - N_ACTIONS2

# The interaction between agent output and simulation model
def mInteraction(k8s_state, k8smean, k8sstd, label_std, label_mean, k8s_net, device):
    # Normalization on k8s state
    k8s_state = np.nan_to_num((k8s_state - k8smean)/k8sstd)
    k8s_output = k8s_net(torch.Tensor(k8s_state).to(device)).detach().numpy()
    k8s_output = (k8s_output * label_std) + label_mean
    return k8s_output

def Interaction(rl_state, a1, a2, k8smean, k8sstd, label_std, label_mean, k8s_net, device):
    # Attach action with rl_state, k8s_state has one more action value
    k8s_state = np.append(rl_state, a1+1)
    k8s_state = np.append(k8s_state, a2+call_startp)
    # Normalization on k8s state
    k8s_state = np.nan_to_num((k8s_state - k8smean)/k8sstd)
    k8s_output = k8s_net(torch.Tensor(k8s_state).to(device)).detach().numpy()
    k8s_output = (k8s_output * label_std) + label_mean
    return k8s_output

# Approximation MLP Model for Microserivices
class MLP(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super(MLP, self).__init__()
        self.layer_input = nn.Linear(dim_in, dim_hidden)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)
        self.layer_hidden1 = nn.Linear(dim_hidden, dim_hidden)
        self.layer_hidden2 = nn.Linear(dim_hidden, dim_hidden)
        self.layer_hidden3 = nn.Linear(dim_hidden,  dim_out)
        self.layer_output = nn.Linear(dim_hidden, dim_out)

    def forward(self, x):
        x = self.layer_input(x)
        x = self.relu(self.dropout(x))
        x = self.layer_hidden1(x)
        x = self.relu(self.dropout(x))
        x = self.layer_hidden2(x)
        x = self.relu(self.dropout(x))
        x = self.layer_output(x)
        return x

# Agent 1 Q-Network
class Net1(nn.Module):
    def __init__(self):
        super(Net1, self).__init__()
        self.fc1 = nn.Linear(N_STATES1, DQNhidden)
        self.fc1.weight.data.normal_(0, 0.1)
        self.out = nn.Linear(DQNhidden, N_ACTIONS1)
        self.out.weight.data.normal_(0, 0.1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        actions_value = self.out(x)
        return actions_value


# Agent 1 DQN
class DQN1(object):
    def __init__(self):
        self.eval_net, self.target_net = Net1(), Net1()
        self.learn_step_counter = 0                                             # for target updating
        self.memory_counter = 0                                                 # for storing memory
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES1 * 2 + 2))             # initialize memory base，one row = one transition state
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()                                           # MSE Loss (loss(xi, yi)=(xi-yi)^2)
        self.average = np.zeros((1, 2*N_STATES1+2))
        self.pwrSumAvg = np.zeros((1, 2*N_STATES1+2))
        self.n = 0


    def choose_action(self, x):                                                 # Define the action selection fucntion
        x = torch.unsqueeze(torch.FloatTensor(x), 0)
        if np.random.uniform() < EPSILON:                                       # genearate a number iid~[0, 1)，select the optimzied action if smaller than elipson
            actions_value = self.eval_net.forward(x)                            # load state x，forward action value
            action = torch.max(actions_value, 1)[1].data.numpy()                # obtain action value
            action = action[0]
        else:                                                                   # Randonly select action
            action = np.random.randint(0, N_ACTIONS1)
        return action

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))
        index = self.memory_counter % MEMORY_CAPACITY                           # obtain the size of loading transition states
        self.memory[index, :] = transition                                      # Load transition
        self.memory_counter += 1                                                # memory_counter plus one after each loading

    def learn(self):
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1


        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        b_memory = self.memory[sample_index, :]
        b_s = torch.FloatTensor(b_memory[:, :N_STATES1])
        b_a = torch.LongTensor(b_memory[:, N_STATES1:N_STATES1+1].astype(int))
        b_r = torch.FloatTensor(b_memory[:, N_STATES1+1:N_STATES1+2])
        b_s_ = torch.FloatTensor(b_memory[:, -N_STATES1:])
        q_eval = self.eval_net(b_s).gather(1, b_a)
        q_next = self.target_net(b_s_).detach()
        q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)
        loss = self.loss_func(q_eval, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

# Agent 2 Q-Network
class Net2(nn.Module):
    def __init__(self):
        super(Net2, self).__init__()
        self.fc1 = nn.Linear(N_STATES2, DQNhidden)
        self.fc1.weight.data.normal_(0, 0.1)
        self.out = nn.Linear(DQNhidden, N_ACTIONS2)
        self.out.weight.data.normal_(0, 0.1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        actions_value = self.out(x)
        return actions_value


# Agent 2 DQN
class DQN2(object):
    def __init__(self):
        self.eval_net, self.target_net = Net2(), Net2()
        self.learn_step_counter = 0                                             # for target updating
        self.memory_counter = 0                                                 # for storing memory
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES2 * 2 + 2))
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()
        self.average = np.zeros((1, 2*N_STATES2+2))
        self.pwrSumAvg = np.zeros((1, 2*N_STATES2+2))
        self.n = 0


    def choose_action(self, x):
        x = torch.unsqueeze(torch.FloatTensor(x), 0)
        if np.random.uniform() < EPSILON:
            actions_value = self.eval_net.forward(x)
            action = torch.max(actions_value, 1)[1].data.numpy()
            action = action[0]
        else:
            action = np.random.randint(0, N_ACTIONS2)
        return action

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        # 目标网络参数更新
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1


        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        b_memory = self.memory[sample_index, :]
        b_s = torch.FloatTensor(b_memory[:, :N_STATES2])
        b_a = torch.LongTensor(b_memory[:, N_STATES2:N_STATES2+1].astype(int))
        b_r = torch.FloatTensor(b_memory[:, N_STATES2+1:N_STATES2+2])
        b_s_ = torch.FloatTensor(b_memory[:, -N_STATES2:])
        q_eval = self.eval_net(b_s).gather(1, b_a)
        q_next = self.target_net(b_s_).detach()
        q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)
        loss = self.loss_func(q_eval, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

dqn1 = DQN1()
dqn2 = DQN2()
k8s_model = MLP(dim_in=dim_in, dim_hidden = dim_hidden,
                               dim_out = dim_out)
k8s_weights = torch.load(PATH)
k8s_model.load_state_dict(k8s_weights)
k8s_model.to(device)
k8s_net = k8s_model.eval()
# Initialize rl_state1, rl_state2
rl_state1 = np.array([3, 2, 2, 3, 1, 1, 100]) #435
#Normalization paramater for k8s states
k8smean = np.array([3.47452161, 2.72070522, 2.01085788, 3, 1, 1, 100, 2.99924747, 435.57299505])
k8sstd = np.array([2.11732473, 2.01623149, 1.52637189, 0, 0, 0, 0, 1.41425137, 13.38385997])
#Normalization paramater for k8s responses
label_std = np.array([3.71739217e-01,7.97757181e+02])
label_mean = np.array([2.57400850e-01, 1.15260615e+03])
epoch_sumreward = []
epoch_sumreward_baseline = []
rolling_baseline = []
rolling_rl = []
for i in range(round):
    print('<<<<<<<<<Episode: %s' % i)
    episode_reward_sum = 0
    episode_baseline_sum = 0
    for all in range(window_size):
        a1 = dqn1.choose_action(rl_state1)
        rl_state2 = np.append(rl_state1, a1+1)
        a2 = dqn2.choose_action(rl_state2)
        k8s_state = np.append(rl_state2, a2+call_startp)
        k8s_output = mInteraction(k8s_state, k8smean, k8sstd, label_std, label_mean, k8s_net, device)
        # baseline Setting
        a_baseline1 = np.random.randint(1, N_ACTIONS1, 1, dtype='l')
        a_baseline2 = np.random.randint(1, N_ACTIONS2, 1, dtype='l')
        k8s_output_baseline = Interaction(rl_state1, a_baseline1,a_baseline2, k8smean, k8sstd, label_std, label_mean, k8s_net, device)
        new_r = reward_function(k8s_output)
        r_baseline = reward_function(k8s_output_baseline)
        top3 = np.random.randint(1, 7, 3, dtype='l')
        rl_state_old1 = rl_state1
        rl_state_old2 = rl_state2
        rl_state1[0], rl_state1[1], rl_state1[2] = top3[0], top3[1], top3[2]
        s_1 = rl_state1
        s_2 = rl_state2
        dqn1.store_transition(rl_state_old1, a1, new_r, s_1)
        dqn2.store_transition(rl_state_old2, a2, new_r, s_2)
        episode_reward_sum += new_r
        episode_baseline_sum += r_baseline
        rl_state1 = s_1
        rl_state2 = s_2
        if dqn1.memory_counter > MEMORY_CAPACITY:
            dqn1.learn()
        if dqn2.memory_counter > MEMORY_CAPACITY:
            dqn2.learn()
    epoch_sumreward.append(episode_reward_sum/window_size)
    epoch_sumreward_baseline.append(episode_baseline_sum/window_size)
    rolling_baseline.append(np.mean(moving_average(np.array(epoch_sumreward_baseline), reward_window_size))/window_size)
    rolling_rl.append(np.mean(moving_average(np.array(epoch_sumreward), reward_window_size))/window_size)
    plt.figure()
    plt.plot(range(len(epoch_sumreward)), epoch_sumreward, color = 'r', label = 'Demulti-RL')
    plt.plot(range(len(epoch_sumreward_baseline)), epoch_sumreward_baseline, color='b', label='Baseline')
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('Reward')
    plt.savefig('modelresult/dm_reward.png')
    plt.figure()
    plt.plot(range(len(rolling_rl)), rolling_rl, color = 'r', label = 'Demulti-RL')
    plt.plot(range(len(rolling_baseline)), rolling_baseline, color='b', label='Baseline')
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('Cumulative Reward')
    plt.savefig('modelresult/dm_creward.png')
    matrix = np.vstack((np.array(rolling_baseline), np.array(rolling_rl)))
    print(rolling_rl[-1])
    np.savetxt('modelresult/dmrecordings.txt', matrix)
