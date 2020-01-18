import pandas as pd
import numpy as np
from snownlp import SnowNLP
ACTIONS=['你管我', '我没错','我错了']
MAX_LEARN_SIZE=10
GAMMA=0.7
ALPHA=0.1
N_STATES=3

def init_qtable(n_states,actions):
    table=pd.DataFrame(np.zeros((n_states,len(actions))),columns=actions,)
    return table


def choose_action(state,q_table):
    state_actions=q_table.iloc[state,:]
    if (np.random.uniform()>0.9) or ((state_actions==0).all()):
        action_name=np.random.choice(ACTIONS)
    else:
        action_name=state_actions.idxmax()
    return action_name

def feedback():
    input_str=input("我：")
    input_str=SnowNLP(input_str)
    sentiment=input_str.sentiments
    if sentiment<0.4:
        R=-1
    elif sentiment<0.7:
        R=0
    elif sentiment<1:
        R=1
    return R

def learn():
    q_table=init_qtable(N_STATES,ACTIONS)
    for episode in range(MAX_LEARN_SIZE):
        S=0
        print("第{0}遍：...................".format(episode))
#        print("我：你错没错？")
        A=choose_action(S,q_table)
        #打印出当前的状态
        print("机器人: ",A)
        R=feedback()
        q_predict=q_table.loc[S,A]
        print(q_predict)
        q_target=R+GAMMA*q_table.iloc[S,:].max()
        print(q_target)
        q_table.loc[S,A]=ALPHA*(q_target-q_predict)
    return q_table

if __name__=='__main__':
    learn()

