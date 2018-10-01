import pandas as pd
import re
import numpy as np
import AES as aes

def get_wave_data(wave_num):
    path_wave_1 = './CPA/EMwavedata10000/aes_tv_0000001-0005000_power.csv' 
    path_wave_2 = './CPA/EMwavedata10000/aes_tv_0005001-0010000_power.csv'

    #wave_1 = pd.read_csv(path_wave_1,header = None, usecols = [3007])
    wave_1 = pd.read_csv(path_wave_1,header = None, usecols = [2836])
    wave_data_1 = wave_1.values.tolist()
    wave_data_1 = [e for inner_list in wave_data_1 for e in inner_list]
    #wave_2 = pd.read_csv(path_wave_2,header = None, usecols = [3007])
    wave_2 = pd.read_csv(path_wave_2,header = None, usecols = [2836])
    wave_data_2 = wave_2.values.tolist()
    wave_data_2 = [e for inner_list in wave_data_2 for e in inner_list]
    wave_data_1.extend(wave_data_2)    
    #print(len(wave_data_1))
    return wave_data_1[:wave_num]

def get_HDtable(HD_num):
    path_base = './result/result'
    #print(HD_num)
    if HD_num == 0:
        path = path_base + '.txt'
    else:
        path = path_base + '_' + str(HD_num) + '.txt'
    f = open(path)
    line = f.readlines()
    #line = re.split(',\n',line)
    f.close()
    
    for i in range(len(line)):
        line[i] = re.split('[,\n]',line[i])
        line[i].pop(-1)        
        if(i != len(line)-1):
            line[i] = list(map(int,line[i]))
    line.pop(-1)
    #print(line)
    #print(len(line[1]))
    #print(len(line))
    #print(type(line))
    #print('--------------------------------')
    return line

def pearson(x,y,n):
    result = []
    y_diff = y - np.mean(y)
    for i in range(len(x)):
        x_diff = x[i][:n] - np.mean(x[i][:n])
        result.append(abs(np.dot(x_diff, y_diff) / (np.sqrt(sum(x_diff ** 2)) * np.sqrt(sum(y_diff ** 2)))))
    key.append(np.argmax(result))
    #key.append(np.max(result))


if __name__ == '__main__':
    key = []
    num = 2800
    wave_data = get_wave_data(num)
    key_inf_0 = get_HDtable(0)
    key_inf_1 = get_HDtable(1)
    key_inf_2 = get_HDtable(2)
    key_inf_3 = get_HDtable(3)
    key_inf_4 = get_HDtable(4)
    key_inf_5 = get_HDtable(5)
    key_inf_6 = get_HDtable(6)
    key_inf_7 = get_HDtable(7)
    key_inf_8 = get_HDtable(8)
    key_inf_9 = get_HDtable(9)
    key_inf_10 = get_HDtable(10)
    key_inf_11 = get_HDtable(11)
    key_inf_12 = get_HDtable(12)
    key_inf_13 = get_HDtable(13)
    key_inf_14 = get_HDtable(14)
    key_inf_15 = get_HDtable(15)
    

    pearson(key_inf_0, wave_data, num)
    pearson(key_inf_1,wave_data,num)
    pearson(key_inf_2,wave_data,num)
    pearson(key_inf_3,wave_data,num)
    pearson(key_inf_4,wave_data,num)
    pearson(key_inf_5,wave_data,num)
    pearson(key_inf_6,wave_data,num)
    pearson(key_inf_7,wave_data,num)
    pearson(key_inf_8,wave_data,num)
    pearson(key_inf_9,wave_data,num)
    pearson(key_inf_10,wave_data,num)
    pearson(key_inf_11,wave_data,num)
    pearson(key_inf_12,wave_data,num)
    pearson(key_inf_13,wave_data,num)
    pearson(key_inf_14,wave_data,num)
    pearson(key_inf_15,wave_data,num)
    
    round_key_10 = []
    for i in range(16):
        if i % 4 == 0 :
            round_key_10.append([key[i]])
        else:
            num = int(i/4)
            round_key_10[num].append(key[i])
    print(round_key_10)

