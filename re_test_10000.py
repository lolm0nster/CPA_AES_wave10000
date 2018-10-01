import pandas as pd
import re
import numpy as np
import tqdm as tq
import math
def get_wave_data(wave_num):
    path_wave_1 = './CPA/EMwavedata10000/aes_tv_0000001-0005000_power.csv' 
    path_wave_2 = './CPA/EMwavedata10000/aes_tv_0005001-0010000_power.csv'

    wave_1 = pd.read_csv(path_wave_1,header = None, usecols = [wave_num])
    wave_data_1 = wave_1.values.tolist()
    wave_data_1 = [e for inner_list in wave_data_1 for e in inner_list]
    wave_2 = pd.read_csv(path_wave_2,header = None, usecols = [wave_num])
    wave_data_2 = wave_2.values.tolist()
    wave_data_2 = [e for inner_list in wave_data_2 for e in inner_list]
    wave_data_1.extend(wave_data_2)    
    #print(len(wave_data_1))
    return wave_data_1

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
    
    y_diff = y[:n] - np.mean(y[:n])
    x_diff = x[:n] - np.mean(x[:n])
    result = abs(np.dot(x_diff, y_diff) / (np.sqrt(sum(x_diff ** 2)) * np.sqrt(sum(y_diff ** 2))))
    if math.isnan(result):
        result = 0
    return result 


if __name__ == '__main__':
    key = []
    #w_num = 2965
    w_num = 2835
    wave_data = get_wave_data(w_num)
    #key_inf_0 = np.array(get_HDtable(0))
    #key_inf_1 = np.array(get_HDtable(1))
    #key_inf_2 = np.array(get_HDtable(2))
    key_inf_3 = np.array(get_HDtable(3))
    key_inf_4 = np.array(get_HDtable(4))
    key_inf_5 = np.array(get_HDtable(5))
    key_inf_6 = np.array(get_HDtable(6))
    key_inf_7 = np.array(get_HDtable(7))
    key_inf_8 = np.array(get_HDtable(8))
    key_inf_9 = np.array(get_HDtable(9))
    key_inf_10 = np.array(get_HDtable(10))
    key_inf_11 = np.array(get_HDtable(11))
    key_inf_12 = np.array(get_HDtable(12))
    key_inf_13 = np.array(get_HDtable(13))
    key_inf_14 = np.array(get_HDtable(14))
    key_inf_15 = np.array(get_HDtable(15))
    rr = []
    num = 4
    for k in tq.tqdm(range(2000)):
        num += 5
        r = []
        for i in range(255):
            r.append(pearson(key_inf_3[i], wave_data, num))
        rr.append(r)
    df = pd.DataFrame(rr)
    df.to_csv('key_3.csv',index = False, header = False)
    rr = []
    num = 4
    for k in tq.tqdm(range(2000)):
        num += 5
        r = []
        for i in range(255):
            r.append(pearson(key_inf_4[i], wave_data, num))
        rr.append(r)
    df = pd.DataFrame(rr)
    df.to_csv('key_4.csv',index = False, header = False)
    rr = []
    num = 4
    for k in tq.tqdm(range(2000)):
        num += 5
        r = []
        for i in range(255):
            r.append(pearson(key_inf_5[i], wave_data, num))
        rr.append(r)
    df = pd.DataFrame(rr)
    df.to_csv('key_5.csv',index = False, header = False)
    rr = []
    num = 4
    for k in tq.tqdm(range(2000)):
        num += 5
        r = []
        for i in range(255):
            r.append(pearson(key_inf_6[i], wave_data, num))
        rr.append(r)
    df = pd.DataFrame(rr)
    df.to_csv('key_6.csv',index = False, header = False)
    rr = []
    num = 4
    for k in tq.tqdm(range(2000)):
        num += 5
        r = []
        for i in range(255):
            r.append(pearson(key_inf_6[i], wave_data, num))
        rr.append(r)
    df = pd.DataFrame(rr)
    df.to_csv('key_6.csv',index = False, header = False)
    rr = []
    num = 4
    for k in tq.tqdm(range(2000)):
        num += 5
        r = []
        for i in range(255):
            r.append(pearson(key_inf_7[i], wave_data, num))
        rr.append(r)
    df = pd.DataFrame(rr)
    df.to_csv('key_7.csv',index = False, header = False)
    rr = []
    num = 4
    for k in tq.tqdm(range(2000)):
        num += 5
        r = []
        for i in range(255):
            r.append(pearson(key_inf_8[i], wave_data, num))
        rr.append(r)
    df = pd.DataFrame(rr)
    df.to_csv('key_8.csv',index = False, header = False)
    rr = []
    num = 4
    for k in tq.tqdm(range(2000)):
        num += 5
        r = []
        for i in range(255):
            r.append(pearson(key_inf_9[i], wave_data, num))
        rr.append(r)
    df = pd.DataFrame(rr)
    df.to_csv('key_9.csv',index = False, header = False)
    rr = []
    num = 4
    for k in tq.tqdm(range(2000)):
        num += 5
        r = []
        for i in range(255):
            r.append(pearson(key_inf_10[i], wave_data, num))
        rr.append(r)
    df = pd.DataFrame(rr)
    df.to_csv('key_10.csv',index = False, header = False)
    rr = []
    num = 4
    for k in tq.tqdm(range(2000)):
        num += 5
        r = []
        for i in range(255):
            r.append(pearson(key_inf_11[i], wave_data, num))
        rr.append(r)
    df = pd.DataFrame(rr)
    df.to_csv('key_11.csv',index = False, header = False)
    rr = []
    num = 4
    for k in tq.tqdm(range(2000)):
        num += 5
        r = []
        for i in range(255):
            r.append(pearson(key_inf_12[i], wave_data, num))
        rr.append(r)
    df = pd.DataFrame(rr)
    df.to_csv('key_12.csv',index = False, header = False)
    rr = []
    num = 4
    for k in tq.tqdm(range(2000)):
        num += 5
        r = []
        for i in range(255):
            r.append(pearson(key_inf_13[i], wave_data, num))
        rr.append(r)
    df = pd.DataFrame(rr)
    df.to_csv('key_13.csv',index = False, header = False)
    rr = []
    num = 4
    for k in tq.tqdm(range(2000)):
        num += 5
        r = []
        for i in range(255):
            r.append(pearson(key_inf_14[i], wave_data, num))
        rr.append(r)
    df = pd.DataFrame(rr)
    df.to_csv('key_14.csv',index = False, header = False)
    rr = []
    num = 4
    for k in tq.tqdm(range(2000)):
        num += 5
        r = []
        for i in range(255):
            r.append(pearson(key_inf_15[i], wave_data, num))
        rr.append(r)
    df = pd.DataFrame(rr)
    df.to_csv('key_15.csv',index = False, header = False)

