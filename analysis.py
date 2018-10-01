import AES as aes

path = './CPA/EMwavedata30000/CIPHERTEXT30000.txt'
path_1 = './CPA/EMwavedata30000/result_'


def text_write(HDw,count):
    path_f = path_1 + str(count) + '.txt'
    f_1 = open(path_f,mode = 'a')
    f_1.writelines(HDweight)
    f_1.write('\n')
    f_1.close()

if __name__ == '__main__':
    f = open(path)
    cipher = [s.strip() for s in f.readlines()]
    f.close()
    count = 0
    #for i in range(256):
    #    HDweight = []
    #    key = [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,i]]
    #    round_keys = key
    #    for j in range(10000):
    #        cipher_text = int(cipher[j],16)
    #        w = aes.text2matrix(cipher_text)
    #        round_9 = aes.decrypt_9(round_keys, cipher_text)
    #        n = w[3][2] ^ round_9[3][2]
    #        HDweight.append(bin(n).count('1'))
    #    HDweight = str(HDweight)
    #    text_write(HDweight)
    
    for k in range(4):
        for h in range(4):
            key = [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]
            for i in range(256):
                key[k][h] = i
                HDweight = []
                round_keys = key
                for j in range(30000):
                    cipher_text = int(cipher[j],16)
                    w = aes.text2matrix(cipher_text)
                    round_9 = aes.decrypt_9(round_keys, cipher_text)
                    if(k==0):
                        n = w[0][h] ^ round_9[0][h]
                    elif(k==1):
                        num = h + 1 
                        if(num > 3):
                            num = num - 4
                        n = w[1][num] ^ round_9[1][num]
                    elif(k==2):
                        num = h + 2 
                        if(num > 3):
                            num = num - 4
                        n = w[2][num] ^ round_9[2][num]
                    elif(k==3):
                        num = h + 3 
                        if(num > 3):
                            num = num - 4
                        n = w[3][num] ^ round_9[3][num]
                            
                    HDweight.append(bin(n).count('1'))
                HDweight = str(HDweight)
                text_write(HDweight , count)
            count += 1
