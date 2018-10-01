import random


def key_gen():
    k = 128
    key = int(random.getrandbits(k))
    #print(bin(key))
    return key

if __name__ == '__main__':
    temp = key_gen()
    #print(bin(temp))
