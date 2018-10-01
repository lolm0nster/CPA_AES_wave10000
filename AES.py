# coding: UTF-8
import make_key
import numpy as np
import copy as cp

Sbox = (
    0x63, 0x7C, 0x77, 0x7B, 0xF2, 0x6B, 0x6F, 0xC5, 0x30, 0x01, 0x67, 0x2B, 0xFE, 0xD7, 0xAB, 0x76,
    0xCA, 0x82, 0xC9, 0x7D, 0xFA, 0x59, 0x47, 0xF0, 0xAD, 0xD4, 0xA2, 0xAF, 0x9C, 0xA4, 0x72, 0xC0,
    0xB7, 0xFD, 0x93, 0x26, 0x36, 0x3F, 0xF7, 0xCC, 0x34, 0xA5, 0xE5, 0xF1, 0x71, 0xD8, 0x31, 0x15,
    0x04, 0xC7, 0x23, 0xC3, 0x18, 0x96, 0x05, 0x9A, 0x07, 0x12, 0x80, 0xE2, 0xEB, 0x27, 0xB2, 0x75,
    0x09, 0x83, 0x2C, 0x1A, 0x1B, 0x6E, 0x5A, 0xA0, 0x52, 0x3B, 0xD6, 0xB3, 0x29, 0xE3, 0x2F, 0x84,
    0x53, 0xD1, 0x00, 0xED, 0x20, 0xFC, 0xB1, 0x5B, 0x6A, 0xCB, 0xBE, 0x39, 0x4A, 0x4C, 0x58, 0xCF,
    0xD0, 0xEF, 0xAA, 0xFB, 0x43, 0x4D, 0x33, 0x85, 0x45, 0xF9, 0x02, 0x7F, 0x50, 0x3C, 0x9F, 0xA8,
    0x51, 0xA3, 0x40, 0x8F, 0x92, 0x9D, 0x38, 0xF5, 0xBC, 0xB6, 0xDA, 0x21, 0x10, 0xFF, 0xF3, 0xD2,
    0xCD, 0x0C, 0x13, 0xEC, 0x5F, 0x97, 0x44, 0x17, 0xC4, 0xA7, 0x7E, 0x3D, 0x64, 0x5D, 0x19, 0x73,
    0x60, 0x81, 0x4F, 0xDC, 0x22, 0x2A, 0x90, 0x88, 0x46, 0xEE, 0xB8, 0x14, 0xDE, 0x5E, 0x0B, 0xDB,
    0xE0, 0x32, 0x3A, 0x0A, 0x49, 0x06, 0x24, 0x5C, 0xC2, 0xD3, 0xAC, 0x62, 0x91, 0x95, 0xE4, 0x79,
    0xE7, 0xC8, 0x37, 0x6D, 0x8D, 0xD5, 0x4E, 0xA9, 0x6C, 0x56, 0xF4, 0xEA, 0x65, 0x7A, 0xAE, 0x08,
    0xBA, 0x78, 0x25, 0x2E, 0x1C, 0xA6, 0xB4, 0xC6, 0xE8, 0xDD, 0x74, 0x1F, 0x4B, 0xBD, 0x8B, 0x8A,
    0x70, 0x3E, 0xB5, 0x66, 0x48, 0x03, 0xF6, 0x0E, 0x61, 0x35, 0x57, 0xB9, 0x86, 0xC1, 0x1D, 0x9E,
    0xE1, 0xF8, 0x98, 0x11, 0x69, 0xD9, 0x8E, 0x94, 0x9B, 0x1E, 0x87, 0xE9, 0xCE, 0x55, 0x28, 0xDF,
    0x8C, 0xA1, 0x89, 0x0D, 0xBF, 0xE6, 0x42, 0x68, 0x41, 0x99, 0x2D, 0x0F, 0xB0, 0x54, 0xBB, 0x16,
)

InvSbox = (
    0x52, 0x09, 0x6A, 0xD5, 0x30, 0x36, 0xA5, 0x38, 0xBF, 0x40, 0xA3, 0x9E, 0x81, 0xF3, 0xD7, 0xFB,
    0x7C, 0xE3, 0x39, 0x82, 0x9B, 0x2F, 0xFF, 0x87, 0x34, 0x8E, 0x43, 0x44, 0xC4, 0xDE, 0xE9, 0xCB,
    0x54, 0x7B, 0x94, 0x32, 0xA6, 0xC2, 0x23, 0x3D, 0xEE, 0x4C, 0x95, 0x0B, 0x42, 0xFA, 0xC3, 0x4E,
    0x08, 0x2E, 0xA1, 0x66, 0x28, 0xD9, 0x24, 0xB2, 0x76, 0x5B, 0xA2, 0x49, 0x6D, 0x8B, 0xD1, 0x25,
    0x72, 0xF8, 0xF6, 0x64, 0x86, 0x68, 0x98, 0x16, 0xD4, 0xA4, 0x5C, 0xCC, 0x5D, 0x65, 0xB6, 0x92,
    0x6C, 0x70, 0x48, 0x50, 0xFD, 0xED, 0xB9, 0xDA, 0x5E, 0x15, 0x46, 0x57, 0xA7, 0x8D, 0x9D, 0x84,
    0x90, 0xD8, 0xAB, 0x00, 0x8C, 0xBC, 0xD3, 0x0A, 0xF7, 0xE4, 0x58, 0x05, 0xB8, 0xB3, 0x45, 0x06,
    0xD0, 0x2C, 0x1E, 0x8F, 0xCA, 0x3F, 0x0F, 0x02, 0xC1, 0xAF, 0xBD, 0x03, 0x01, 0x13, 0x8A, 0x6B,
    0x3A, 0x91, 0x11, 0x41, 0x4F, 0x67, 0xDC, 0xEA, 0x97, 0xF2, 0xCF, 0xCE, 0xF0, 0xB4, 0xE6, 0x73,
    0x96, 0xAC, 0x74, 0x22, 0xE7, 0xAD, 0x35, 0x85, 0xE2, 0xF9, 0x37, 0xE8, 0x1C, 0x75, 0xDF, 0x6E,
    0x47, 0xF1, 0x1A, 0x71, 0x1D, 0x29, 0xC5, 0x89, 0x6F, 0xB7, 0x62, 0x0E, 0xAA, 0x18, 0xBE, 0x1B,
    0xFC, 0x56, 0x3E, 0x4B, 0xC6, 0xD2, 0x79, 0x20, 0x9A, 0xDB, 0xC0, 0xFE, 0x78, 0xCD, 0x5A, 0xF4,
    0x1F, 0xDD, 0xA8, 0x33, 0x88, 0x07, 0xC7, 0x31, 0xB1, 0x12, 0x10, 0x59, 0x27, 0x80, 0xEC, 0x5F,
    0x60, 0x51, 0x7F, 0xA9, 0x19, 0xB5, 0x4A, 0x0D, 0x2D, 0xE5, 0x7A, 0x9F, 0x93, 0xC9, 0x9C, 0xEF,
    0xA0, 0xE0, 0x3B, 0x4D, 0xAE, 0x2A, 0xF5, 0xB0, 0xC8, 0xEB, 0xBB, 0x3C, 0x83, 0x53, 0x99, 0x61,
    0x17, 0x2B, 0x04, 0x7E, 0xBA, 0x77, 0xD6, 0x26, 0xE1, 0x69, 0x14, 0x63, 0x55, 0x21, 0x0C, 0x7D,
)

Rcon = (0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1B, 0x36)
coefficient = [0x02, 0x03, 0x01, 0x01]

def get_key(round_key_10):
    seed_key = cp.deepcopy(round_key_10)
    for i in range(1,11):
        n = 4 * i - 1
        matrix_0 = []
        matrix_1 = []
        matrix_2 = []
        matrix_3 = []
        for h in range(4):
            for k in range(4):
                if h == 0:
                    matrix_0.append(seed_key[k][0])
                elif h==1:
                    matrix_1.append(seed_key[k][1])
                elif h==2:
                    matrix_2.append(seed_key[k][2])
                elif h==3:
                    matrix_3.append(seed_key[k][3])
        mt0 = np.array(matrix_0)
        mt1 = np.array(matrix_1)
        mt2 = np.array(matrix_2)
        mt3 = np.array(matrix_3)
        pre_key_3 = np.bitwise_xor(mt3,mt2)
        pre_key_2 = np.bitwise_xor(mt2,mt1)
        pre_key_1 = np.bitwise_xor(mt1,mt0)
        #SubByte & Rcon
        sword = []
        temp = []
        p_k_3 = cp.deepcopy(pre_key_3)
        rword = rot_word(p_k_3)
        for j in range(4):
            if j % 4== 0:
                sword.append( Sbox[rword[j]] )
                temp.append( sword[j] ^ Rcon[10-i] )
            else:
                sword.append( Sbox[rword[j]] )
                temp.append( sword[j] )
        temp_np = np.array(temp)
        pre_key_0 = np.bitwise_xor(temp_np,mt0)
        seed_key[0][0],seed_key[0][1],seed_key[0][2],seed_key[0][3] = pre_key_0[0], pre_key_1[0], pre_key_2[0],pre_key_3[0]
        seed_key[1][0],seed_key[1][1],seed_key[1][2],seed_key[1][3] = pre_key_0[1], pre_key_1[1], pre_key_2[1],pre_key_3[1]
        seed_key[2][0],seed_key[2][1],seed_key[2][2],seed_key[2][3] = pre_key_0[2], pre_key_1[2], pre_key_2[2],pre_key_3[2]
        seed_key[3][0],seed_key[3][1],seed_key[3][2],seed_key[3][3] = pre_key_0[3], pre_key_1[3], pre_key_2[3],pre_key_3[3]
    return seed_key


def key_expand(key):
    round_keys = text2matrix(key)
    print(round_keys)
    for i in range(1,11):
        n = 4 * i - 1
        matrix_0 = []
        matrix_1 = []
        matrix_2 = []
        matrix_3 = []
        for h in range(4):
            for k in range(4):
                if h == 0:
                    matrix_0.append(round_keys[4*(i-1)+k][0])
                elif h==1:
                    matrix_1.append(round_keys[4*(i-1)+k][1])
                elif h==2:
                    matrix_2.append(round_keys[4*(i-1)+k][2])
                elif h==3:
                    matrix_3.append(round_keys[4*(i-1)+k][3])
        mat = cp.deepcopy(matrix_3)
        rword = rot_word(mat)
        #SubByte & Rcon
        sword = []
        temp = []
        for j in range(4):
            if j % 4== 0:
                sword.append( Sbox[rword[j]] )
                temp.append( sword[j] ^ Rcon[i-1] )
            else:
                sword.append( Sbox[rword[j]] )
                temp.append( sword[j] )
        #key_gen
        x_0 = []
        x_1 = []
        x_2 = []
        x_3 = []
        for h in range(4):
            if h == 0:
                x_0.append(temp[0] ^ matrix_0[0])
                x_1.append(temp[1] ^ matrix_0[1])
                x_2.append(temp[2] ^ matrix_0[2])
                x_3.append(temp[3] ^ matrix_0[3])
            if h == 1:
                x_0.append(x_0[0] ^ matrix_1[0])
                x_1.append(x_1[0] ^ matrix_1[1])
                x_2.append(x_2[0] ^ matrix_1[2])
                x_3.append(x_3[0] ^ matrix_1[3])
            if h == 2:
                x_0.append(x_0[1] ^ matrix_2[0])
                x_1.append(x_1[1] ^ matrix_2[1])
                x_2.append(x_2[1] ^ matrix_2[2])
                x_3.append(x_3[1] ^ matrix_2[3])
            if h == 3:
                x_0.append(x_0[2] ^ matrix_3[0])
                x_1.append(x_1[2] ^ matrix_3[1])
                x_2.append(x_2[2] ^ matrix_3[2])
                x_3.append(x_3[2] ^ matrix_3[3])
        round_keys.append(x_0)
        round_keys.append(x_1)
        round_keys.append(x_2)
        round_keys.append(x_3)
    ##print(len(round_keys))
    return round_keys


def text2matrix(text):
    mask = pow(2,8)-1
    #print(bin(text))
    matrix = []
    for i in range(16):
        byte = (text >> (8 * (15 - i))) & mask
        if i < 4  :
            matrix.append([byte])
        else:
            num = i % 4
            matrix[num].append(byte)
    cp_matrix = cp.deepcopy(matrix)
    return cp_matrix

def rot_word(word):
    #print(word)
    temp = word[0]
    word[0] = word[1]
    word[1] = word[2]
    word[2] = word[3]
    word[3] = temp
    #print(word)
    return word

def encript(round_keys,plaintext):
    plain_state = text2matrix(plaintext)
    #print('plain_state:', plain_state)
    s = add_round_key(plain_state, round_keys[:4])
    for i in range(1,11):
    #for i in range(1,2):
        s = sub_bytes(s)
        #print('sub',s)
        s = shift_rows(s)
        #print('shif',s)
        if i != 10:
           s = mix_colums(s)
        s = add_round_key(s, round_keys[4*i : 4 * (i + 1)])
    return s

#encript process
#----------ここから-----------
def add_round_key(s,k):
    #print(k)
    for i in range(4):
        for j in range(4):
            s[i][j] ^= k[i][j]
    return s

def sub_bytes(s):
    for i in range(4):
        for j in range(4):
            s[i][j] = Sbox[s[i][j]]
    return s

def shift_rows(s):
    #s[0][1], s[1][1], s[2][1], s[3][1] = s[1][1], s[2][1], s[3][1], s[0][1]
    #s[0][2], s[1][2], s[2][2], s[3][2] = s[2][2], s[3][2], s[0][2], s[1][2]
    #s[0][3], s[1][3], s[2][3], s[3][3] = s[3][3], s[0][3], s[1][3], s[2][3]
    s[1][0], s[1][1], s[1][2], s[1][3] = s[1][1], s[1][2], s[1][3], s[1][0]
    s[2][0], s[2][1], s[2][2], s[2][3] = s[2][2], s[2][3], s[2][0], s[2][1]
    s[3][0], s[3][1], s[3][2], s[3][3] = s[3][3], s[3][0], s[3][1], s[3][2]
    return s

def mix_colums(s):
    s_1 = cp.deepcopy(s)
    for i in range(4):
        Tmp = s_1[0][i] ^ s_1[1][i] ^ s_1[2][i] ^ s_1[3][i]
        u = s_1[0][i]
        s_1[0][i] ^= Tmp ^ xtime( s_1[0][i] ^ s_1[1][i])
        s_1[1][i] ^= Tmp ^ xtime( s_1[1][i] ^ s_1[2][i])
        s_1[2][i] ^= Tmp ^ xtime( s_1[2][i] ^ s_1[3][i])
        s_1[3][i] ^= Tmp ^ xtime( s_1[3][i] ^ u)
    result = cp.deepcopy(s_1)
    return result

def xtime(a):
    return (((a << 1) ^(((a >> 7) & 1) * 0x1B)) & 0xFF)


def decrypt_9(round_keys, cipher_text):
    cipher_state = text2matrix(cipher_text)
    s = cp.deepcopy(cipher_state)
    s = add_round_key(s, round_keys)
    s = inv_shift_rows(s)
    #print(s)
    s = inv_sub_bytes(s)
    #print(s)
    return s

#----------ここまで----------

def decrypt(round_keys, chiphertext):
    #print(hex(chiphertext))
    chipher_state = text2matrix(chiphertext)
    print(chipher_state)
    s = cp.deepcopy(chipher_state)

    s = add_round_key(s,round_keys[40:])
    print('after_add',s)
    for i in range(9, -1, -1):
        s = inv_shift_rows(s)
        print(s)
        s = inv_sub_bytes(s)
        s = add_round_key(s,round_keys[4 * i : 4 * (i+1)])
        print('after_add',s)
        if (i != 0):
            s = inv_mixcolumns(s)
    return s

#decrypt process
#----------ここから----------
def inv_sub_bytes(s):
    for i in range(4):
        for j in range(4):
            s[i][j] = InvSbox[s[i][j]]
    return s

def inv_shift_rows(s):
    #s[0][1], s[1][1], s[2][1], s[3][1] = s[3][1], s[0][1], s[1][1], s[2][1]
    #s[0][2], s[1][2], s[2][2], s[3][2] = s[2][2], s[3][2], s[0][2], s[1][2]
    #s[0][3], s[1][3], s[2][3], s[3][3] = s[1][3], s[2][3], s[3][3], s[0][3]
    s[1][0], s[1][1], s[1][2], s[1][3] = s[1][3], s[1][0], s[1][1], s[1][2]
    s[2][0], s[2][1], s[2][2], s[2][3] = s[2][2], s[2][3], s[2][0], s[2][1]
    s[3][0], s[3][1], s[3][2], s[3][3] = s[3][1], s[3][2], s[3][3], s[3][0]
    return s

def inv_mixcolumns(s):
    for i in range(4):
        u = xtime(xtime(s[0][i] ^ s[2][i]))
        v = xtime(xtime(s[1][i] ^ s[3][i]))
        s[0][i] ^= u
        s[1][i] ^= v
        s[2][i] ^= u
        s[3][i] ^= v
    return mix_colums(s)
#----------ここまで----------

def matrix2text(matrix):
    text = 0
    for i in range(4):
        for j in range(4):
            text |= (matrix[i][j] << (120-8 * (4 * 1 + j)))
    return text

if __name__ == '__main__':
    #key = make_key.key_gen()
    key = 0x2b7e151628aed2a6abf7158809cf4f3c
    plaintext = 0x3243f6a8885a308d313198a2e0370734
    ciphertext = 0x3925841d02dc09fbdc118597196a0b32
    #print(type(key))
    round_keys = key_expand(key)
    #print('enc')
    #enc_result = encript(round_keys, plaintext)
    #print('dec')
    #dec_result = decrypt(round_keys, ciphertext)
    #print('enq_result:',enc_result)
    #print('dec_result:',dec_result)
    round_key_10 = 0xd014f9a8c9ee2589e13f0cc8b6630ca6
    round_key_10 = text2matrix(round_key_10)
    print(round_key_10)
    result = get_key(round_key_10)
    print(result)
