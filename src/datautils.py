import glob
import os
import numpy as np
import re
import math


def read_data_to_mat_MHE(path, file_count, measure_size, save_to_file):
    print("Reading Data from: ", path)

    n = measure_size
    m = 0
    i = 0
    
    for filename in glob.glob(os.path.join(path, '*.txt')):
        if i != file_count:
            print(os.path.join(os.getcwd(), filename))
            f = open(os.path.join(os.getcwd(), filename), 'r', encoding="utf-8-sig")
            text_f = f.read()
            m = m + text_f.count(" ") + text_f.count("\n") + 1 + n
            i = i + 1
        else:
            break
    
    m = np.ceil(m / n)
    data_mat = np.zeros((int(m) ,int(n * 88)), int)

    i = 0
    f_count = 0

    for filename in glob.glob(os.path.join(path, '*.txt')):
        if f_count != file_count:
            f = open(os.path.join(os.getcwd(), filename), 'r', encoding="utf-8-sig")
            durations = re.split("\s|\n", f.read())
            for s in durations:
                for c in s:
                    ascii = (ord(c) - 33)
                    if ascii > 87:
                        continue
                    data_mat[i // n][((i % n) * 88) + ascii] = 1
                i = i + 1
            if i % n != 0:
                i = i + (n - (i % n))
            f_count = f_count + 1
        else:
            break
    data_mat = data_mat[~np.all(data_mat == 0, axis=1)]
    print("Data Matrix: " + str(data_mat.shape))

    shuffle_block_sz = int(len(data_mat)/6)
    for c in range(shuffle_block_sz):
        np.random.shuffle(data_mat[c*shuffle_block_sz:(c+1)*shuffle_block_sz])

    if save_to_file:
        i = 0
        for c in range(int(math.ceil(len(data_mat)/50))):
            if i != file_count:
                #np.save("data/Dataset/" + str(c) +  ".npy", data_mat[c*50:50*(c+1)][:measure_size*88]) 
                np.save("data/numpy_arrays/1.npy", data_mat) 
                break

                i = i + 1
            else:
                break
    return data_mat


def lstm_dataset(path, measure_size, add_labels = False, num_files = 10000):
    data = read_data_to_mat_MHE(path, num_files, measure_size, False)
    data = np.unique(data, axis = 0)
    np.random.shuffle(data)
    if add_labels:
        x = data_to_timesequence(data, 88, False).astype(np.float64)
        y = x[0:, 1:]
        x = x[0:, :-1]
        return x, y
    else:
        return data_to_timesequence(data, 88, False).astype(np.float64)


def data_from_npz_MHE(path):
    arr = np.load(path)
    print("Loaded Data Matrix: " + str(arr.shape), " from: " + path)
    return arr

def data_to_timesequence(data, element_size, save_to_file, i = 0):
    new_data = np.empty((np.size(data, 0), int(np.size(data, 1) / element_size), element_size), int)
    for x in range(np.size(data, 0)):
        for y in range(int(np.size(data, 1) / element_size)):
            new_data[x][y] = data[x:x+1, y*88:88 * (y+1)]
    print("Timesequence Tensor size: ", str(new_data.shape))
    if save_to_file:
        np.save("data/lstm_data/data_lstm" + str(i) + ".npy", new_data)
    return new_data

def MHE_to_txt(vec, threshold, index = 0):
    vec = np.reshape(vec, np.prod(vec.shape))
    print("Max value in array: " + str(np.amax(vec)))
    #vec = vec / np.amax(vec)

    output_ascii = ""
    for i in range(len(vec)):
        if vec[i] >= threshold:
            output_ascii += (chr(i % 88 + 33))
        if i % 88 == 0:
            output_ascii += (" ")
    if index != 0:
        file = open("Input-00" + str(index) + ".txt", "w+")
    else:
        file = open("Input-001.txt", "w+")

    file.write(output_ascii)






                


     



