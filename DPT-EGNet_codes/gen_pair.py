import numpy as np
import random
import math
from scipy.io import wavfile
import librosa
import glob
import os
import h5py
import time


def gen_train_pair():

    train_clean_path = './datasets/female2_trainset_clean'
    train_noisy_path = './datasets/female2_trainset_noisy'
    train_mix_path = './datasets/female2_trainset_h5py'

    train_clean_name = sorted(os.listdir(train_clean_path))
    train_noisy_name = sorted(os.listdir(train_noisy_path))

    # print(train_clean_name)
    # print(train_noisy_name)

    for count in range(len(train_clean_name)):

        clean_name = train_clean_name[count]
        noisy_name = train_noisy_name[count]
        # print(clean_name, noisy_name)

        file_name = '%s_%d' % ('train_mix', count+1)
        train_writer = h5py.File(train_mix_path + '/' + file_name, 'w')

        clean_audio, sr = librosa.load(os.path.join(train_clean_path, clean_name), sr=16000)
        noisy_audio, sr1 = librosa.load(os.path.join(train_noisy_path, noisy_name), sr=16000)

        train_writer.create_dataset('noisy_raw', data=noisy_audio.astype(np.float32), chunks=True)
        train_writer.create_dataset('clean_raw', data=clean_audio.astype(np.float32), chunks=True)
        train_writer.close()

    # save .txt file
    print('sleep for 3 secs...')
    time.sleep(3)
    print('begin save .txt file...')
    train_file_list = sorted(glob.glob(os.path.join(train_mix_path, '*')))
    read_train = open("female2_train_file_list", "w+")

    for i in range(len(train_file_list)):
        read_train.write("%s\n" % (train_file_list[i]))

    read_train.close()
    print('making training data finished!')


def gen_val_pair():

    test_clean_path = './datasets/female2_valset_clean'
    test_noisy_path = './datasets/female2_valset_noisy'
    val_mix_path = './datasets/female2_valset_h5py'
    val_clean_name = sorted(os.listdir(test_clean_path))
    val_noisy_name = sorted(os.listdir(test_noisy_path))
    # test_log_path = './logfiles/log_testset.txt'
    # with open(test_log_path, 'r') as test_log_file:
    #     file_list = [line.split()[0] for line in test_log_file.readlines() if '2.5' in line]
    # print(file)
    # print(len(file))

    for count in range(len(val_clean_name)):

        clean_name = val_clean_name[count]
        noisy_name = val_noisy_name[count]
        # print(clean_name, noisy_name)

        file_name = '%s_%d' % ('train_mix', count+1)
        val_writer = h5py.File(val_mix_path + '/' + file_name, 'w')

        clean_audio, sr = librosa.load(os.path.join(test_clean_path, clean_name), sr=16000)
        noisy_audio, sr1 = librosa.load(os.path.join(test_noisy_path, noisy_name), sr=16000)

        val_writer.create_dataset('noisy_raw', data=noisy_audio.astype(np.float32), chunks=True)
        val_writer.create_dataset('clean_raw', data=clean_audio.astype(np.float32), chunks=True)
        val_writer.close()

    # save .txt file
    print('sleep for 3 secs...')
    time.sleep(3)
    print('begin save .txt file...')
    val_file_list = sorted(glob.glob(os.path.join(val_mix_path, '*')))
    read_val = open("female2_validation_file_list", "w+")

    for i in range(len(val_file_list)):
        read_val.write("%s\n" % (val_file_list[i]))

    read_val.close()
    print('making validation data finished!')


def gen_test_pair():

    test_clean_path = './datasets/female2_testset_clean'
    test_noisy_path = './datasets/female2_testset_noisy'
    test_mix_path = './datasets/female2_testset_h5py'

    test_clean_name = sorted(os.listdir(test_clean_path))
    test_noisy_name = sorted(os.listdir(test_noisy_path))

    # print(train_clean_name)
    # print(train_noisy_name)

    for count in range(len(test_clean_name)):

        clean_name = test_clean_name[count]
        noisy_name = test_noisy_name[count]
        # print(clean_name, noisy_name)
        file_name = '%s_%d' % ('test_mix', count + 1)
        train_writer = h5py.File(test_mix_path + '/' + file_name, 'w')

        clean_audio, sr = librosa.load(os.path.join(test_clean_path, clean_name), sr=16000)
        noisy_audio, sr1 = librosa.load(os.path.join(test_noisy_path, noisy_name), sr=16000)

        train_writer.create_dataset('noisy_raw', data=noisy_audio.astype(np.float32), chunks=True)
        train_writer.create_dataset('clean_raw', data=clean_audio.astype(np.float32), chunks=True)
        train_writer.close()

    # save .txt file
    print('sleep for 3 secs...')
    time.sleep(3)
    print('begin save .txt file...')
    test_file_list = sorted(glob.glob(os.path.join(test_mix_path, '*')))
    read_test = open("female2_test_file_list", "w+")

    for i in range(len(test_file_list)):
        read_test.write("%s\n" % (test_file_list[i]))

    read_test.close()
    print('making testing data finished!')


if __name__ == "__main__":

    gen_train_pair()
    gen_val_pair()
    gen_test_pair()












