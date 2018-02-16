from random import shuffle
import numpy as np
import math
import cv2
from tqdm import tqdm
import os


def random_mini_batch(X, Y, minibatch_size = 8):
    minibatches = []
    m = X.shape[0]
    #Shuffle
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation,:,:,:]
    shuffled_Y = Y[permutation,:]
    
    num_complete_minibatches = math.floor(m/minibatch_size)
    for i in range(num_complete_minibatches):
        minibatch_x = shuffled_X[(i*minibatch_size):((i+1)*minibatch_size),:,:,:]
        minibatch_y = shuffled_Y[(i*minibatch_size):((i+1)*minibatch_size),:]
        minibatch = (minibatch_x,minibatch_y)
        minibatches.append(minibatch)
    if m % minibatch_size != 0:
        minibatch_x = shuffled_X[(num_complete_minibatches*minibatch_size):m,:,:,:]
        minibatch_y = shuffled_Y[(num_complete_minibatches*minibatch_size):m,:]
        minibatch = (minibatch_x,minibatch_y)
        minibatches.append(minibatch)
    return minibatches

def create_traindata(num_data = 1000, IMG_SIZE = 100, TRAIN_DIR = ".\\train"):
    training_data = []
    training_label = []
    for i, img in tqdm(enumerate(os.listdir(TRAIN_DIR))):
        if i < num_data:
            label = img.split('.')[-3]# for dog cat dataset
            path = os.path.join(TRAIN_DIR, img)
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
            training_label.append([np.array(label)])
            training_data.append([np.array(img)])
    print("{0} Files loaded".format(num_data))
    return training_data, training_label  