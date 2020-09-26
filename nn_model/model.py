import numpy as np
from matplotlib import pyplot as plt
import h5py
import pickle
import os
from .nn_algorithm import *

def load_data():
    train_dataset = h5py.File('datasets/train_catvnoncat.h5','r') #train kümesi için features ve labels
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) #feature
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) #label

    test_dataset = h5py.File('datasets/test_catvnoncat.h5','r') #test kümesi için features ve labels
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) #feature
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) #label

    classes = np.array(test_dataset["list_classes"][:])

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

def dataset_review(train_set_x,train_set_y,test_set_x,test_set_y):
    print(train_set_x.shape) # train verisinin boyutları (209,64,64,3)
    print(train_set_x.shape[0]) # train versinin örnek sayısı (209)
    print(test_set_x.shape[0]) # test versinin örnek sayısı (50)
    print(train_set_y.shape) #train veri etiketinin boyutu (1,209)
    print(test_set_y.shape) #test veri etiketinin boyutu (1,50)
    # Not: Bu fonksiyonu fixed_and_normalized fonksiyonundan önce ham dataseti incelemek için kullanın.
    plt.imshow(train_set_x_orig[105]) #verisetindeki resimleri indexleri değiştirerek görebilirsiniz.
    plt.show()

def dataset_fixed_and_normalized(train_set_x,test_set_x):
    #piksel verilerini birleştirerek 2 boyutlu matris oluşturacağız. Her piksel feature.
    train_set_x = train_set_x.reshape(train_set_x.shape[0],-1).T #satırlar feature sütunlar örnek veri (12288,209)
    test_set_x = test_set_x.reshape(test_set_x.shape[0],-1).T #(12288,50)
    #verilireden daha iyi sonuç alamak için normalized işlemi uyguluyoruz.
    train_set_x = train_set_x/255
    test_set_x = test_set_x/255 
    return train_set_x, test_set_x

def catModel(X, Y, layer_dims, learning_rate=0.0075, epochs=3000, print_cost=False):
    costs = []
    #parameter initializa
    parameters = initilaize_parameters(layer_dims)
    #gradient descent
    for i in range(0,epochs):
        #forward propagation
        AL, caches = nn_forward_propagation(X,parameters)
        #compute cost
        cost = cost_function(AL,Y)
        #backward propagation
        grads = nn_backward_propagation(AL,Y,caches)
        #update parameters
        parameters = update_parameters(parameters,grads,learning_rate)
        #cost print
        if print_cost and i % 100 == 0:
            print("Cost after epoch {}: {}".format(i,cost))
        if i % 100 == 0:
            costs.append(cost)
    return parameters, costs

def file_parameter_write(parameter):#pickle ile pythın nesnelerini txt dosyasına yazıp okuyabiliyoruz.
    with open("parameter.txt",'wb') as handle:
        pickle.dump(parameter,handle)

def file_parameter_read(filepath = "parameter.txt"):
    if os.path.isfile(filepath):
        with open(filepath, 'rb') as handle:
            parameter = pickle.loads(handle.read())
        return parameter,True
    else:
        return 'notFile',False #dosya yok


"""
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_data()
#dataset_review(train_set_x_orig,train_set_y,test_set_x_orig,test_set_y)
train_set_x, test_set_x = dataset_fixed_and_normalized(train_set_x_orig,test_set_x_orig)
layer_dims = [12288,20,7,5,1]
param, cost = catModel(train_set_x,train_set_y,layer_dims,epochs=2500,print_cost=True)"""

"""
import csv

with open('employee_file2.csv', mode='w', newline='') as csv_file:
    fieldnames = ['x1', 'x2', 'x3']
    value = ['John Smith','Accounting','November']
    dic = {}
    for i in range(len(fieldnames)):
        dic.__setitem__('x'+str(i+1),value[i])
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

    writer.writeheader()
    writer.writerow(dic)
    writer.writerow({'x1': 'Erica Meyers', 'x2': 'IT', 'x3': 'March'})

import pandas as pd

df = pd.read_csv('employee_file2.csv')
a = np.array(df['x1'])
a = a.reshape((2,1))
print(a,a.shape)"""


