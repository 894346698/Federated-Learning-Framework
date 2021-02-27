# -*- coding: UTF-8 -*-
from socketIO_client import SocketIO, LoggingNamespace
import numpy as np
import pandas as pd
import keras
import random
import time
import json
import pickle
import codecs
from keras.models import load_model
from keras.layers import Dense, Dropout, Flatten, Activation
from keras import Input, Model, Sequential
from keras.datasets import mnist
from keras.layers import Dense, LeakyReLU, BatchNormalization, Reshape
from keras.models import model_from_json
from keras.optimizers import Adam
from socketIO_client import SocketIO, LoggingNamespace
from model_server import obj_to_pickle_string, pickle_string_to_obj
import matplotlib.pyplot as plt
import threading
from paillier import *

np.set_printoptions(suppress=True)
np.set_printoptions(suppress=True)
largeint = 100000
flg_init = True

def noisyCount(sensitivety,epsilon):
    beta = sensitivety/epsilon
    u1 = np.random.random()
    u2 = np.random.random()
    if u1 <= 0.5:
        n_value = -beta*np.log((1.-u2)*2*beta)#laplace概率分布的反函数
    else:
        n_value = beta*np.log(u2*2*beta)
    return n_value

def laplace_mech(data,sensitivety,epsilon):
    data += noisyCount(sensitivety,epsilon)
    if data<-1:data=-1
    if data>1:data=1
    return data
def build_model():
    # ~5MB worth of parameters
    model = Sequential()
    # model.add(Conv2D(32, kernel_size=(3, 3),
    #                  activation='relu',
    #                  input_shape=(28, 28, 1)))
    # model.add(Conv2D(64, (3, 3), activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.25))
    # model.add(Flatten())
    # model.add(Dense(128, activation='relu'))
    # model.add(Dropout(0.5))
    # model.add(Dense(10, activation='softmax'))
    # model = Sequential([Dense(32, input_dim=784), Activation('relu'), Dense(16), \
    #                     Activation('relu'), Dense(10), Activation('softmax')])
    img_shape = (28, 28, 1)
    model.add(Flatten(input_shape=img_shape))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(1024))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(256))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(10, activation='sigmoid'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.SGD(lr=0.001, momentum=0.9, decay=0.0000001, nesterov=False),
                  metrics=['accuracy'])
    return model


class Client1(object):
    def __init__(self, server_host, server_port):
        self.sio = SocketIO(server_host, server_port)
        self.register_handles()
        print("启动客户端Client1（sent wakeup）")
        self.sio.emit('client_ready')
        self.sio.wait()
        self.blindness
        self.req_num = 1

    def register_handles(self):
        def req_train():
            # self.req_num = self.req_num+1
            print('继续请求第' + '次更新：')
            self.sio.emit('client_ready')

        def on_connect():
            print('connect')

        def on_disconnect():
            print('disconnect')

        def on_reconnect():
            print('reconnect')

        def on_blind(*args):
            req = args[0]
            self.blindness = req['blind']
            print('blind=', self.blindness)
            self.sio._close()

        self.sio.on('connect', on_connect)
        self.sio.on('disconnect', on_disconnect)
        self.sio.on('reconnect', on_reconnect)
        self.sio.on('blind', on_blind)


class Client2(object):
    def __init__(self, server_host, server_port):
        self.sio = SocketIO(server_host, server_port)
        self.register_handles()
        print("启动客户端Client2（sent wakeup）")
        self.sio.emit('client_req_update')
        self.sio.wait()
        self.weights
        self.priv
        self.pub
        self.req_num = 1

    def register_handles(self):
        def on_connect():
            print('connect')

        def on_disconnect():
            print('disconnect')

        def on_reconnect():
            print('reconnect')

        def on_request_update(*args):
            req = args[0]
            round_number = req['round_number']
            self.priv = pickle_string_to_obj(req['priv'])
            self.pub = pickle_string_to_obj(req['pub'])
            print("轮数", round_number)
            if int(round_number) == 0:
                self.weights = pickle_string_to_obj(req['current_weights'])
            else:
                t11 = time.time()
                self.weights = client_decrypt(self.priv, self.pub, pickle_string_to_obj(req['current_weights']))
                t22 = time.time()
                with open("client3_log.txt", "a") as f:
                    f.write('解密时长' + str(t22 - t11) + '\n')
            # print('current_weights=', self.weights)
            self.sio._close()

        self.sio.on('connect', on_connect)
        self.sio.on('disconnect', on_disconnect)
        self.sio.on('reconnect', on_reconnect)
        self.sio.on('request_update', on_request_update)


class Client3(object):
    def __init__(self, server_host, server_port, weights):
        self.sio = SocketIO(server_host, server_port)
        self.register_handles()
        print("Client1启动客户端3（sent wakeup）")

        self.sio.emit('client1_ready', {
            'weights': obj_to_pickle_string(weights),
        }

                      )

    def register_handles(self):
        def on_connect():
            print('connect')

        def on_disconnect():
            print('disconnect')

        def on_reconnect():
            print('reconnect')

        self.sio.on('connect', on_connect)
        self.sio.on('disconnect', on_disconnect)
        self.sio.on('reconnect', on_reconnect)


class Client_server(object):
    def __init__(self, server_host, server_port, weights):
        self.sio = SocketIO(server_host, server_port)
        self.register_handles()
        print("启动客户端Client4（client_update）")

        self.weights = weights
        self.req_num = 1
        self.sio.emit('client_update', {
            'weights': obj_to_pickle_string(weights)
        })

    def register_handles(self):
        def req_train():
            # self.req_num = self.req_num+1
            print('继续请求第' + '次更新：')
            self.sio.emit('client_ready')

        def on_connect():
            print('connect')

        def on_disconnect():
            print('disconnect')

        def on_reconnect():
            print('reconnect')

        def on_request_update(*args):
            req = args[0]
            self.weights = req['current_weights']
            # print('current_weights=', self.weights)
            self.sio._close()

        self.sio.on('connect', on_connect)
        self.sio.on('disconnect', on_disconnect)
        self.sio.on('reconnect', on_reconnect)
        self.sio.on('request_update', on_request_update)


def train(weights):
    batch_size = 2000#总数据量
    model_config = (weights)
    local_model.set_weights(model_config)
    (X_train, Y_train), (_, _) = mnist.load_data()
    X_train = X_train / 127.5 - 1.
    X_train = np.expand_dims(X_train, axis=3)
    idx = np.random.randint(0, X_train.shape[0], batch_size)
    x_train = X_train[idx]

    for i in range(batch_size):
        for j in range(28):
            for k in range(28):
                x_train[i][j][k][0]=laplace_mech(x_train[i][j][k][0],0.01,0.8)
    y_t = np.zeros((batch_size, 10))
    for ii in range(batch_size):
        for j in range(10):
            if (Y_train[idx[ii]] == j):
                y_t[ii][j] = 1
    y_train = y_t


    idx = np.random.randint(0, X_train.shape[0], 1000)
    x_test = X_train[idx]
    y_t1 = np.zeros((1000, 10))
    for ii in range(1000):
        for j in range(10):
            if (Y_train[idx[ii]] == j):
                y_t1[ii][j] = 1
    y_test = y_t1
    print('###本地训练begin train_one_round###')
    local_model.compile(loss=keras.losses.categorical_crossentropy,
                        optimizer=keras.optimizers.Adadelta(),
                        metrics=['accuracy'])
    local_model.fit(x_train, y_train,
                    epochs=2,#训练轮数
                    batch_size=32,#每次训练数据
                    verbose=1,
                    validation_data=(x_test, y_test)
                    )
    # print('###fit###')
    score = local_model.evaluate(x_test, y_test, verbose=1)
    print('Train loss:', score[0])
    print('Train accuracy:', score[1])
    with open("client3_log.txt", "a") as f:
        f.write('准确率：' + str(score[1]) + '\n')
    # print(local_model.get_weights())
    print("上传参数", local_model.get_weights()[0][0][0])
    return local_model.get_weights()


def client_encrypt(pub, w):
    print("clirnt开始加密")
    print("加密上传", type(w[0]), w[0][0][0])
    print("加密上传", type(w[0][0]), encrypt1(pub, int(w[0][0][0] * largeint)))
    wht = []
    for i in range(len(w)):
        w[i] = w[i].tolist()
        for j in range(len(w[i])):
            # print(i,j,type(w[0][0]))
            if type(w[i][j]) == type(1.1):
                w[i][j] = encrypt1(pub, int(w[i][j] * largeint))
            else:
                for k in range(len(w[i][j])):
                    w[i][j][k] = encrypt1(pub, int(w[i][j][k] * largeint))
                    # w[i][j][k]=np.array(w[i][j][k])
            # print(i, j, type(w[0][0]))
    # print("w2[][]", type(w[0][0]), w[0][0][0])
    print("client加密完成")
    return w


def client_decrypt(priv, pub, w):
    print("client解密开始")
    print("下载",w[0][0][0])
    print("下载",w[0][0][0])
    for i in range(len(w)):
        #w[i]=w[i].tolist()
        for j in range(len(w[i])):
            if type(w[i][j]) == type(1):
                w[i][j] = decrypt1(priv, pub, (w[i][j])) / largeint / 3 #解密
            else:
                for k in range(len(w[i][j])):
                    w[i][j][k] = decrypt1(priv, pub, (w[i][j][k])) / largeint / 3
        w[i] = np.array(w[i])
    print("client解密完成")
    print("下载解密完成", w[0][0][0])

    return w
def blind(w,a):
    for i in range(len(w)):
        # w[i]=w[i].tolist()
        for j in range(len(w[i])):
            if type(w[i][j]) == type(1):
                b = (w[i][j])*a
            else:
                for k in range(len(w[i][j])):
                    b =  (w[i][j][k])*a
    return w
if __name__ == "__main__":
    local_model = build_model()
    with open("client3_log.txt", "w") as f:
        f.write('client3_log\n')
        f.write('数据量2000\n')
    for i in range(200):
        t_tol_begin=time.time()
        print(i, "轮循环开始")
        t1=time.time()
        with open("client3_log.txt", "a") as f:
            f.write('\n\n#####'+str(i)+'轮循环开始#####\n')
        client1 = Client1("192.168.1.111", 6000)
        t2= time.time()
        with open("client3_log.txt", "a") as f:
            f.write('传输盲化因子通信开销'+str(t2-t1)+'\n')

        t1 = time.time()
        client2 = Client2("192.168.1.111", 6001)
        t2 = time.time()
        with open("client3_log.txt", "a") as f:

            f.write('解密时长+传输模型参数通信开销'+str(t2-t1)+'\n')
        t1 = time.time()
        weights=train(client2.weights)
        t2 = time.time()
        with open("client3_log.txt", "a") as f:
            f.write('训练时长' + str(t2 - t1) + '\n')
        t1 = time.time()
        weights = client_encrypt(client2.pub,weights )
        t2 = time.time()
        with open("client3_log.txt", "a") as f:
            f.write('加密时长'+str(t2-t1)+'\n')
        t1 = time.time()

        weights1 = blind(weights,client1.blindness)

        t2 = time.time()
        with open("client3_log.txt", "a") as f:
            f.write('盲化时长'+str(t2-t1)+'\n')
        t1 = time.time()
        client4 = Client_server("192.168.1.111", 6001, weights)
        t2 = time.time()
        with open("client3_log.txt", "a") as f:
            f.write('更新上传时长' + str(t2 - t1) + '\n')
        print(i, "轮循环结束")
        t_tol_end = time.time()
        with open("client3_log.txt", "a") as f:

            f.write(str(i)+'轮循总时长' + str(t_tol_end - t_tol_begin) + '\n')
            f.write('#####' + str(i) + '轮循环结束#####\n')
