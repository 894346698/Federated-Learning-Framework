import pickle
import keras
import uuid
from paillier import *
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, LeakyReLU
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import datetime
import msgpack
import random
import codecs
import numpy as np
import json
import msgpack_numpy
from keras.layers import Dense, Dropout, Flatten, Activation
import sys
import time
from keras.models import load_model
from flask import *
from flask_socketio import SocketIO
from flask_socketio import *
length=64

priv1, pub1 = generate_keypair(length)#密钥长度
largeint=100000
np.set_printoptions(suppress=True)
with open("model_server_log.txt", "w") as f:
    f.write('model_server\n')
    f.write('密钥长度'+str(length)+'\n')
class GlobalModel(object):
    """docstring for GlobalModel"""
    def __init__(self):
        self.model = self.build_model()
        self.current_weights = self.model.get_weights()
        self.img_shape = (28,28,1)
        # for convergence check集合检查
        self.prev_train_loss = None
        # all rounds; losses[i] = [round#, timestamp, loss]
        # round# could be None if not applicable
        self.training_start_time = int(round(time.time()))  # 当前时间戳四舍五入
        self.start = int(round(time.time()))

    def build_model(self):
        raise NotImplementedError()

    # client_updates = [(w, n)..]
    def update_weights(self, client_weights):
        #new_weights = [np.zeros(w.shape) for w in self.current_weights]
        #for i in range(len(new_weights)):
            #new_weights[i] = client_weights[i]
        self.current_weights = client_weights
        print('服务器更新成功！')
class GlobalModel_MNIST_CNN(GlobalModel):
    def __init__(self):
        super(GlobalModel_MNIST_CNN, self).__init__()

    def build_model(self):
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
        self.img_shape = (28, 28, 1)
        model.add(Flatten(input_shape=self.img_shape))
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

class model_Server(object):
    MIN_NUM_WORKERS = 5
    MAX_NUM_ROUNDS = 50
    NUM_CLIENTS_CONTACTED_PER_ROUND = 5
    ROUNDS_BETWEEN_VALIDATIONS = 2
    init=True
    def __init__(self, global_model, host, port):
        self.global_model = global_model()
        self.update_client_sids = set()
        self.ready_client_sids = set()
        self.flag = 1
        self.first_id = 0
        self.secend_id = 0
        self.app = Flask(__name__)
        self.socketio = SocketIO(self.app)
        self.host = host
        self.port = port
        self.model_id = str(uuid.uuid4())  # 随机生成uuid

        #####
        # training states
        self.current_round = 0  # -1 for not yet started尚未开始
        self.current_round_client_updates = []
        self.eval_client_updates = []
        self.client_updates_weights = []
        self.cnt = 0
        self.i = 0
        self.a=[]
        self.register_handles()


    def register_handles(self):
        # single-threaded async, no need to lock

        @self.socketio.on('connect')
        def handle_connect():
            print(request.sid[0], "connected")

        @self.socketio.on('reconnect')
        def handle_reconnect():
            print(request.sid, "reconnected")

        @self.socketio.on('client_req_update')
        def handle_wake_up():
            self.ready_client_sids.add(request.sid)
            print("连接客户: ", request.sid,len(self.ready_client_sids))
            con=3#增加用户
            print("连接客户: ", request.sid, len(self.ready_client_sids)%con == 0  )

            if (len(self.ready_client_sids)%con == 0  ):
                while self.flag==0:
                    time.sleep(0.1)
                id = []
                for idd in self.ready_client_sids:
                    id.append(idd)
                self.ready_client_sids = set()
                emit('request_update', {
                    'model_id': self.model_id,
                    'round_number': self.current_round,
                    'current_weights': obj_to_pickle_string(self.global_model.current_weights),
                    'weights_format': 'pickle',
                    'priv': obj_to_pickle_string(priv1),
                    'pub': obj_to_pickle_string(pub1),

                }, room=id[0])
                print("客户端",id[0])
                emit('request_update', {
                    'model_id': self.model_id,
                    'round_number': self.current_round,
                    'current_weights': obj_to_pickle_string(self.global_model.current_weights),
                    'weights_format': 'pickle',
                    'priv': obj_to_pickle_string(priv1),
                    'pub': obj_to_pickle_string(pub1),
                }, room=id[1])
                print("客户端", id[1])
                emit('request_update', {
                    'model_id': self.model_id,
                    'round_number': self.current_round,
                    'current_weights': obj_to_pickle_string(self.global_model.current_weights),
                    'weights_format': 'pickle',
                    'priv': obj_to_pickle_string(priv1),
                    'pub': obj_to_pickle_string(pub1),
                }, room=id[2])
                print("客户端", id[2])

                self.current_round+=1


        @self.socketio.on('client_update')
        def handle_client_update(data):
            self.update_client_sids.add(request.sid)
            print("连接更新: ", request.sid, len(self.update_client_sids))
            print("连接更新: ", request.sid, len(self.update_client_sids) % 3 == 0)
            self.a.append(pickle_string_to_obj(data['weights']))

            if (len(self.a)%3 == 0):
                self.flag=0
                t1 = time.time()

                self.client_updates_weights=add(self.a[0],add(self.a[1],self.a[2]))
                self.global_model.update_weights(self.client_updates_weights)
                self.flag = 1
                t2=time.time()
                with open("model_server_log.txt", "a") as f:
                    f.write(str(self.i)+'轮循环聚合时长' + str(t2 - t1) + '\n')
                self.i+=1
                self.a=[]
                #GAN


    def start(self):
        self.socketio.run(self.app, host=self.host, port=self.port)


def obj_to_pickle_string(x):
    return codecs.encode(pickle.dumps(x), "base64").decode()
    # return msgpack.packb(x, default=msgpack_numpy.encode)
    # TODO: compare pickle vs msgpack vs json for serialization; tradeoff: computation vs network IO


def pickle_string_to_obj(s):
    return pickle.loads(codecs.decode(s.encode(), "base64"))  # 模型返序列化loads，编解码en/decode
    # return msgpack.unpackb(s, object_hook=msgpack_numpy.decode)




def add(a,b):
    #print()
    for i in range(len(a)):
        #b[i]=b[i].tolist()
        for j in range(len(a[i])):
            if type(a[i][j])==type(1):
                a[i][j] = b[i][j]*(a[i][j])
            else:
                for k in range(len(a[i][j])):
                    a[i][j][k]=b[i][j][k]*(a[i][j][k])
    return a


if __name__ == '__main__':
    # When the application is in debug mode the Werkzeug development server is still used
    # and configured properly inside socketio.run(). In production mode the eventlet web server
    # is used if available, else the gevent web server is used.

    server = model_Server(GlobalModel_MNIST_CNN, "192.168.1.111", 6001)
    print("listening on  192.168.1.111:6001");
    server.start()