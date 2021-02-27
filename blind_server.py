import pickle
import keras
import uuid
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
# https://github.com/lebedov/msgpack-numpy
from keras.layers import Dense, Dropout, Flatten, Activation
import sys
import time
from keras.models import load_model
from flask import *
from flask_socketio import SocketIO
from flask_socketio import *



class Server(object):
    def __init__(self, host, port):
        self.first_id = 0
        self.secend_id = 0
        self.third_id = 0
        self.ready_client_sids = set()
        self.app = Flask(__name__)
        self.socketio = SocketIO(self.app)
        self.host = host
        self.port = port
        self.register_handles()


    def register_handles(self):
        @self.socketio.on('connect')
        def handle_connect():
            print(request.sid, "connected")

        @self.socketio.on('reconnect')
        def handle_reconnect():
            print(request.sid, "reconnected")

        @self.socketio.on('disconnect')
        def handle_reconnect():
            print(request.sid, "disconnected")
            if request.sid in self.ready_client_sids:
                self.ready_client_sids.remove(request.sid)




        @self.socketio.on('client_ready')
        def handle_client_ready():
            print('客户端', request.sid, '请求更新')
            self.ready_client_sids.add(request.sid)
            if (len(self.ready_client_sids)%3 == 0):
                id = []
                for idd in self.ready_client_sids:
                    id.append(idd)
                emit('blind', {
                    'blind':0.5
                }, room=id[0])

                emit('blind', {
                    'blind': 2
                }, room=id[1])
                emit('blind', {
                    'blind': 1
                }, room=id[2])

    def start(self):
        self.socketio.run(self.app, host=self.host, port=self.port)
if __name__ == '__main__':
    server = Server("192.168.1.111", 6000)
    print("listening on 192.168.1.111:6000");
    server.start()