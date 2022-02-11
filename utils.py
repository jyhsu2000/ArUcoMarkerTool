import functools
import threading
import time
from collections import deque

import cv2


class Singleton(type):
    __instances = {}
    __lock = threading.Lock()

    def __call__(cls, *args, **kwargs):
        if cls not in cls.__instances:
            with cls.__lock:
                if cls not in cls.__instances:
                    cls.__instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls.__instances[cls]


def synchronized(wrapped):
    __lock = threading.Lock()

    @functools.wraps(wrapped)
    def _wrap(*args, **kwargs):
        with __lock:
            return wrapped(*args, **kwargs)

    return _wrap


class Camera(metaclass=Singleton):
    camera = None

    def __init__(self):
        self.connect()

    @synchronized
    def read(self):
        ret, frame = self.camera.read()
        return ret, frame

    @synchronized
    def connect(self):
        print('Camera connecting...')
        self.camera = cv2.VideoCapture(0)
        print('VideoCapture created')
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.camera.set(cv2.CAP_PROP_FPS, 60)
        width = self.camera.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT)
        fps = self.camera.get(cv2.CAP_PROP_FPS)
        print(f'Resolution: {width} * {height}')
        print(f'FPS: {fps}')

    @synchronized
    def reconnect(self):
        print('Trying to reconnect...')
        self.camera.release()
        self.connect()


class CameraLooper(threading.Thread):
    camera = None
    window = None
    ret = None
    frame = None
    recent_frame_count = 10
    recent_frame_time = deque([0.0], maxlen=recent_frame_count)
    fps = 0.0

    def __init__(self, window):
        threading.Thread.__init__(self)
        self.window = window
        self.daemon = True
        self.camera = Camera()
        self.start()

    def run(self):
        self.camera_loop()
        threading.Timer(0, self.run).start()

    def camera_loop(self):
        ret, frame = self.camera.read()
        if not ret:
            self.camera.reconnect()
            return

        new_frame_time = time.time()
        self.fps = 1 / ((new_frame_time - self.recent_frame_time[0]) / self.recent_frame_count)
        self.recent_frame_time.append(new_frame_time)

        self.ret = ret
        self.frame = frame

    def read(self):
        return self.ret, self.frame
