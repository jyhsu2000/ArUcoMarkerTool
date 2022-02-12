import functools
import threading
import time
from collections import deque

import cv2
import numpy as np


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


def embed_img(src_img: np.array, dest_img: np.array, dest_points: list, alpha: float = 1) -> np.array:
    h, w, _ = dest_img.shape
    # 座標
    src_points = np.array([(0, 0), (0, h), (w, h), (w, 0)])
    dest_points = np.array(dest_points)
    # 計算轉換矩陣
    homo, status = cv2.findHomography(src_points, dest_points)
    # 轉換圖片座標
    src_img = cv2.resize(src_img, (w, h))
    p_src_img = cv2.warpPerspective(src_img, homo, (w, h))
    # 建立遮罩
    white_pad = np.full((h, w), 255, np.uint8)
    fg_mask = cv2.warpPerspective(white_pad, homo, (w, h))
    bg_mask = cv2.bitwise_not(fg_mask)
    # 合成
    masked_fg_img = cv2.bitwise_or(p_src_img, p_src_img, mask=fg_mask)
    masked_bg_img = cv2.bitwise_or(dest_img, dest_img, mask=bg_mask)
    masked_img = cv2.bitwise_or(masked_fg_img, masked_bg_img)
    output = cv2.addWeighted(masked_img, alpha, dest_img, 1 - alpha, 0)

    return output


def create_text_pad(text: str = 'Text', text_color=(0, 255, 255), bg_color=(255, 0, 0)) -> np.array:
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_scale = 5
    thickness = 5

    (w, h), _ = cv2.getTextSize(text, font, text_scale, thickness)

    text_pad = np.zeros((h + 30, w, 3), np.uint8)
    text_pad[:] = bg_color
    text_pad = cv2.putText(text_pad, text, (5, h + 15), font, text_scale, text_color, thickness)

    return text_pad
