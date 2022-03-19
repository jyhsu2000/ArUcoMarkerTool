import functools
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np

from config import VIDEO_CAPTURE_SOURCE


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
    cv2_camera: cv2.VideoCapture = None

    def __init__(self):
        self.connect()

    @synchronized
    def read(self) -> Tuple[bool, np.ndarray]:
        ret, frame = self.cv2_camera.read()
        return ret, frame

    @synchronized
    def connect(self) -> None:
        print('Camera connecting...')
        self.cv2_camera = cv2.VideoCapture(VIDEO_CAPTURE_SOURCE)
        print('VideoCapture created')
        self.cv2_camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cv2_camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cv2_camera.set(cv2.CAP_PROP_FPS, 60)
        width = self.cv2_camera.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = self.cv2_camera.get(cv2.CAP_PROP_FRAME_HEIGHT)
        fps = self.cv2_camera.get(cv2.CAP_PROP_FPS)
        print(f'Resolution: {width} * {height}')
        print(f'FPS: {fps}')

    @synchronized
    def reconnect(self) -> None:
        print('Trying to reconnect...')
        self.release()
        self.connect()

    @synchronized
    def release(self) -> None:
        print('Camera releasing...')
        self.cv2_camera.release()


class CameraLooper(threading.Thread):
    is_running: bool = False
    camera: Camera = None
    ret: bool = None
    frame: np.ndarray = None
    recent_frame_count: int = 10
    recent_frame_time: deque = deque([0.0], maxlen=recent_frame_count)
    fps: float = 0.0

    def __init__(self):
        self.is_running = True
        threading.Thread.__init__(self)
        self.daemon = True
        self.camera = Camera()
        self.start()
        print('CameraLooper started')

    def run(self) -> None:
        if not self.is_running:
            return
        self.camera_loop()
        threading.Timer(0, self.run).start()

    def camera_loop(self) -> None:
        ret, frame = self.camera.read()
        if not ret and self.is_running:
            self.camera.reconnect()
            return

        new_frame_time = time.time()
        self.fps = 1 / ((new_frame_time - self.recent_frame_time[0]) / self.recent_frame_count)
        self.recent_frame_time.append(new_frame_time)

        self.ret = ret
        self.frame = frame

    def read(self) -> Tuple[bool, np.ndarray]:
        return self.ret, self.frame

    def stop(self) -> None:
        self.is_running = False
        self.camera.release()
        self.join()
        print('CameraLooper stopped')


@dataclass
class Chessboard:
    # 棋盤格模板規格
    w: int
    h: int
    square_size_mm: float

    @property
    def objp(self):
        # 世界坐標系中的棋盤格點,例如(0,0,0), (1,0,0), (2,0,0) ....,(8,5,0)，去掉Z坐標，記為二維矩陣
        objp = np.zeros((self.w * self.h, 3), np.float32)
        objp[:, :2] = np.mgrid[0:self.w, 0:self.h].T.reshape(-1, 2)
        square_size_mm = 24.6  # 棋盤格的寬度（務必正確設定，避免影響距離估算）
        objp = objp * square_size_mm
        return objp


def embed_img(src_img: np.array, dest_img: np.array, dest_points: list, alpha: float = 1) -> np.array:
    h, w, _ = dest_img.shape
    # 座標
    src_points = np.array([(0, 0), (0, h), (w, h), (w, 0)], dtype=np.float32)
    dest_points = np.array(dest_points, dtype=np.float32)
    # 計算轉換矩陣
    transformation_matrix = cv2.getPerspectiveTransform(src_points, dest_points)
    # 轉換圖片座標
    src_img = cv2.resize(src_img, (w, h))
    p_src_img = cv2.warpPerspective(src_img, transformation_matrix, (w, h))
    # 建立遮罩
    white_pad = np.full((h, w), 255, np.uint8)
    fg_mask = cv2.warpPerspective(white_pad, transformation_matrix, (w, h))
    bg_mask = cv2.bitwise_not(fg_mask)
    # 合成
    masked_fg_img = cv2.bitwise_or(p_src_img, p_src_img, mask=fg_mask)
    masked_bg_img = cv2.bitwise_or(dest_img, dest_img, mask=bg_mask)
    masked_img = cv2.bitwise_or(masked_fg_img, masked_bg_img)
    output = cv2.addWeighted(masked_img, alpha, dest_img, 1 - alpha, 0)

    return output


@functools.lru_cache(maxsize=999)
def create_text_pad(text: str = 'Text', text_color=(0, 255, 255), bg_color=(255, 0, 0)) -> np.array:
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_scale = 5
    thickness = 5

    (w, h), _ = cv2.getTextSize(text, font, text_scale, thickness)

    text_pad = np.zeros((h + 30, w, 3), np.uint8)
    text_pad[:] = bg_color
    text_pad = cv2.putText(text_pad, text, (5, h + 15), font, text_scale, text_color, thickness)

    return text_pad


def eat_events(window):
    """
    Simple, elegant fix
    "Eats" extra events created from updating tables.  Call it right after doing the update operation.
    Will eat however many events are created, fixing the issue where different platforms may create a different number of events.
    @see https://github.com/PySimpleGUI/PySimpleGUI/issues/4268#issuecomment-843423532
    """
    while True:
        event, values = window.read(timeout=0)
        if event == '__TIMEOUT__':
            break
    return


def eat_next_event(window, event_name: str):
    """
    Simple, elegant fix
    "Eats" extra events created from updating tables.  Call it right after doing the update operation.
    Will eat however many events are created, fixing the issue where different platforms may create a different number of events.
    @see https://github.com/PySimpleGUI/PySimpleGUI/issues/4268#issuecomment-843423532
    """
    event_queue = []
    while True:
        event, values = window.read(timeout=0)
        if event == event_name:
            break
        if event == '__TIMEOUT__':
            break

        # Queue other events for later
        event_queue.append((event, values[event]))

    # Write events back for main loop
    for event, value in event_queue:
        window.write_event_value(event, value)

    return


def save_coefficients(camera_matrix, distortion_coefficients, path='camera.yml'):
    """ Save the camera matrix and the distortion coefficients to given path/file. """
    cv_file = cv2.FileStorage(path, cv2.FILE_STORAGE_WRITE)
    cv_file.write('camera_matrix', camera_matrix)
    cv_file.write('distortion_coefficients', distortion_coefficients)
    # note you *release* you don't close() a FileStorage object
    cv_file.release()


def load_coefficients(path='camera.yml'):
    """ Loads camera matrix and distortion coefficients. """
    # FILE_STORAGE_READ
    cv_file = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)

    # note we also have to specify the type to retrieve other wise we only get a
    # FileNode object back instead of a matrix
    camera_matrix = cv_file.getNode('camera_matrix').mat()
    distortion_coefficients = cv_file.getNode('distortion_coefficients').mat()

    cv_file.release()
    return [camera_matrix, distortion_coefficients]
