#!/usr/bin/env python
import copy
import os.path
import time
from collections import deque

import PySimpleGUI as sg
import cv2
import imutils
import numpy as np
from PIL import Image, ImageTk

from utils import CameraLooper

calibration_images_path = './calibration_images'
thumbnail_size = (400, 300)

# 找棋盤格角點
# 設置尋找亞像素角點的參數，採用的停止準則是最大循環次數30和最大誤差容限0.001
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)  # 阈值
# 棋盤格模板規格
w = 9  # 10 - 1
h = 6  # 7  - 1
# 世界坐標系中的棋盤格點,例如(0,0,0), (1,0,0), (2,0,0) ....,(8,5,0)，去掉Z坐標，記為二維矩陣
objp = np.zeros((w * h, 3), np.float32)
objp[:, :2] = np.mgrid[0:w, 0:h].T.reshape(-1, 2)
objp = objp * 18.1  # 18.1 mm


def main():
    calibration_image_filenames = os.listdir(calibration_images_path)

    sg.theme('DefaultNoMoreNagging')

    layout = [
        [sg.Text('CalibrateCamera', size=(40, 1), justification='center', font='Helvetica 20', expand_x=True)],
        [
            sg.Column([
                # TODO: 清單應顯示檔案編號及總數量
                # TODO: 應有刪除功能
                [sg.Listbox(values=calibration_image_filenames, key='listbox', size=(40, 10), expand_x=True, expand_y=True, enable_events=True)],
                [sg.Image(filename='', key='thumbnail', size=(400, 1))],
                [sg.Image(filename='', key='thumbnail_with_marker', size=(400, 1))],
            ], expand_y=True),
            sg.Image(filename='', key='image'),
        ],
        [
            sg.Text('', key='capture_fps', size=(15, 1), justification='center', font='Helvetica 20'),
            sg.Text('', key='process_fps', size=(15, 1), justification='center', font='Helvetica 20'),
            sg.Column([
                [sg.Button('Capture', key='capture', font='Helvetica 20', enable_events=True)],
            ], element_justification='right', expand_x=True),
        ],
    ]

    window = sg.Window('CalibrateCamera', layout, location=(100, 100))

    camera_looper = CameraLooper(window)

    recent_frame_count = 10
    recent_frame_time = deque([0.0], maxlen=recent_frame_count)

    while True:
        event, values = window.read(timeout=0)
        if event == sg.WIN_CLOSED:
            return

        if event == 'listbox':
            # FIXME: 處理過慢，拖慢主執行緒
            image = cv2.imread(os.path.join(calibration_images_path, values['listbox'][0]))
            thumbnail_image = imutils.resize(image, width=thumbnail_size[0], height=thumbnail_size[1])
            window['thumbnail'].update(data=ImageTk.PhotoImage(image=Image.fromarray(thumbnail_image[:, :, ::-1])))

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # 找到棋盤格角點
            ret, corners = cv2.findChessboardCorners(gray, (w, h), None)
            image_with_marker = copy.deepcopy(image)
            if ret:
                # 在原角點的基礎上尋找亞像素角點
                # cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                # 追加進入世界三維點和平面二維點中
                # objpoints.append(objp)
                # imgpoints.append(corners)
                # 將角點在圖像上顯示
                cv2.drawChessboardCorners(image_with_marker, (w, h), corners, ret)
            thumbnail_image_with_marker = imutils.resize(image_with_marker, width=thumbnail_size[0], height=thumbnail_size[1])
            window['thumbnail_with_marker'].update(data=ImageTk.PhotoImage(image=Image.fromarray(thumbnail_image_with_marker[:, :, ::-1])))

        ret, frame = camera_looper.read()
        if not ret:
            continue

        if event == 'capture':
            print('capture')
            filename = time.strftime('%Y%m%d_%H%M%S', time.localtime()) + '.jpg'
            file_path = os.path.join(calibration_images_path, time.strftime('%Y%m%d_%H%M%S', time.localtime()) + '.jpg')
            print(f'{file_path=}')
            if not os.path.exists(os.path.dirname(file_path)):
                os.makedirs(os.path.dirname(file_path))
            cv2.imwrite(file_path, frame)

            # Update file list
            calibration_image_filenames = os.listdir(calibration_images_path)
            selected_index = calibration_image_filenames.index(filename)
            window['listbox'].update(values=calibration_image_filenames, set_to_index=selected_index, scroll_to_index=selected_index)
            # Trigger listbox event
            window.write_event_value('listbox', (filename,))

        # img_bytes = cv2.imencode('.png', frame)[1].tobytes()
        img_bytes = ImageTk.PhotoImage(image=Image.fromarray(frame[:, :, ::-1]))
        window['image'].update(data=img_bytes)
        window['capture_fps'].update(f'Capture: {camera_looper.fps:.1f} fps')

        new_frame_time = time.time()
        show_fps = 1 / ((new_frame_time - recent_frame_time[0]) / recent_frame_count)
        recent_frame_time.append(new_frame_time)
        window['process_fps'].update(f'Process: {show_fps:.1f} fps')


if __name__ == '__main__':
    main()
