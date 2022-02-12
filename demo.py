#!/usr/bin/env python
import time
from collections import deque

import PySimpleGUI as sg
import cv2.aruco as aruco
from PIL import Image, ImageTk

from utils import CameraLooper, embed_img, create_text_pad

ARUCO_DICT = {
    "DICT_4X4_50": aruco.DICT_4X4_50,
    "DICT_4X4_100": aruco.DICT_4X4_100,
    "DICT_4X4_250": aruco.DICT_4X4_250,
    "DICT_4X4_1000": aruco.DICT_4X4_1000,
    "DICT_5X5_50": aruco.DICT_5X5_50,
    "DICT_5X5_100": aruco.DICT_5X5_100,
    "DICT_5X5_250": aruco.DICT_5X5_250,
    "DICT_5X5_1000": aruco.DICT_5X5_1000,
    "DICT_6X6_50": aruco.DICT_6X6_50,
    "DICT_6X6_100": aruco.DICT_6X6_100,
    "DICT_6X6_250": aruco.DICT_6X6_250,
    "DICT_6X6_1000": aruco.DICT_6X6_1000,
    "DICT_7X7_50": aruco.DICT_7X7_50,
    "DICT_7X7_100": aruco.DICT_7X7_100,
    "DICT_7X7_250": aruco.DICT_7X7_250,
    "DICT_7X7_1000": aruco.DICT_7X7_1000,
    "DICT_ARUCO_ORIGINAL": aruco.DICT_ARUCO_ORIGINAL,
    "DICT_APRILTAG_16h5": aruco.DICT_APRILTAG_16h5,
    "DICT_APRILTAG_25h9": aruco.DICT_APRILTAG_25h9,
    "DICT_APRILTAG_36h10": aruco.DICT_APRILTAG_36h10,
    "DICT_APRILTAG_36h11": aruco.DICT_APRILTAG_36h11
}


def main():
    sg.theme('DefaultNoMoreNagging')

    layout = [
        [sg.Text('ArUcoMarkerDemo', size=(40, 1), justification='center', font='Helvetica 20', expand_x=True)],
        [sg.Image(filename='', key='image')],
        [
            sg.Text('ArUco Dictionary:'),
            sg.Combo(values=list(ARUCO_DICT.keys()), key='dict_select', readonly=True, size=(40, 1),
                     default_value='DICT_ARUCO_ORIGINAL', enable_events=True),
        ],
        [
            sg.Text('', key='capture_fps', size=(20, 1), justification='center', font='Helvetica 20'),
            sg.Text('', key='show_fps', size=(20, 1), justification='center', font='Helvetica 20'),
            sg.Text('', key='marker_count', size=(10, 1), justification='center', font='Helvetica 20'),
        ],
    ]

    window = sg.Window('ArUcoMarkerDemo', layout, location=(100, 100))

    camera_looper = CameraLooper(window)

    selected_aruco_dict = ARUCO_DICT['DICT_ARUCO_ORIGINAL']

    recent_frame_count = 10
    recent_frame_time = deque([0.0], maxlen=recent_frame_count)
    while True:
        event, values = window.read(timeout=0)
        if event == sg.WIN_CLOSED:
            return

        if event == 'dict_select':
            selected_aruco_dict = ARUCO_DICT[values['dict_select']]

        ret, frame = camera_looper.read()
        if not ret:
            continue

        aruco_dict = aruco.Dictionary_get(selected_aruco_dict)
        aruco_params = aruco.DetectorParameters_create()
        (corners, ids, rejected) = aruco.detectMarkers(frame, aruco_dict, parameters=aruco_params)

        window['marker_count'].update(f'{len(corners)} markers')

        if len(corners) > 0:
            # flatten the ArUco IDs list
            ids = ids.flatten()

            # aruco.drawDetectedMarkers(frame, corners, ids)

            # loop over the detected ArUCo corners
            for (markerCorner, markerID) in zip(corners, ids):
                # extract the marker corners (which are always returned in top-left, top-right, bottom-right, and bottom-left order)
                corners = markerCorner.reshape((4, 2))
                (top_left, top_right, bottom_right, bottom_left) = corners
                # convert each of the (x, y)-coordinate pairs to integers
                top_right = (int(top_right[0]), int(top_right[1]))
                bottom_right = (int(bottom_right[0]), int(bottom_right[1]))
                bottom_left = (int(bottom_left[0]), int(bottom_left[1]))
                top_left = (int(top_left[0]), int(top_left[1]))
                # draw the bounding box of the ArUCo detection
                # cv2.line(frame, top_left, top_right, (0, 255, 0), 2)
                # cv2.line(frame, top_right, bottom_right, (0, 255, 0), 2)
                # cv2.line(frame, bottom_right, bottom_left, (0, 255, 0), 2)
                # cv2.line(frame, bottom_left, top_left, (0, 255, 0), 2)

                # compute and draw the center (x, y)-coordinates of the ArUco marker
                c_x = int((top_left[0] + bottom_right[0]) / 2.0)
                c_y = int((top_left[1] + bottom_right[1]) / 2.0)
                # # cv2.circle(frame, (c_x, c_y), 4, (0, 0, 255), -1)

                # draw the ArUco marker ID on the frame
                # cv2.putText(frame, str(markerID),
                #             (top_left[0], top_left[1] - 15),
                #             cv2.FONT_HERSHEY_SIMPLEX,
                #             0.5, (0, 255, 0), 1)

                text_pad = create_text_pad(str(markerID))
                frame = embed_img(text_pad, frame, [top_left, bottom_left, bottom_right, top_right], alpha=0.7)

        # img_bytes = cv2.imencode('.png', frame)[1].tobytes()
        img_bytes = ImageTk.PhotoImage(image=Image.fromarray(frame[:, :, ::-1]))
        window['image'].update(data=img_bytes)
        window['capture_fps'].update(f'Capture: {camera_looper.fps:.1f} fps')

        new_frame_time = time.time()
        show_fps = 1 / ((new_frame_time - recent_frame_time[0]) / recent_frame_count)
        recent_frame_time.append(new_frame_time)
        window['show_fps'].update(f'Show: {show_fps:.1f} fps')


if __name__ == '__main__':
    main()
