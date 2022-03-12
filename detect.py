#!/usr/bin/env python
import math
import time
from collections import deque

import PySimpleGUI as sg
import cv2
import cv2.aruco as aruco
import numpy as np
import pandas as pd
from PIL import Image, ImageTk

from utils import CameraLooper, embed_img, create_text_pad, load_coefficients

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
    default_aruco_dict_name = 'DICT_ARUCO_ORIGINAL'
    selected_aruco_dict = ARUCO_DICT[default_aruco_dict_name]
    draw_custom_marker = False
    draw_axis = False
    undistortion = True

    sg.theme('DefaultNoMoreNagging')

    empty_detected_marker_df = pd.DataFrame(columns=['marker_id', '偏航(yaw)', '俯仰(pitch)', '滾動(roll)', '距離(distance)'])

    layout = [
        [sg.Text('ArUcoMarkerDetection', size=(40, 1), justification='center', font='Helvetica 20', expand_x=True)],
        [
            sg.Image(filename='', key='image'),
            sg.Table(
                values=empty_detected_marker_df.values.tolist(),
                headings=empty_detected_marker_df.columns.tolist(),
                auto_size_columns=False,
                display_row_numbers=False,
                justification='left',
                # col_widths=[30, 10],
                # num_rows=10,
                key='detected_marker_table',
                expand_x=True,
                expand_y=True,
                enable_events=True,
            )
        ],
        [
            sg.Text('ArUco Dictionary:'),
            sg.Combo(values=list(ARUCO_DICT.keys()), key='dict_select', readonly=True, size=(40, 1),
                     default_value=default_aruco_dict_name, enable_events=True),
            sg.Checkbox('Draw custom marker', key='draw_custom_marker', enable_events=True, default=draw_custom_marker),
            sg.Checkbox('Draw axis', key='draw_axis', enable_events=True, default=draw_axis),
            sg.Checkbox('Undistortion', key='undistortion', enable_events=True, default=undistortion),
        ],
        [
            sg.Text('', key='capture_fps', size=(15, 1), justification='center', font='Helvetica 20'),
            sg.Text('', key='process_fps', size=(15, 1), justification='center', font='Helvetica 20'),
            sg.Text('', key='marker_count', size=(10, 1), justification='center', font='Helvetica 20'),
        ],
    ]

    window = sg.Window('ArUcoMarkerDetection', layout, location=(100, 100))

    camera_looper = CameraLooper()

    recent_frame_count = 10
    recent_frame_time = deque([0.0], maxlen=recent_frame_count)

    # 鏡頭校準相關參數
    camera_matrix, distortion_coefficients = load_coefficients()
    if camera_matrix is None:
        print('No "camera_matrix" in camera.yml. Use default value.')
        camera_matrix = np.array([[2000., 0., 1280 / 2.],
                                  [0., 2000., 720 / 2.],
                                  [0., 0., 1.]])
    if distortion_coefficients is None:
        print('No "distortion_coefficients" in camera.yml. Use default value.')
        distortion_coefficients = np.array([0., 0., 0., 0., 0.])

    print('camera_matrix:\n', camera_matrix)
    print('distortion_coefficients:\n', distortion_coefficients)

    try:
        while True:
            event, values = window.read(timeout=0)
            if event == sg.WIN_CLOSED:
                break

            if event == 'dict_select':
                selected_aruco_dict = ARUCO_DICT[values['dict_select']]
            if event == 'draw_custom_marker':
                draw_custom_marker = values['draw_custom_marker']
            if event == 'draw_axis':
                draw_axis = values['draw_axis']
            if event == 'undistortion':
                undistortion = values['undistortion']

            ret, frame = camera_looper.read()
            if not ret:
                continue

            if undistortion:
                # 畸變修正
                h, w = frame.shape[:2]
                new_camera_mtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, distortion_coefficients, (w, h), 0, (w, h))
                # 以下兩種方案皆可達到相同效果
                # @see https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_calibration/py_calibration.html#undistortion
                # 方案一：undistort
                frame = cv2.undistort(frame, camera_matrix, distortion_coefficients, None, new_camera_mtx)
                # 方案二：initUndistortRectifyMap
                # map_x, map_y = cv2.initUndistortRectifyMap(camera_matrix, distortion_coefficients, None, new_camera_mtx, (w, h), 5)
                # frame = cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR)
                # 裁剪 ROI
                x, y, w, h = roi
                frame = frame[y:y + h, x:x + w]

            aruco_dict = aruco.Dictionary_get(selected_aruco_dict)
            aruco_params = aruco.DetectorParameters_create()
            (corners, ids, rejected) = aruco.detectMarkers(frame, aruco_dict, parameters=aruco_params)

            window['marker_count'].update(f'{len(corners)} markers')

            if len(corners) > 0:
                # flatten the ArUco IDs list
                ids = ids.flatten()

                if not draw_custom_marker:
                    aruco.drawDetectedMarkers(frame, corners, ids)

                detected_markers = []
                # loop over the detected ArUCo corners
                for (markerCorner, markerID) in zip(corners, ids):
                    # extract the marker corners (which are always returned in top-left, top-right, bottom-right, and bottom-left order)
                    corners = markerCorner.reshape((4, 2))
                    (top_left, top_right, bottom_right, bottom_left) = corners
                    # convert each of the (x, y)-coordinate pairns to integers
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
                    # c_x = int((top_left[0] + bottom_right[0]) / 2.0)
                    # c_y = int((top_left[1] + bottom_right[1]) / 2.0)
                    # # cv2.circle(frame, (c_x, c_y), 4, (0, 0, 255), -1)

                    # draw the ArUco marker ID on the frame
                    # cv2.putText(frame, str(markerID),
                    #             (top_left[0], top_left[1] - 15),
                    #             cv2.FONT_HERSHEY_SIMPLEX,
                    #             0.5, (0, 255, 0), 1)

                    if draw_custom_marker:
                        text_pad = create_text_pad(str(markerID))
                        frame = embed_img(text_pad, frame, [top_left, bottom_left, bottom_right, top_right], alpha=0.7)

                    rvec, tvec, marker_points = aruco.estimatePoseSingleMarkers(markerCorner, 0.02, camera_matrix, distortion_coefficients)
                    # 繪製軸線
                    if draw_axis:
                        aruco.drawAxis(frame, camera_matrix, distortion_coefficients, rvec, tvec, 0.01)

                    # 計算角度
                    deg = rvec[0][0][2] * 180 / np.pi
                    R = np.zeros((3, 3), dtype=np.float64)
                    cv2.Rodrigues(rvec, R)
                    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
                    singular = sy < 1e-6
                    if not singular:  # 偏航，俯仰，滾動
                        x = math.atan2(R[2, 1], R[2, 2])
                        y = math.atan2(-R[2, 0], sy)
                        z = math.atan2(R[1, 0], R[0, 0])
                    else:
                        x = math.atan2(-R[1, 2], R[1, 1])
                        y = math.atan2(-R[2, 0], sy)
                        z = 0
                    # 偏航，俯仰，滾動换成角度
                    rx = x * 180.0 / 3.141592653589793
                    ry = y * 180.0 / 3.141592653589793
                    rz = z * 180.0 / 3.141592653589793

                    # 計算距離
                    # TODO: 距離計算跟 marker 的實際尺寸有關，須確認如何讓使用者設定
                    distance = ((tvec[0][0][2] + 0.02) * 0.13) * 100
                    # print("ID {} 偏航 {} 俯仰 {} 滾動 {} 距離 {}".format(markerID, rx, ry, rz, distance))
                    detected_markers.append(pd.DataFrame({
                        'marker_id': markerID,
                        '偏航(yaw)': round(rx),
                        '俯仰(pitch)': round(ry),
                        '滾動(roll)': round(rz),
                        '距離(distance)': round(distance, 2),
                    }, index=[0]))
                if detected_markers:
                    detected_marker_df = pd.concat([empty_detected_marker_df] + detected_markers).sort_values(by=['marker_id'])
                else:
                    detected_marker_df = empty_detected_marker_df
                window['detected_marker_table'].update(values=detected_marker_df.values.tolist())
            else:
                window['detected_marker_table'].update(values=empty_detected_marker_df.values.tolist())

            # img_bytes = cv2.imencode('.png', frame)[1].tobytes()
            img_bytes = ImageTk.PhotoImage(image=Image.fromarray(frame[:, :, ::-1]))
            window['image'].update(data=img_bytes)
            window['capture_fps'].update(f'Capture: {camera_looper.fps:.1f} fps')

            new_frame_time = time.time()
            show_fps = 1 / ((new_frame_time - recent_frame_time[0]) / recent_frame_count)
            recent_frame_time.append(new_frame_time)
            window['process_fps'].update(f'Process: {show_fps:.1f} fps')
    finally:
        camera_looper.stop()


if __name__ == '__main__':
    main()
