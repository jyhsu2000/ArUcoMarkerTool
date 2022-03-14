#!/usr/bin/env python
import math
import re
import time
from collections import deque

import PySimpleGUI as sg
import cv2
import cv2.aruco as aruco
import numpy as np
import pandas as pd
from PIL import Image, ImageTk, ImageOps
from scipy.spatial.transform import Rotation as R

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


def euler_from_quaternion(x, y, z, w):
    """
    Convert a quaternion into euler angles (roll, pitch, yaw)
    roll is rotation around x in radians (counterclockwise)
    pitch is rotation around y in radians (counterclockwise)
    yaw is rotation around z in radians (counterclockwise)
    """
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = math.atan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = math.asin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = math.atan2(t3, t4)

    return roll_x, pitch_y, yaw_z  # in radians


def main():
    default_aruco_dict_name = 'DICT_6X6_1000'
    selected_aruco_dict = ARUCO_DICT[default_aruco_dict_name]
    draw_crosshair = True
    draw_custom_marker = False
    draw_axis = False
    undistortion = True

    marker_length_mm = 103
    marker_length_mm = 21

    sg.theme('DefaultNoMoreNagging')

    empty_detected_marker_df = pd.DataFrame(columns=['id', '偏航(yaw)', '俯仰(pitch)', '滾動(roll)', '橫向偏移(cm)', '縱向偏移(cm)', '距離(cm)'])

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
            sg.Checkbox('Draw crosshair', key='draw_crosshair', enable_events=True, default=draw_crosshair),
            sg.Checkbox('Draw custom marker', key='draw_custom_marker', enable_events=True, default=draw_custom_marker),
            sg.Checkbox('Draw axis', key='draw_axis', enable_events=True, default=draw_axis),
            sg.Checkbox('Undistortion', key='undistortion', enable_events=True, default=undistortion),
            sg.Text('Marker length (mm):'),
            sg.Text(marker_length_mm, key='marker_length_mm'),
            sg.InputText(key='marker_length_mm_input', size=(10, 1), justification='center', enable_events=True, default_text=marker_length_mm),
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
            if event == 'draw_crosshair':
                draw_crosshair = values['draw_crosshair']
            if event == 'draw_custom_marker':
                draw_custom_marker = values['draw_custom_marker']
            if event == 'draw_axis':
                draw_axis = values['draw_axis']
            if event == 'undistortion':
                undistortion = values['undistortion']
            if event == 'marker_length_mm_input':
                marker_length_mm_input = values['marker_length_mm_input']
                if len(marker_length_mm_input) > 7:
                    marker_length_mm_input = marker_length_mm_input[:7]
                else:
                    if re.match(r'^\d*\.?\d*$', marker_length_mm_input):
                        if re.match(r'^\d+\.?\d*$', marker_length_mm_input):
                            marker_length_mm = float(marker_length_mm_input)
                    else:
                        marker_length_mm_input = marker_length_mm
                window['marker_length_mm_input'].update(marker_length_mm_input)
                window['marker_length_mm'].update(marker_length_mm)

            ret, frame = camera_looper.read()
            if not ret:
                continue
            # reize 圖片
            # frame = cv2.resize(frame,(640,360), interpolation=cv2.INTER_AREA)

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

                    rotation_vectors, translation_vectors, marker_points = aruco.estimatePoseSingleMarkers(markerCorner, marker_length_mm, camera_matrix, distortion_coefficients)
                    # 繪製軸線
                    if draw_axis:
                        aruco.drawAxis(frame, camera_matrix, distortion_coefficients, rotation_vectors, translation_vectors, 0.01)

                    rotation_matrix = np.eye(4)
                    rotation_matrix[0:3, 0:3] = cv2.Rodrigues(np.array(rotation_vectors[0][0]))[0]
                    r = R.from_matrix(rotation_matrix[0:3, 0:3])
                    quat = r.as_quat()

                    transform_rotation_x = quat[2]
                    transform_rotation_y = quat[1]
                    transform_rotation_z = quat[0]
                    transform_rotation_w = quat[3]

                    roll_x, yaw_y, pitch_z = euler_from_quaternion(transform_rotation_x,
                                                                   transform_rotation_y,
                                                                   transform_rotation_z,
                                                                   transform_rotation_w)

                    roll_x = math.degrees(roll_x)
                    yaw_y = math.degrees(yaw_y)
                    pitch_z = math.degrees(pitch_z)

                    # r = R.from_matrix(rotation_matrix[0:3, 0:3])
                    # quat = r.as_quat()
                    # deg = rotation_vectors[0][0][2] * 180 / np.pi
                    # R = np.zeros((3, 3), dtype=np.float64)
                    # cv2.Rodrigues(rotation_vectors, R)
                    # sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
                    # singular = sy < 1e-6
                    # if not singular:  # 偏航，俯仰，滾動
                    #     x = math.atan2(R[2, 1], R[2, 2])
                    #     y = math.atan2(-R[2, 0], sy)
                    #     z = math.atan2(R[1, 0], R[0, 0])
                    # else:
                    #     x = math.atan2(-R[1, 2], R[1, 1])
                    #     y = math.atan2(-R[2, 0], sy)
                    #     z = 0
                    # # 偏航，俯仰，滾動换成角度
                    # rx = np.rad2deg(x)
                    # ry = np.rad2deg(y)
                    # rz = np.rad2deg(z)

                    # 計算距離
                    # print("ID {} 偏航 {} 俯仰 {} 滾動 {} 距離 {}".format(markerID, rx, ry, rz, distance))
                    detected_markers.append(pd.DataFrame({
                        'id': markerID,
                        '偏航(yaw)': round(yaw_y),
                        '俯仰(pitch)': round(pitch_z),
                        '滾動(roll)': round(roll_x),
                        '橫向偏移(cm)': round(translation_vectors[0][0][0] / 10),
                        '縱向偏移(cm)': round(translation_vectors[0][0][1] / 10),
                        '距離(cm)': round(translation_vectors[0][0][2] / 10),
                    }, index=[0]))
                if detected_markers:
                    detected_marker_df = pd.concat([empty_detected_marker_df] + detected_markers).sort_values(by=['id'])
                else:
                    detected_marker_df = empty_detected_marker_df
                window['detected_marker_table'].update(values=detected_marker_df.values.tolist())
            else:
                window['detected_marker_table'].update(values=empty_detected_marker_df.values.tolist())

            if draw_crosshair:
                pen_radius = max(frame.shape[0], frame.shape[1]) / 256
                center_x, center_y = frame.shape[1] // 2, frame.shape[0] // 2
                percent = max(frame.shape[0], frame.shape[1]) / 100
                cv2.line(frame, (center_x, int(center_y - percent * 2)), (center_x, int(center_y - percent * 1)), (0, 0, 255), int(pen_radius))
                cv2.line(frame, (center_x, int(center_y + percent * 1)), (center_x, int(center_y + percent * 2)), (0, 0, 255), int(pen_radius))
                cv2.line(frame, (int(center_x - percent * 2), center_y), (int(center_x - percent * 1), center_y), (0, 0, 255), int(pen_radius))
                cv2.line(frame, (int(center_x + percent * 1), center_y), (int(center_x + percent * 2), center_y), (0, 0, 255), int(pen_radius))
                cv2.line(frame, (center_x, int(center_y - percent * 2)), (center_x, int(center_y + percent * 2)), (0, 255, 255), int(pen_radius // 3))
                cv2.line(frame, (int(center_x - percent * 2), center_y), (int(center_x + percent * 2), center_y), (0, 255, 255), int(pen_radius // 3))

            # img_bytes = cv2.imencode('.png', frame)[1].tobytes()
            image = Image.fromarray(frame[:, :, ::-1])
            resized_image = ImageOps.contain(image, (1080, 1080))
            img_bytes = ImageTk.PhotoImage(image=resized_image)
            window['image'].update(data=img_bytes)
            window['capture_fps'].update(f'Capture: {camera_looper.fps:.1f} fps')

            new_frame_time = time.time()
            show_fps = 1 / ((new_frame_time - recent_frame_time[0]) / recent_frame_count)
            recent_frame_time.append(new_frame_time)
            window['process_fps'].update(f'Process: {show_fps:.1f} fps')
    finally:
        camera_looper.stop()
        window.close()


if __name__ == '__main__':
    main()
