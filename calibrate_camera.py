import os
import pyrealsense2 as rs
import numpy as np
import cv2
import glob

def convert_intrinsic_mat(ppx, ppy, fx, fy):
    mtx = np.zeros((3, 3))
    mtx[0][0] = fx
    mtx[1][1] = fy
    mtx[0][2] = ppx
    mtx[1][2] = ppy
    return mtx
def get_images():
    SIGN = 0
    pipeline = rs.pipeline()
    config = rs.config()
    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()
    device_product_line = str(device.get_info(rs.camera_info.product_line))
    found_rgb = False
    for s in device.sensors:
        if s.get_info(rs.camera_info.name) == 'RGB Camera':
            found_rgb = True
            break
    if not found_rgb:
        print("The demo requires Depth camera with Color sensor")
        exit(0)

    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    if device_product_line == 'L500':
        config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
    else:
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Start streaming
    pipeline.start(config)
    try:
        while True:
            camera_matrix = []
            # Wait for a coherent pair of frames: depth and color
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            depth_profile = rs.video_stream_profile(depth_frame.get_profile())
            color_profile = rs.video_stream_profile(color_frame.get_profile())
            color_intrinsic = color_profile.get_intrinsics()
            depth_intrinsic = depth_profile.get_intrinsics()
            color_mtx = convert_intrinsic_mat(color_intrinsic.ppx, color_intrinsic.ppy, color_intrinsic.fx,
                                              color_intrinsic.fy)
            depth_mtx = convert_intrinsic_mat(depth_intrinsic.ppx, depth_intrinsic.ppy, depth_intrinsic.fx,
                                              depth_intrinsic.fy)
            camera_matrix.append(color_mtx)
            camera_matrix.append(depth_mtx)
            if not depth_frame or not color_frame:
                continue
            # Convert images to numpy arrays
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            cv2.imshow('img', color_image)
            k = cv2.waitKey(1) & 0xFF
            if k == ord('s'):
                try:
                    os.makedirs('./images')
                except:
                    pass
                cv2.imwrite(f'./images/{SIGN}.png', color_image)
                SIGN = SIGN + 1
                if SIGN == 2:
                    break

    finally:
        pipeline.stop()
def Calibrate_camera():
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objp = np.zeros((8*6,3),np.float32) #8,6 corners nums
    objp[:, :2] = np.mgrid[0:8, 0:6].T.reshape(-1, 2)

    objpoints = []
    imgpoints = []
    images = glob.glob('./images/*.png')
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        ret,corners = cv2.findChessboardCorners(gray, (8,6), None)
        if ret == True:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners)
            # Draw and display the corners
            cv2.drawChessboardCorners(img, (8, 6), corners2, ret)
            cv2.imshow('img', img)
            cv2.waitKey(500)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    np.save('mtx',mtx)
    np.save('dist',dist)
if __name__ == '__main__':
    get_images()
    Calibrate_camera()