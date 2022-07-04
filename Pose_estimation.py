import pyrealsense2 as rs
import numpy as np
import cv2
import cv2.aruco as aruco


def Estimate_Pose_ArUco():
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
    parameters = aruco.DetectorParameters_create()
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
            # Wait for a coherent pair of frames: depth and color
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue
            # Convert images to numpy arrays
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            markercorners, markerIds, rejectedImagePoints = aruco.detectMarkers(color_image, dictionary,parameters=parameters)
            aruco.drawDetectedMarkers(color_image,markercorners,markerIds)
            mtx = np.load('mtx.npy')
            dist = np.load('dist.npy')
            rvecs,tvecs,_ = aruco.estimatePoseSingleMarkers(markercorners,0.05,mtx,dist)
            try:
                cv2.drawFrameAxes(color_image,mtx,dist,rvecs[0],tvecs[0],0.1)
            except:
                pass
            cv2.imshow('img',color_image)
            k = cv2.waitKey(1) & 0xFF
            if k == ord('s'):
                return rvecs, tvecs

    finally:
        pipeline.stop()

def catch_points(robot):
    R_target2cam = []
    t_target2cam = []
    R_gripper2base = []
    t_gripper2base = []
    #The position numbers >= 10,the three is least
    position = [
        [0.15, -0.34, 0.29, -10, 200, -10, 80],#Attention,this is tvecs and rvecs,not euler.
        [-0.01, -0.38, 0.25, -5, 185, -10, 80],
        [-0.15, -0.4, 0.25, -5, 160, -10, 80]
    ]
    for i in range(len(position)):
        Pos_dict = {}
        # Pos_dict['pos'] = position[i][:3]
        # Pos_dict['ros'] = position[i][3:6]
        # Pos_dict['Fixture'] = position[i][6]
        # robot.move_to(Pos_dict)
        R_gripper2base.append(cv2.Rodrigues(np.array(Pos_dict['ros'])/180*np.pi)[0])
        t_gripper2base.append(Pos_dict['pos'])
        rvecs,tvecs = Estimate_Pose_ArUco()
        R_target2cam.append(cv2.Rodrigues(np.array(rvecs)/180*np.pi)[0])
        t_target2cam.append(tvecs)
    np.save('R_gripper2base',R_gripper2base)
    np.save('t_gripper2base',t_gripper2base)
    np.save('R_target2cam',R_target2cam)
    np.save('t_target2cam',t_target2cam)


def HandEye_Calibration():
    R_target2cam = np.load('R_target2cam.npy')
    t_target2cam = np.load('t_target2cam.npy')
    R_gripper2base = np.load('R_gripper2base.npy')
    t_gripper2base = np.load('t_gripper2base.npy')
    R_cam2gripper,t_cam2gripper = cv2.calibrateHandEye(R_gripper2base,t_gripper2base,R_target2cam,t_target2cam,cv2.CALIB_HAND_EYE_TSAI)

    T_cam2gripper = np.eye(4)
    T_cam2gripper[:3,:3] = R_cam2gripper
    T_cam2gripper[:3,3] = np.array(t_cam2gripper).ravel()
    print(T_cam2gripper)
    np.save('T_cam2gripper',T_cam2gripper)
    print('File T_cam2gripper.npy had been saved!')

if __name__ == "__main__":
    catch_points()#go to position,press s key
    HandEye_Calibration()
