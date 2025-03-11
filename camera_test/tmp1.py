import pyrealsense2 as rs
# import numpy as np
# import cv2
import random
# import torch
from ours import *
import time
 
# model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
#
# model = torch.hub.load('ultralytics/yolov5', 'yolov5l6')
# model.conf = 0.5
def get_mid_pos(frame,box,depth_data,randnum):
    distance_list = []
    mid_pos = [(box[0] + box[2])//2, (box[1] + box[3])//2] #确定索引深度的中心像素位置
    min_val = min(abs(box[2] - box[0]), abs(box[3] - box[1])) #确定深度搜索范围
    #print(box,)
    for i in range(randnum):
        bias = random.randint(-min_val//4, min_val//4)
        dist = depth_data[int(mid_pos[1] + bias), int(mid_pos[0] + bias)]
        cv2.circle(frame, (int(mid_pos[0] + bias), int(mid_pos[1] + bias)), 4, (255,0,0), -1)
 
        #print(int(mid_pos[1] + bias), int(mid_pos[0] + bias))
        if dist:
            distance_list.append(dist)
    distance_list = np.array(distance_list)
    distance_list = np.sort(distance_list)[randnum//2-randnum//4:randnum//2+randnum//4] #冒泡排序+中值滤波
    cv2.imshow('1', frame)
 
 
 
 
    return np.mean(distance_list)
def dectshow(org_img, boxs,depth_data):
    img = org_img.copy()
    for box in boxs:
        cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
        intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics
        #
        # x1, y1 = (box[0] ,box[1])
        # print((x1,y1))
        # x1, y2 = (box[1] ,box[1])
        # x2, y1 = (box[0] ,box[1])
        # x2, y2 = (box[0] ,box[1])
 
        x1 = box[0]
        y1 = box[1]
        x2 = box[2]
        y2 = box[3]
 
        cv2.circle(img, (x1, y1), 5, (0, 0, 255), -1)  # 红色圆点
        cv2.circle(img, (x2, y1), 5, (0, 255, 0), -1)  # 绿色圆点
        cv2.circle(img, (x2, y2), 5, (255, 0, 0), -1)  # 蓝色圆点
 
        depth1 = depth_frame.get_distance(x1, y1)  # 获取第一个点的深度值
        depth2 = depth_frame.get_distance(x2, y1)  # 获取第二个点的深度值
        depth3 = depth_frame.get_distance(x2, y2)  # 获取第三个点的深度值
 
        point1 = rs.rs2_deproject_pixel_to_point(intrinsics, [x1, y1], depth1)
        point2 = rs.rs2_deproject_pixel_to_point(intrinsics, [x2, y1], depth2)
        point3 = rs.rs2_deproject_pixel_to_point(intrinsics, [x2, y2], depth3)
 
        cv2.imshow('1', img)
 
        # 计算两点之间的距离
        width = np.linalg.norm(np.array(point2) - np.array(point1))
        height = np.linalg.norm(np.array(point3) - np.array(point2))
        print("宽 is:", width, "m")
        print("长 is:", height, "m")
 
        cv2.putText(img, f'W: {str(width)[:4] }m H: {str(height)[:4]}m', (int(box[0]), int(box[1]) - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 2)
        # if not depth_frame.is_depth_frame():
        #     continue
        #
        # # 确定坐标值是否在depth_image的范围之内
        # if x1 >= 0 and x1 < depth_image.shape[1] and y1 >= 0 and y1 < depth_image.shape[0]:
        #     depth1 = depth_frame.get_distance(x1, y1)  # 获取第一个点的深度值
        #     continue
        # else:
        #     depth1 = None
        #
        #
        # # 确定坐标值是否在depth_image的范围之内
        # if x2 >= 0 and x2 < depth_image.shape[1] and y1 >= 0 and y1 < depth_image.shape[0]:
        #     depth2 = depth_frame.get_distance(x2, y1)  # 获取第二个点的深度值
        #     continue
        # else:
        #     depth2 = None
        #
        # # 确定坐标值是否在depth_image的范围之内
        # if x2 >= 0 and x2 < depth_image.shape[1] and y2 >= 0 and y2 < depth_image.shape[0]:
        #     depth3 = depth_frame.get_distance(x2, y2)  # 获取第三个点的深度值
        #     continue
        # else:
        #     depth3 = None
 
        # if x1 < 0 or x1 >= depth_image.shape[1] or y1 < 0 or y1 >= depth_image.shape[0]:
        #     # 处理超出范围的情况
        #     continue  # 跳过该点的处理
        # if x2 < 0 or x2 >= depth_image.shape[1] or y2 < 0 or y2 >= depth_image.shape[0]:
        #     # 处理超出范围的情况
        #     continue  # 跳过该点的处理
 
 
 
        dist = get_mid_pos(org_img, box, depth_data, 24)
        cv2.putText(img, box[4] + ' ' + str(dist / 1000)[:4] + 'm' ,
                    (int(box[0]), int(box[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        # if not depth_frame.is_depth_frame():
        #     continue
        #
        # # 确定坐标值是否在depth_image的范围之内
        # if x1 >= 0 and x1 < depth_image.shape[1] and y1 >= 0 and y1 < depth_image.shape[0]:
        #     depth1 = depth_frame.get_distance(x1, y1)  # 获取第一个点的深度值
        #     continue
        # else:
        #     depth1 = None
        #
        #
        # # 确定坐标值是否在depth_image的范围之内
        # if x2 >= 0 and x2 < depth_image.shape[1] and y1 >= 0 and y1 < depth_image.shape[0]:
        #     depth2 = depth_frame.get_distance(x2, y1)  # 获取第二个点的深度值
        #     continue
        # else:
        #     depth2 = None
        #
        # # 确定坐标值是否在depth_image的范围之内
        # if x2 >= 0 and x2 < depth_image.shape[1] and y2 >= 0 and y2 < depth_image.shape[0]:
        #     depth3 = depth_frame.get_distance(x2, y2)  # 获取第三个点的深度值
        #     continue
        # else:
        #     depth3 = None
        if x1 < 0 or x1 >= depth_image.shape[1] or y1 < 0 or y1 >= depth_image.shape[0]:
            # 处理超出范围的情况
            continue  # 跳过该点的处理
        if x2 < 0 or x2 >= depth_image.shape[1] or y2 < 0 or y2 >= depth_image.shape[0]:
            # 处理超出范围的情况
            continue  # 跳过该点的处理
    cv2.imshow('dec_img', img)
 
 
if __name__ == "__main__":
    # Configure depth and color streams
    onnx_path = 'yolov5s.onnx'
    model = Yolov5ONNX(onnx_path)
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 60)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 60)
    # Start streaming
    pipeline.start(config)
    pTime = 0
    try:
        while True:
            # Wait for a coherent pair of frames: depth and color
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            depth_intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics
            if not depth_frame or not color_frame:
                continue
            # Convert images to numpy arrays
 
            depth_image = np.asanyarray(depth_frame.get_data())
 
            color_image = np.asanyarray(color_frame.get_data())
 
            im = color_image.copy()    # 备份原图
 
 
            cTime = time.time()
            fps = 1 / (cTime - pTime)
            pTime = cTime
            cv2.putText(color_image, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                        (255, 0, 255), 3)
 
 
 
            results , boxs = model.detect(color_image)
 
            dectshow(im, boxs, depth_image)
 
            # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
            # Stack both images horizontally
            images = np.hstack((color_image, depth_colormap))
            # Show images
            cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('RealSense', images)
            key = cv2.waitKey(1)
            # Press esc or 'q' to close the image window
            if key & 0xFF == ord('q') or key == 27:
                cv2.destroyAllWindows()
                break
    finally:
        # Stop streaming
        pipeline.stop()