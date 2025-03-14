from openni import openni2
import numpy as np
import cv2
import time
import torch
 
DEVICE_INFO = {}
WINDOW_NAME_DEPTH = 'Depth Image'
WINDOW_NAME_COLOR = 'Color Image'
COLOR_MAP_TYPE = 8  # 可以尝试不同的色彩映射, 有0~11种渲染的模式,8 色彩鲜艳，2的色彩正常，0和11为黑白色
ALPHA_VALUE = 0.17
MAX_DISTANCE_CM = 800  # 最大有效距离，单位为厘米
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.5
FONT_COLOR = (0, 0, 0)
FONT_THICKNESS = 1
last_click_time = 0
click_x, click_y = -1, -1
distance_text = ""
 
 
def mousecallback(event, x, y, flags, param):
    global click_x, click_y, distance_text, last_click_time
    if event == cv2.EVENT_LBUTTONDOWN:  # 单击事件，需要双击事件就是cv2.EVENT_LBUTTONDBLCLK
        click_x, click_y = x, y
        # 确保坐标在有效范围内
        if 0 <= y < dpt.shape[0] and 0 <= x < dpt.shape[1]:
            distance = dpt[y, x] / 10.0  # 若深度值是以毫米为单位，转换为厘米
            distance_text = f"Dis: {distance:.2f} cm"
            last_click_time = time.time()

# 检查CUDA可用性
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Running on the GPU")
else:
    device = torch.device("cpu")
    print("Running on the CPU")
 
def initialize_depth_camera():
    """初始化深度相机，返回设备和深度流"""
    try:
        # 确保OpenNI2已初始化
        openni2.initialize()
        print("OpenNI2初始化成功")
        
        # 尝试打开设备
        dev = openni2.Device.open_any()
        if dev is None:
            print("无法打开深度相机设备")
            return None, None
            
        print("设备信息:", dev.get_device_info())
        
        # 创建深度流
        depth_stream = dev.create_depth_stream()
        if depth_stream is None:
            print("无法创建深度流")
            dev.close()
            return None, None
            
        # 设置图像配准模式
        dev.set_image_registration_mode(True)
        
        # 启动深度流
        depth_stream.start()
        print("深度流启动成功")
        
        return dev, depth_stream
    except Exception as e:
        print(f"初始化深度相机时出错: {e}")
        return None, None

def cleanup_resources(dev, depth_stream, cap):
    """清理资源"""
    try:
        if depth_stream is not None:
            depth_stream.stop()
            print("深度流已停止")
        
        if dev is not None:
            dev.close()
            print("设备已关闭")
            
        openni2.unload()
        print("OpenNI2已卸载")
        
        if cap is not None and cap.isOpened():
            cap.release()
            print("摄像头已释放")
            
        cv2.destroyAllWindows()
        print("所有窗口已关闭")
    except Exception as e:
        print(f"清理资源时出错: {e}")

if __name__ == "__main__":
    dev = None
    depth_stream = None
    cap = None
    dpt = np.zeros((480, 640), dtype=np.float32)  # 初始化全局dpt变量
    
    try:
        # 初始化深度相机
        dev, depth_stream = initialize_depth_camera()
        if dev is None or depth_stream is None:
            print("深度相机初始化失败，程序退出")
            exit(1)
        
        # 初始化普通摄像头
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("无法打开普通摄像头")
            cleanup_resources(dev, depth_stream, None)
            exit(1)
        
        # 创建窗口并设置鼠标回调
        cv2.namedWindow(WINDOW_NAME_DEPTH)
        cv2.setMouseCallback(WINDOW_NAME_DEPTH, mousecallback)
        cv2.namedWindow(WINDOW_NAME_COLOR)
        
        frame_count = 0
        start_time = time.time()
        
        while True:
            # 获取深度图像
            frame = depth_stream.read_frame()
            if frame is None:
                print("无法读取深度帧，重试中...")
                time.sleep(0.1)
                continue
                
            # 处理深度数据
            try:
                dframe_data = np.array(frame.get_buffer_as_triplet()).reshape([480, 640, 2])
                dpt1 = np.asarray(dframe_data[:, :, 0], dtype='float32')
                dpt2 = np.asarray(dframe_data[:, :, 1], dtype='float32')
                
                dpt2 *= 255
                dpt = dpt1 + dpt2
                
                # 检查深度数据是否有效
                if np.isnan(dpt).any() or np.isinf(dpt).any():
                    print("深度数据包含无效值，跳过此帧")
                    continue
                    
                dim_gray = cv2.convertScaleAbs(dpt, alpha=ALPHA_VALUE)
                depth_colormap = cv2.applyColorMap(dim_gray, COLOR_MAP_TYPE)
                
                # 显示距离信息
                if click_x >= 0 and click_y >= 0 and (time.time() - last_click_time) < 5:
                    depth_colormap = cv2.putText(depth_colormap, distance_text, (click_x, click_y), FONT, FONT_SCALE,
                                                FONT_COLOR, FONT_THICKNESS, cv2.LINE_AA)
                
                cv2.imshow(WINDOW_NAME_DEPTH, depth_colormap)
            except Exception as e:
                print(f"处理深度数据时出错: {e}")
                continue
            
            # 获取彩色图像
            ret, frame = cap.read()
            if ret:
                frame = cv2.flip(frame, 1)
                cv2.imshow(WINDOW_NAME_COLOR, frame)
            else:
                print("无法读取彩色帧")
            
            # 计算帧率
            frame_count += 1
            if frame_count % 30 == 0:
                end_time = time.time()
                fps = 30 / (end_time - start_time)
                print(f"FPS: {fps:.2f}")
                start_time = end_time
            
            # 检查键盘输入
            key = cv2.waitKey(10)
            if key & 0xFF == ord('q'):
                print("用户按下q键，程序退出")
                break
            elif key & 0xFF == ord('r'):
                print("用户按下r键，重置点击位置")
                click_x, click_y = -1, -1
        
    except Exception as e:
        print(f"程序运行时出错: {e}")
    finally:
        # 确保资源被正确释放
        cleanup_resources(dev, depth_stream, cap)