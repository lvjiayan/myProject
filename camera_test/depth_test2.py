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
        distance = dpt[y, x] / 10.0  # 若深度值是以毫米为单位，转换为厘米
        distance_text = f"Dis: {distance:.2f} cm"
        last_click_time = time.time()

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Running on the GPU")
else:
    device = torch.device("cpu")
    print("Running on the CPU")
 
if __name__ == "__main__":
    try:
        openni2.initialize()
        dev = openni2.Device.open_any()
        print(dev.get_device_info())
        depth_stream = dev.create_depth_stream()
        dev.set_image_registration_mode(True)
        depth_stream.start()
        cap = cv2.VideoCapture(0)
        cv2.namedWindow(WINDOW_NAME_DEPTH)
        cv2.setMouseCallback(WINDOW_NAME_DEPTH, mousecallback)
 
        while True:
            frame = depth_stream.read_frame()
            dframe_data = np.array(frame.get_buffer_as_triplet()).reshape([480, 640, 2])
            dpt1 = np.asarray(dframe_data[:, :, 0], dtype='float32')
            dpt2 = np.asarray(dframe_data[:, :, 1], dtype='float32')
 
            dpt2 *= 255
            dpt = dpt1 + dpt2
            dim_gray = cv2.convertScaleAbs(dpt, alpha=ALPHA_VALUE)
            # 对深度图像进行渲染
            depth_colormap = cv2.applyColorMap(dim_gray, COLOR_MAP_TYPE)
 
            if click_x >= 0 and click_y >= 0 and (time.time() - last_click_time) < 5:
                depth_colormap = cv2.putText(depth_colormap, distance_text, (click_x, click_y), FONT, FONT_SCALE,
                                             FONT_COLOR, FONT_THICKNESS, cv2.LINE_AA)
 
            cv2.imshow(WINDOW_NAME_DEPTH, depth_colormap)
 
            ret, frame = cap.read()
            if ret:
                frame = cv2.flip(frame, 1)
                cv2.imshow(WINDOW_NAME_COLOR, frame)
 
            key = cv2.waitKey(33)
            if key & 0xFF == ord('q'):
                break
        depth_stream.stop()
        dev.close()
        openni2.unload()
        cap.release()
        cv2.destroyAllWindows()
 
    except Exception as e:
        print(f"An error occurred: {e}")