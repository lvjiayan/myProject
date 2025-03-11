from openni import openni2
import numpy as np
import cv2
import time
import torch

from ultralytics import YOLO
import threading 
 
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

def hsv2bgr(h, s, v):
    h_i = int(h * 6)
    f = h * 6 - h_i
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)
    
    r, g, b = 0, 0, 0

    if h_i == 0:
        r, g, b = v, t, p
    elif h_i == 1:
        r, g, b = q, v, p
    elif h_i == 2:
        r, g, b = p, v, t
    elif h_i == 3:
        r, g, b = p, q, v
    elif h_i == 4:
        r, g, b = t, p, v
    elif h_i == 5:
        r, g, b = v, p, q

    return int(b * 255), int(g * 255), int(r * 255)

def random_color(id):
    h_plane = (((id << 2) ^ 0x937151) % 100) / 100.0
    s_plane = (((id << 3) ^ 0x315793) % 100) / 100.0
    return hsv2bgr(h_plane, s_plane, 1)
 
def mousecallback(event, x, y, flags, param):
    global click_x, click_y, distance_text, last_click_time
    if event == cv2.EVENT_LBUTTONDOWN:  # 单击事件，需要双击事件就是cv2.EVENT_LBUTTONDBLCLK
        click_x, click_y = x, y
        distance = dpt[y, x] / 10.0  # 若深度值是以毫米为单位，转换为厘米
        distance_text = f"Dis: {distance:.2f} cm"
        last_click_time = time.time()

# 检查CUDA是否可用
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Running on the GPU")
else:
    device = torch.device("cpu")
    print("Running on the CPU")


def yolo_infer():
    global cap

    while(True):
        ret, color_frame = cap.read()
        if not ret:
            break

        color_frame = cv2.flip(color_frame, 1)

        # 使用YOLOv5进行检测
        # print("----> 111")
        # results = model(color_frame)[0]
        # print("----> 222")
        # names   = results.names
        # boxes   = results.boxes.data.tolist()

        # for obj in boxes:
        #     left, top, right, bottom = int(obj[0]), int(obj[1]), int(obj[2]), int(obj[3])
        #     center_x = (left + right) / 2
        #     center_y = (right + bottom) / 2
        #     # center_distance = dpt[center_y, center_x] / 10.0   ##单位是厘米
        #     confidence = obj[4]
        #     label = int(obj[5])
        #     color = random_color(label)
        #     cv2.rectangle(color_frame, (left, top), (right, bottom), color=color ,thickness=2, lineType=cv2.LINE_AA)
        #     caption = f"{names[label]} {confidence:.2f}"
        #     w, h = cv2.getTextSize(caption, 0, 1, 2)[0]
        #     cv2.rectangle(color_frame, (left - 3, top - 33), (left + w + 10, top), color, -1)
        #     cv2.putText(color_frame, caption, (left, top - 5), 0, 1, (0, 0, 0), 2, 16)

        # cv2.imshow(WINDOW_NAME_COLOR, color_frame)

        # if cv2.waitKey(20) & 0xFF == ord('q'):
        #     break

        # print("yolo_infer is called")
        # time.sleep(0.1)
    

def depth_infer():

    global depth_stream
    cv2.namedWindow(WINDOW_NAME_DEPTH)
    cv2.setMouseCallback(WINDOW_NAME_DEPTH, mousecallback)

    while True:
        depth_frame = depth_stream.read_frame()
        dframe_data = np.array(depth_frame.get_buffer_as_triplet()).reshape([480, 640, 2])
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

        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

        time.sleep(0.1)


# 新版本加载模型
# model = YOLO("yolov8s.pt")
# model.eval()
# 将模型和数据移至GPU
# model = model.to(device)

openni2.initialize()
dev = openni2.Device.open_any()

print(dev.get_device_info())
depth_stream = dev.create_depth_stream()
depth_stream.start()
dev.set_image_registration_mode(True)

cap = cv2.VideoCapture(0)

if __name__ == "__main__":

    # yolo_infer()
    # depth_infer()

    t1 = threading.Thread(target = yolo_infer)
    t2 = threading.Thread(target = depth_infer)

    t1.start()
    t2.start()

    t1.join()
    t2.join()

    print("所有线程执行完毕")

    cap.release()
    depth_stream.stop()
    dev.close()
    cv2.destroyAllWindows()
    openni2.unload()
