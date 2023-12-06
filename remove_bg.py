import cv2
import numpy as np
import os

# 加载 YOLOv3 模型
net = cv2.dnn.readNet("yolo/yolov3.weights", "yolo/yolov3.cfg")

# 获取 COCO 数据集的类别名称
with open("yolo/coco.names", "r") as f:
    classes = f.read().strip().split("\n")


# 读取img目录下的全部文件
image_files = os.listdir("img")

for image_file in image_files:
    # 读取图像
    image = cv2.imread(os.path.join("img", image_file))

    # 获取图像尺寸
    height, width = image.shape[:2]

    # 创建 blob，用于进行前向传播
    blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)

    # 设置输入 blob
    net.setInput(blob)

    # 进行前向传播
    layer_names = net.getUnconnectedOutLayersNames()
    outputs = net.forward(layer_names)

    # 初始化列表以保存检测结果
    boxes = []
    confidences = []
    class_ids = []

    # 遍历每个输出层
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    for i in indices.flatten():
        x, y, w, h = boxes[i]

        roi_x = max(0, x - 10)
        roi_y = max(0, y - 10)
        roi_w = min(w + 20, width - roi_x)
        roi_h = min(h + 20, height - roi_y)
        roi = image[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]

        mask = np.zeros(roi.shape[:2], np.uint8)
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)
        rect = (10, 10, roi_w - 20, roi_h - 20)

        cv2.grabCut(roi, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

        kernel = np.ones((5, 5), np.uint8)
        mask2 = cv2.erode(mask2, kernel, iterations=1)

        transparent_roi = np.zeros((roi_h, roi_w, 4), dtype=np.uint8)
        for c in range(3):
            transparent_roi[:, :, c] = roi[:, :, c] * mask2
        transparent_roi[:, :, 3] = mask2 * 255

        cv2.imwrite(f"out/cropped_{image_file}_{i}.png", transparent_roi)

#cv2.imshow("YOLOv3 Object Detection", image)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
