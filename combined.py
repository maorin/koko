import cv2
import numpy as np
import os


source_dir = "laplacian_out"

# 加载 YOLOv3 模型
net = cv2.dnn.readNet("yolo/yolov3.weights", "yolo/yolov3.cfg")

# 获取 COCO 数据集的类别名称
with open("yolo/coco.names", "r") as f:
    classes = f.read().strip().split("\n")

image_files = os.listdir(source_dir)

for image_file in image_files:
    image = cv2.imread(os.path.join("img", image_file))
    height, width = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layer_names = net.getUnconnectedOutLayersNames()
    outputs = net.forward(layer_names)

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

                # 使用 YOLOv3 的边界框作为 GrabCut 的矩形区域
                mask = np.zeros(image.shape[:2], np.uint8)
                bgdModel = np.zeros((1, 65), np.float64)
                fgdModel = np.zeros((1, 65), np.float64)
                rect = (x, y, w, h)
                cv2.grabCut(image, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

                mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
                transparent_roi = np.zeros((height, width, 4), dtype=np.uint8)
                for c in range(3):
                    transparent_roi[y:y+h, x:x+w, c] = image[y:y+h, x:x+w, c] * mask2[y:y+h, x:x+w]
                transparent_roi[y:y+h, x:x+w, 3] = mask2[y:y+h, x:x+w] * 255

                # 保存裁剪结果
                cv2.imwrite(f"combined_out/cropped_{image_file}_{class_id}.png", transparent_roi)
