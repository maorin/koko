import cv2
import numpy as np
import os

# 加载 YOLOv3 模型
net = cv2.dnn.readNet("yolo/yolov3.weights", "yolo/yolov3.cfg")
with open("yolo/coco.names", "r") as f:
    classes = f.read().strip().split("\n")

image_files = os.listdir("img")
some_threshold = 50

confidence_threshold = 0.5

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

            if confidence > confidence_threshold:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                # 使用 Laplacian 边缘检测
                roi = image[y:y+h, x:x+w]
                gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                blurred_roi = cv2.GaussianBlur(gray_roi, (3, 3), 0)
                laplacian_roi = cv2.Laplacian(blurred_roi, cv2.CV_64F)
                laplacian_roi = cv2.convertScaleAbs(laplacian_roi)
                
                contours, _ = cv2.findContours(laplacian_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                mask_roi = np.zeros(gray_roi.shape, dtype=np.uint8)
                for contour in contours:
                    if cv2.contourArea(contour) > some_threshold:
                        cv2.drawContours(mask_roi, [contour], -1, 255, cv2.FILLED)

                # 使用 GrabCut 算法
                bgdModel = np.zeros((1, 65), np.float64)
                fgdModel = np.zeros((1, 65), np.float64)
                rect = (10, 10, w - 20, h - 20)
                cv2.grabCut(roi, mask_roi, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

                mask2 = np.where((mask_roi == 2) | (mask_roi == 0), 0, 1).astype('uint8')
                transparent_roi = np.zeros((h, w, 4), dtype=np.uint8)
                for c in range(3):
                    transparent_roi[:, :, c] = roi[:, :, c] * mask2
                transparent_roi[:, :, 3] = mask2 * 255

                # 保存裁剪结果
                cv2.imwrite(f"yolo_grabcut_laplacian/cropped_{image_file}_{class_id}.png", transparent_roi)
