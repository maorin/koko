import cv2
import numpy as np

# 加载 YOLOv3 模型
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

# 获取 COCO 数据集的类别名称
with open("coco.names", "r") as f:
    classes = f.read().strip().split("\n")

# 读取图像
image = cv2.imread("test.png")

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

        if confidence > 0.5:  # 设置置信度阈值
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)

            # 计算边界框的坐标
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

# 应用非最大抑制以去除重叠的边界框
indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

# 在图像上绘制检测结果
if len(indices) > 0:
    for i in indices.flatten():
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        confidence = confidences[i]

        color = (0, 255, 0)  # 框的颜色，可以自行调整
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        cv2.putText(image, f"{label} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # 裁剪矩形框外部的部分
        image = image[max(0, y):min(y + h, height), max(0, x):min(x + w, width)]

# 显示检测结果
cv2.imshow("YOLOv3 Object Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
