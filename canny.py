import cv2
import numpy as np
import os

# 读取img目录下的全部文件
image_files = os.listdir("img")

for image_file in image_files:

    # 读取图像
    #image = cv2.imread(image_file)
    image = cv2.imread(os.path.join("img", image_file))

    # 转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 应用Canny边缘检测
    edges = cv2.Canny(gray, 50, 150)

    # 查找轮廓
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 创建一个与原图大小一致的全黑图像
    mask = np.zeros_like(image)

    # 在全黑图像上绘制白色轮廓
    cv2.drawContours(mask, contours, -1, (255, 255, 255), thickness=cv2.FILLED)

    # 创建RGBA图像
    rgba = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)

    # 使用轮廓作为掩膜提取前景
    rgba[:, :, 3] = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    # 保存结果图像
    cv2.imwrite('canny_out/%s' % image_file, rgba)

# 显示结果
#cv2.imshow('Cropped Image', result)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
