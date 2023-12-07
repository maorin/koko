import cv2
import numpy as np
import os

some_threshold = 50


# 读取img目录下的全部文件
image_files = os.listdir("img")

for image_file in image_files:

    # 读取图像
    #image = cv2.imread(image_file)
    image = cv2.imread(os.path.join("img", image_file))

    # 转换为灰度图像
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 应用高斯模糊以减少噪声
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    # 应用Laplacian边缘检测
    laplacian = cv2.Laplacian(blurred, cv2.CV_64F)

    # 将结果转换为uint8类型
    laplacian = cv2.convertScaleAbs(laplacian)

    '''
    # 显示边缘检测结果
    cv2.imshow('laplacian', laplacian)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    '''

    # 使用找到的边缘进行轮廓检测
    contours, _ = cv2.findContours(laplacian.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 创建一个全黑的掩膜图像
    mask = np.zeros(gray.shape, dtype=np.uint8)

    # 填充所有显著的轮廓
    for contour in contours:
        if cv2.contourArea(contour) > some_threshold:  # 设置一个阈值来过滤掉太小的轮廓
            cv2.drawContours(mask, [contour], -1, color=255, thickness=cv2.FILLED)

    # 使用形态学操作填充内部小洞
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)

    # 膨胀和侵蚀操作
    mask = cv2.dilate(mask, kernel, iterations=1)  # 膨胀
    mask = cv2.erode(mask, kernel, iterations=1)   # 侵蚀

    # 创建一个新的4通道图像来放置结果
    result = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)

    # 将原始图像的颜色复制到结果图像
    result[:, :, 0:3] = image

    # 将掩膜复制到alpha通道，创建透明度
    result[:, :, 3] = mask

    # 保存结果图像
    cv2.imwrite('laplacian_out/%s' % image_file, result)

# 显示结果
#cv2.imshow('Cropped Image', result)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
