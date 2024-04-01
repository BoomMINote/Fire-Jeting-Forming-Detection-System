import cv2
import numpy as np
import time 

bias = 450 # edges search range, center-bias -> center+bias
tube_width = 19.10
text_bias = 350
space = 50

cap = cv2.VideoCapture(3)  # 调用Hikon camera
if False == cap.isOpened():
    print("Hikon camera is not opened!")
else:
    print("Hikon camera ok!")
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2448)  # 设置图像宽度
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2048)  # 设置图像高度
cap.set(cv2.CAP_PROP_FPS , 60)   # 设置帧率
def canny_edge_detection(frame, bias = 600, tube_width = 19.10): 
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Convert the frame to grayscale for edge detection 
    blurred = cv2.GaussianBlur(src=gray, ksize=(5, 5), sigmaX=0.9) # 3,5,0.5
    edges = cv2.Canny(blurred, 70, 135) # Perform Canny edge detection  # 2048,2448
    bias = bias
    # 0,1,2,3,4, 5 ,6,7,8,9,X
    #     0,1,2, 3 ,4,5,6
    l = edges.shape[1]//2-bias 
    r = edges.shape[1]//2+bias
    newedges = edges[:,l+10:r-10]
    tube_width = tube_width
    dist = []
    abs_dist = []
    left  = [] # left  coordinate
    right = [] # right coordinate
    for row in newedges:
        left_index = np.argmax(row != 0) # 寻找第一个非零值（即边缘）的索引
        left.append(left_index+l+10)
        if np.any(row):
            right_index = len(row) - np.argmax(row[::-1] != 0) - 1
            right.append(right_index+l+10)
        else:
            right_index = 0
            right.append(right_index+l+10)
        dist.append(abs(right_index - left_index)+1)
    # np.savetxt('differences.txt', dist, fmt='%d')
    for i in range(len(dist)):
        if dist[i] == 1:
            abs_dist.append(0)
        else:
            abs_dist.append(dist[i] / max(dist) * tube_width)
    # np.savetxt('abs_differences.txt', abs_dist, fmt='%.2f')
    # np.savetxt('left.txt', left, fmt='%d')
    # np.savetxt('right.txt', right, fmt='%d')
    return blurred, edges, abs_dist, left, right
prev_frame_time = 0
new_frame_time = 0
while True:
    new_frame_time = time.time() 
    ret, frame = cap.read()
    blurred, edges, abs_dist, left, right = canny_edge_detection(frame, bias, tube_width) 
    # edges = edges.expand for better visualization
    kernel = np.ones((7, 7), dtype=np.uint8)
    edges = cv2.dilate(edges, kernel, 3)
    # edges = cv2.resize(edges , (918 , 768))
    # frame = cv2.resize(frame , (918 , 768))
    
    fps = 1/(new_frame_time-prev_frame_time) 
    prev_frame_time = new_frame_time 
    fps = str(int(fps))
    font = cv2.FONT_HERSHEY_SIMPLEX 
    
    # print(edges.shape, frame.shape)
    # edges = edges.reshape((edges.shape[0], edges.shape[1], 1))
    # print(edges.shape, frame.shape)
    # edges = np.concatenate([edges, frame[:,:,1], edges], axis=2)
    # frame[:,:,0] = edges 
    frame[:,:,2] = edges # 0 or 2
    edges = frame
    cv2.putText(edges, fps, (14, 150), font, 5, (100, 200, 150), 6, cv2.LINE_AA) 
    
    text_bias = text_bias
    l = edges.shape[1]//2-text_bias 
    r = edges.shape[1]//2+text_bias
    l = int(l); r = int(r)
    space = space
    for i in range(0+space, edges.shape[0], space):
        cv2.line(edges, (l, i), (r, i), (255, 255, 255), 2) # 在图片上绘制横线
        # print(left[i],right[i])
        cv2.arrowedLine(edges, (left[i],i),   (right[i], i), (25,190,250), 5, cv2.LINE_AA, 0, 0.03)
        cv2.arrowedLine(edges, (right[i], i), (left[i],i),   (25,190,250), 5, cv2.LINE_AA, 0, 0.03)
        cv2.putText(edges, str(f"{abs_dist[i]:.2f}")+"mm", (r+14, i+14), font, 1.3, (255, 255, 255), 3, cv2.LINE_AA) # 在横线上显示数字
    edges = cv2.resize(edges , (918 , 768))
    frame = cv2.resize(frame , (918 , 768))
    # cv2.putText(frame, fps, (7, 70), font, 3, (100, 255, 0), 3, cv2.LINE_AA) 
    # cv2.imshow("Original", frame) 
    # cv2.imshow("Blurred", blurred) 
    cv2.imshow("Fire Jeting Forming Detection System", edges) 
    input = cv2.waitKey(1)
    if input == ord('q'):
        break
 
cap.release()  # 释放摄像头
cv2.destroyAllWindows()  # 销毁窗口