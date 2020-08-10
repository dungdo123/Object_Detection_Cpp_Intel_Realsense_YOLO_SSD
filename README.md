# Object_Detection_C++_Intel_Realsense_camera_YOLO_SSD

Object detection on SR305 (Intel Realsense depth camera) using pre-trained Yolov3, SSD and Opencv

In this project, I implemented object detectors on a depth camera using some popular pre-trained model such as YoloV3, SSD. The code is written in C++ because the Intel realsense series are supported strongly by this language. Beside object detection task, The camera also give us the depth information of the object with quite high accuracy (SR305 series uses light code techniques to get the depth). The results show that SSD detector run much more faster than YoloV3 with my available hardwares, the avarage inference time are ~0.3s and ~0.7s per frame respectively. 

Requirements and setup:
- Hardwares: Intel SR305 camera (400 series are possible), Dell core i5, no GPU
- Software: VS2019 environment, OpenCV 4.2, librealsense SDK
- Download pre-trained YoloV3, SSD models: yolov3.weights, yolov3.cfg, coco.names, MobileNetSSD_deploy.caffemodel, MobileNetSSD_deploy.prototxt (just google for these files, there are many available sources)

References:
1. https://github.com/krutikabapat/DNN-Object-Detection-YOLOv3
2. https://github.com/IntelRealSense/librealsense/tree/master/examples
