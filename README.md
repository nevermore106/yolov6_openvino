# yolov6_openvino
简介:
YoloV6 OpenVINO 版本, 实现快速的推理与部署。

# requirements:

    Ubuntu 18.04+
    
    OpenVINO 2023
    
    OpenCV2

# deploy:
1. 更改CMakeLists.txt中的OpenVINO安装路径
2.

```bash
cd yolov6_openvino
mkdir build
cd build
cmake ..
make
./openvino_test
```
