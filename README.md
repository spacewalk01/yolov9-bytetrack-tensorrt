
<h1 align="center"><span>YOLOv9 TensorRT C++</span></h1>

This repository provides a C++ implementation of the state-of-the-art [YOLOv9](https://github.com/WongKinYiu/yolov9) object detection model, optimized with TensorRT for real-time inference and combined with the [ByteTracker](https://github.com/Vertical-Beach/ByteTrack-cpp) object tracker.

<p align="center" margin: 0 auto;>
  <img src="assets/demo.gif" width="640px" />
</p>

## 🚀 Usage

``` shell
yolov9-bytetrack-trt.exe yolov9-c.engine test.mp4 # the video path
```

## 🛠️ Setup

We refer to our [docs/INSTALL.md](https://github.com/spacewalk01/tensorrt-yolov9/blob/main/docs/INSTALL.md) for detailed installation instructions.

Note that Bytetracker is directly used without any modification.

## 👏 Acknowledgement

This project is based on the following awesome projects:
- [Yolov9](https://github.com/WongKinYiu/yolov9) - YOLOv9: Learning What You Want to Learn Using Programmable Gradient Information.
- [ByteTrack](https://github.com/Vertical-Beach/ByteTrack-cpp) - C++ implementation of ByteTrack algorithm.
- [TensorRT-Yolov9](https://github.com/spacewalk01/tensorrt-yolov9) - C++ implementation of YOLOv9 using TensorRT API.
