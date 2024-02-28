
<h1 align="center"><span>YOLOv9 + ByteTracker</span></h1>

This project integrates YOLOv9 and ByteTracker for real-time, TensorRT-optimized object detection and tracking, extending the existing [TensorRT-Yolov9](https://github.com/spacewalk01/tensorrt-yolov9) implementation.

<p align="center" margin: 0 auto;>
  <img src="assets/demo.gif" width="360px" />
  <img src="assets/highway.gif" width="360px" />
</p>

## Usage


CPP(TensorRT):

``` shell
yolov9-bytetrack-trt.exe yolov9-c.engine test.mp4 # the video path
```

Python(ONNX):

``` shell
python yolov9_bytetrack.py --model yolov9-c-converted.onnx --video_path test_video.mp4 --output_video_path result.mp4
```

## What is next?

- [ ] Python(TensorRT)

## Setup

Refer to our [docs/INSTALL.md](https://github.com/spacewalk01/yolov9-bytetrack-tensorrt/blob/main/docs/INSTALL.md) for detailed installation instructions. Note that Bytetracker is directly integrated without any modification.

## Requirement
   - TensorRT
   - CUDA, CudaNN
   - Eigen 3.3
   - C++ compiler with C++17 or higher support
   - CMake 3.14 or higher
   - OpenCV
     
## Acknowledgement

This project is based on the following awesome projects:
- [YOLOv9](https://github.com/WongKinYiu/yolov9) - YOLOv9: Learning What You Want to Learn Using Programmable Gradient Information.
- [ByteTrack](https://github.com/Vertical-Beach/ByteTrack-cpp) - C++ implementation of ByteTrack algorithm. 
- [TensorRT-Yolov9](https://github.com/spacewalk01/tensorrt-yolov9) - C++ implementation of YOLOv9 using TensorRT API.
