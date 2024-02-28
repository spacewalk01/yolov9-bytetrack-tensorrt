
<h1 align="center"><span>YOLOv9 + ByteTracker (Python)</span></h1>

This version integrates YOLOv9 and ByteTracker using ONNX Runtime inference in Python.

## Usage

``` shell
python yolov9_bytetrack.py --model yolov9-c-converted.onnx --video_path test_video.mp4 --output_video_path result.mp4
```
 
## Setup

Follow instruction step 1~4 in  [docs/INSTALL.md](https://github.com/spacewalk01/yolov9-bytetrack-tensorrt/blob/main/docs/INSTALL.md). 

## Requirement
   - cython_bbox
     
## Acknowledgement

This project is based on the following awesome projects:
- [YOLOv9](https://github.com/WongKinYiu/yolov9) - YOLOv9: Learning What You Want to Learn Using Programmable Gradient Information.
- [ByteTrack](https://github.com/ifzhang/ByteTrack) - Original python implementation of ByteTrack algorithm. 
