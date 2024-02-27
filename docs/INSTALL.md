## Installation

1. Setup [yolov9](https://github.com/WongKinYiu/yolov9) and download [yolov9-c.pt](https://github.com/WongKinYiu/yolov9/releases/download/v0.1/yolov9-c.pt) model.
3. Convert the model to onnx format:

- Copy `general.y` in this repo to `utils/general.py` in yolov9 installation folder
- Copy `export.py` in this repo to yolov9 installation folder
- Then export the model
``` shell
python export.py --weights yolov9-c.pt --include onnx
```
3. Build a TensorRT engine: 

``` shell
trtexec.exe --onnx=yolov9-c.onnx --explicitBatch --saveEngine=yolov9-c.engine --fp16
```

4. Install Eigen referring to [this guide](https://rubengerritsen.nl/docs/02_cmake/01_windows/). Maybe need administrative privileges.

5. Set `opencv` and `tensorrt` installation paths in [CMakeLists.txt](https://github.com/spacewalk01/yolov9-bytetrack-tensorrt/blob/main/CMakeLists.txt):

```
# Find and include OpenCV
set(OpenCV_DIR "your path to OpenCV")
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Set TensorRT path if not set in environment variables
set(TENSORRT_DIR "your path to TensorRT")
```

6. Build:
   
    1. Windows:
    ```bash
     mkdir build
    cd build
    cmake ..
    cmake --build . --config Release
    ```

    2. Linux(not tested):
    ```bash
    mkdir build
    cd build && mkdir out_dir
    cmake ..
    make

