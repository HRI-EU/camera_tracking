# camera_tracking

[![pipeline status](https://dmz-gitlab.honda-ri.de/robotics/camera_tracking/badges/noetic/pipeline.svg)](https://dmz-gitlab.honda-ri.de/robotics/camera_tracking/-/commits/noetic)
[![coverage](https://dmz-gitlab.honda-ri.de/robotics/camera_tracking/badges/noetic/coverage.svg)](https://dmz-gitlab.honda-ri.de/robotics/camera_tracking/-/commits/noetic)

This project offers camera-based marker tracking.

# Usage

## To use standalone
You can use the package standalone on your host by executing:
```bash
# Setup a virtual python environment.
./create_venv.sh 

# Source the virtual python environment and set some paths. 
source local_sit.env

# E.g., start tracking using the Azure camera with socket interface. 
./scripts/azure_tracking_socket.py --standalone --visualize --aruco --body
```

## Using camera_tracking without installation
You need to add the location of the camera_tracking python package to the PYTHONPATH.
```bash
PROJECT_PATH=root_of_this_project
export PYTHONPATH=${PYTHONPATH:+${PYTHONPATH}:}${PROJECT_PATH}/src
```

## Using a non-standard installation of the Azure SDK
Normally all dependencies of the Azure camera are installed in several system locations requiring root access.
We provide a way to use a custom installation. For this you need to provide the dependencies in a structure like:
```bash
blue-17:/>tree /hri/sit/LTS/External/AzureKinect/1.4.1
/hri/sit/LTS/External/AzureKinect/1.4.1
├── bin
│   ├── dnn_model_2_0_lite_op11.onnx
│   └── dnn_model_2_0_op11.onnx
└── lib
    ├── libcublasLt.so.11
    ├── libcublas.so.11
    ├── libcudart.so.11.0
    ├── libcudnn_cnn_infer.so.8
    ├── libcudnn_ops_infer.so.8
    ├── libcudnn.so.8
    ├── libdepthengine.so.2.0
    ├── libk4abt.so
    ├── libk4abt.so.1.1
    ├── libk4abt.so.1.1.2
    ├── libk4arecord.so
    ├── libk4arecord.so.1.4
    ├── libk4arecord.so.1.4.1
    ├── libk4a.so
    ├── libk4a.so.1.4
    ├── libk4a.so.1.4.1
    ├── libonnxruntime_providers_cuda.so
    ├── libonnxruntime_providers_shared.so
    ├── libonnxruntime_providers_tensorrt.so
    └── libonnxruntime.so.1.10.0
```
Then you need to point AZURE_CUSTOM_PATH to that directory and also customize the LD_LIBRARY_PATH:
```bash
export AZURE_CUSTOM_PATH=/hri/sit/LTS/External/AzureKinect/1.4.1
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:+${LD_LIBRARY_PATH}:}${AZURE_CUSTOM_PATH}/lib
```

## Azure requires udev rules
To connect the Azure via USB under linux some udev rules are required:
```bash
blue-17:/>cat /etc/udev/rules.d/55-kinect-usb.rules
# Bus 002 Device 116: ID 045e:097a Microsoft Corp.  - Generic Superspeed USB Hub
# Bus 001 Device 015: ID 045e:097b Microsoft Corp.  - Generic USB Hub
# Bus 002 Device 118: ID 045e:097c Microsoft Corp.  - Azure Kinect Depth Camera
# Bus 002 Device 117: ID 045e:097d Microsoft Corp.  - Azure Kinect 4K Camera
# Bus 001 Device 016: ID 045e:097e Microsoft Corp.  - Azure Kinect Microphone Array

BUS!="usb", ACTION!="add", SUBSYSTEM!=="usb_device", GOTO="k4a_logic_rules_end"

ATTRS{idVendor}=="045e", ATTRS{idProduct}=="097a", MODE="0666", GROUP="plugdev"
ATTRS{idVendor}=="045e", ATTRS{idProduct}=="097b", MODE="0666", GROUP="plugdev"
ATTRS{idVendor}=="045e", ATTRS{idProduct}=="097c", MODE="0666", GROUP="plugdev"
ATTRS{idVendor}=="045e", ATTRS{idProduct}=="097d", MODE="0666", GROUP="plugdev"
ATTRS{idVendor}=="045e", ATTRS{idProduct}=="097e", MODE="0666", GROUP="plugdev"

LABEL="k4a_logic_rules_end"
```

# Issues 

- When starting the azur tracking sometimes it crashes. Usually a restart of the program works.
```bash
(venv) blue-17:~/camera_tracking/scripts(noetic)>./azure_tracking_socket.py --aruco --body --visualize --standalone --color-resolution 1536P
Using dnn model path '/hri/sit/LTS/External/AzureKinect/1.4.1/bin/dnn_model_2_0_op11.onnx'.
Running stand-alone.
Step 0 mean times: overall 0.042s | capture 0.016s | processing: 0.026s | body 0.026s | aruco 0.001s
[2023-10-09 12:52:44.137] [error] [t=1101202] /__w/1/s/extern/Azure-Kinect-Sensor-SDK/src/image/image.c (51): k4a_image_t_get_context(). Invalid k4a_image_t (nil)
[2023-10-09 12:52:44.137] [error] [t=1101202] /__w/1/s/extern/Azure-Kinect-Sensor-SDK/src/image/image.c (389): Invalid argument to image_get_buffer(). image_handle ((nil)) is not a valid handle of type k4a_image_t
[2023-10-09 12:52:44.137] [error] [t=1101202] [K4ABT] ../src/TrackerHost/DepthFrameBlobK4A.cpp (13): Initialize(). Get depth buffer from the capture handle failed!
[2023-10-09 12:52:44.137] [error] [t=1101202] [K4ABT] ../src/TrackerHost/TrackerHost.cpp (274): EnqueueCapture(). Initialize DepthFrameBlob failed!
Body tracker capture enqueue failed!
  File "/usr/lib/python3.8/threading.py", line 890, in _bootstrap
    self._bootstrap_inner()
  File "/usr/lib/python3.8/threading.py", line 932, in _bootstrap_inner
    self.run()
  File "/usr/lib/python3.8/threading.py", line 870, in run
    self._target(*self._args, **self._kwargs)
  File "/hri/localdisk/stephanh/camera_tracking/src/camera_tracking/base_tracking.py", line 64, in worker
    landmarks = self.tracker.process(data)
  File "/hri/localdisk/stephanh/camera_tracking/src/camera_tracking/azure_tracking.py", line 73, in process
    body_frame = self.body_tracker.update()
  File "/hri/localdisk/stephanh/camera_tracking/venv/lib/python3.8/site-packages/pykinect_azure/k4abt/tracker.py", line 39, in update
    self.enqueue_capture(Device.capture.handle(), timeout_in_ms)
  File "/hri/localdisk/stephanh/camera_tracking/venv/lib/python3.8/site-packages/pykinect_azure/k4abt/tracker.py", line 44, in enqueue_capture
    _k4abt.VERIFY(_k4abt.k4abt_tracker_enqueue_capture(self._handle, capture_handle, timeout_in_ms), "Body tracker capture enqueue failed!")
  File "/hri/localdisk/stephanh/camera_tracking/venv/lib/python3.8/site-packages/pykinect_azure/k4abt/_k4abt.py", line 184, in VERIFY
    traceback.print_stack()
```

- When starting the azure tracking it cannot start the depth engine. This happens when you start it remotely. You need to be visually logged into the remote host.
```bash
(venv) remote-host:~/camera_tracking(noetic)>./scripts/azure_tracking_socket.py --body 
Error: OpenGL 4.4 context creation failed. You could try updating your graphics drivers.
[2023-10-09 13:34:16.184] [error] [t=1163249] /__w/1/s/extern/Azure-Kinect-Sensor-SDK/src/dewrapper/dewrapper.c (154): depth_engine_start_helper(). Depth engine create and initialize failed with error code: 207.
[2023-10-09 13:34:16.184] [error] [t=1163249] /__w/1/s/extern/Azure-Kinect-Sensor-SDK/src/dewrapper/dewrapper.c (157): depth_engine_start_helper(). OpenGL 4.4 context creation failed. You could try updating your graphics drivers.
[2023-10-09 13:34:16.184] [error] [t=1163249] /__w/1/s/extern/Azure-Kinect-Sensor-SDK/src/dewrapper/dewrapper.c (160): deresult == K4A_DEPTH_ENGINE_RESULT_SUCCEEDED returned failure in depth_engine_start_helper()
[2023-10-09 13:34:16.184] [error] [t=1163249] /__w/1/s/extern/Azure-Kinect-Sensor-SDK/src/dewrapper/dewrapper.c (194): depth_engine_start_helper(dewrapper, dewrapper->fps, dewrapper->depth_mode, &depth_engine_max_compute_time_ms, &depth_engine_output_buffer_size) returned failure in depth_engine_thread()
[2023-10-09 13:34:16.184] [error] [t=1163232] /__w/1/s/extern/Azure-Kinect-Sensor-SDK/src/dewrapper/dewrapper.c (552): dewrapper_start(). Depth Engine thread failed to start
[2023-10-09 13:34:16.184] [error] [t=1163232] /__w/1/s/extern/Azure-Kinect-Sensor-SDK/src/depth/depth.c (398): dewrapper_start(depth->dewrapper, config, depth->calibration_memory, depth->calibration_memory_size) returned failure in depth_start()
[2023-10-09 13:34:16.184] [error] [t=1163232] /__w/1/s/extern/Azure-Kinect-Sensor-SDK/src/depth_mcu/depth_mcu.c (359): cmd_status == CMD_STATUS_PASS returned failure in depthmcu_depth_stop_streaming()
[2023-10-09 13:34:16.184] [error] [t=1163232] /__w/1/s/extern/Azure-Kinect-Sensor-SDK/src/depth_mcu/depth_mcu.c (362): depthmcu_depth_stop_streaming(). ERROR: cmd_status=0x00000063
[2023-10-09 13:34:16.184] [error] [t=1163232] /__w/1/s/extern/Azure-Kinect-Sensor-SDK/src/sdk/k4a.c (895): depth_start(device->depth, config) returned failure in k4a_device_start_cameras()
Start K4A cameras failed!
```
