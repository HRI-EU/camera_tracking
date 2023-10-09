# camera_tracking

[![pipeline status](https://dmz-gitlab.honda-ri.de/robotics/camera_tracking/badges/noetic/pipeline.svg)](https://dmz-gitlab.honda-ri.de/robotics/camera_tracking/-/commits/noetic)
[![coverage](https://dmz-gitlab.honda-ri.de/robotics/camera_tracking/badges/noetic/coverage.svg)](https://dmz-gitlab.honda-ri.de/robotics/camera_tracking/-/commits/noetic)

# Basic Setup

## To use standalone
You can use the package standalone on your host by executing:
```bash
# Setup a virtual python environment.
./create_venv.sh 

# Source the virtual python environment and set some paths. 
source local_sit.env

./scripts/azure_tracking_socket.py --standalone --visualize --aruco --body
```

## Using camera_tracking without intallation.
You need to add the location of the camera_tracking python package to the PYTHONPATH.
```bash
PROJECT_PATH=root_of_this_repo
export PYTHONPATH=${PYTHONPATH:+${PYTHONPATH}:}${PROJECT_PATH}/src
```

## Using a non-standard installation of the Azure SDK.
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
Then you need point AZURE_CUSTOM_PATH to that directory:
```bash
export AZURE_CUSTOM_PATH=/hri/sit/LTS/External/AzureKinect/1.4.1

# You also need to add the libs to the LD_LIBRARY_PATH.
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:+${LD_LIBRARY_PATH}:}${AZURE_CUSTOM_PATH}/lib
```
