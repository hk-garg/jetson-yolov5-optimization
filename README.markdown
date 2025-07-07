# NVIDIA Jetson Xavier NX Setup Guide

This guide outlines the steps to set up the NVIDIA Jetson Xavier NX Developer Kit, run a YOLOv5 model, and execute an optimized TensorRT inference file.

## Prerequisites
- NVIDIA Jetson Xavier NX Developer Kit
- MicroSD card (16GB or larger, UHS-I speed class recommended)
- Laptop with internet connection and SD card reader
- Micro USB cable
- HDMI monitor, USB keyboard, and mouse
- Power supply (12V, 2A or higher, barrel connector)
- NVIDIA Developer account
- Conda installed on Jetson for environment management

## Setup Jetson Xavier NX

1. **Download JetPack SD Card Image**
   - Visit [NVIDIA Developer Downloads](https://developer.nvidia.com/downloads).
   - Select *Jetson* > *Jetson Xavier NX Developer Kit* under SD Card Image Method.
   - Log in or register for a free NVIDIA Developer account.
   - Download the latest Jetson Xavier NX SD Card Image (JetPack).

2. **Flash MicroSD Card**
   - Insert the microSD card into your laptop.
   - Format the card using [SD Memory Card Formatter](https://www.sdcard.org/downloads/formatter/).
   - Use [Balena Etcher](https://www.balena.io/etcher/):
     - Select the downloaded JetPack image (.zip file).
     - Choose the microSD card as the target.
     - Click *Flash* to write the image.
   - Alternatively, use command line (Linux/macOS example):
     ```bash
     unzip ~/Downloads/jetson-nx-developer-kit-sd-card-image.zip | sudo dd of=/dev/sdx bs=1M status=progress
     ```
     Replace `/dev/sdx` with your microSD card’s device name.

3. **Set Up Jetson Xavier NX**
   - Insert the flashed microSD card into the slot on the underside of the Jetson Xavier NX module (label facing up, until it clicks).
   - Connect the Jetson to:
     - HDMI monitor via HDMI port.
     - USB keyboard and mouse via USB ports.
     - Power supply via barrel connector (9-16V, not the 19V supply included).
   - Connect the micro USB cable to your laptop for data transfer (not power).

4. **Boot and Initial Configuration**
   - Power on the Jetson (green LED near micro USB port lights up).
   - Follow on-screen prompts to:
     - Accept the EULA.
     - Select language, keyboard, and time zone.
     - Set a username and password (e.g., username: `nvidia`, password: `nvidia`).
   - The Jetson boots to the Ubuntu desktop (18.04 or 20.04, depending on JetPack version).

5. **Install JetPack Components**
   - On your laptop, download and install [NVIDIA SDK Manager](https://developer.nvidia.com/nvidia-sdk-manager).
   - Run SDK Manager:
     ```bash
     sudo dpkg -i sdkmanager_<version>_amd64.deb
     ```
   - Log in with your NVIDIA account.
   - Select *Jetson Xavier NX* as target hardware and the matching JetPack version.
   - Keep the micro USB cable connected for virtual Ethernet (IP: 192.168.55.1 for Jetson, 192.168.55.100 for laptop).
   - Follow prompts to install libraries and drivers (ensure username/password match Jetson setup).
   - Reboot the Jetson after installation.

6. **Verify Setup**
   - Log in to the Ubuntu desktop.
   - Open a terminal and check system status:
     ```bash
     nvcc --version
     ```
     This confirms CUDA installation.
   - Configure WiFi (optional):
     ```bash
     nmcli r wifi on
     nmcli d wifi list
     nmcli d wifi connect <SSID> password <PASSWORD>
     ```

## Running YOLOv5 Model

1. **Set Up Environment**
   - Create a Conda environment to isolate dependencies:
     ```bash
     conda create -n yolov5 python=3.8
     conda activate yolov5
     ```
   - Install PyTorch and other dependencies compatible with Jetson’s CUDA:
     ```bash
     pip install torch torchvision --index-url https://download.pytorch.org/whl/cu113
     pip install opencv-python numpy pyyaml tqdm
     ```
     *Purpose*: Ensures a clean environment with required libraries for YOLOv5.

2. **Clone YOLOv5 Repository**
   - Clone the official YOLOv5 repository from Ultralytics:
     ```bash
     git clone https://github.com/ultralytics/yolov5.git
     cd yolov5
     pip install -r requirements.txt
     ```
     *Purpose*: Downloads YOLOv5 code and installs additional dependencies.

3. **Run Inference**
   - Download a pre-trained YOLOv5 model (e.g., `yolov5s.pt` for small model) from [YOLOv5 releases](https://github.com/ultralytics/yolov5/releases).
   - Run inference on a folder of images (e.g., five images in `data/images/`):
     ```bash
     python detect.py --weights yolov5s.pt --img 640 --conf 0.25 --source data/images/
     ```
     *Purpose*: Performs object detection on images, saving results with bounding boxes in `runs/detect/exp/`.
     *Options*:
       - `--weights`: Choose other models like `yolov5m.pt` (medium), `yolov5l.pt` (large), or `yolov5x.pt` (extra-large) for higher accuracy but slower inference.
       - `--img`: Adjust input image size (e.g., 320 for faster inference, 1280 for higher accuracy).
       - `--conf`: Set confidence threshold (e.g., 0.4 for stricter detections).

4. **Verify Results**
   - Check output images in `runs/detect/exp/` for detected objects (e.g., persons, cars) with bounding boxes.
   - Average inference time: ~42 ms per image on Jetson Xavier NX.
   - *Purpose*: Confirms model detects objects correctly.

## Running Optimized TensorRT Inference

1. **Install TensorRT**
   - Verify TensorRT is installed via JetPack:
     ```bash
     dpkg -l | grep tensorrt
     ```
     *Purpose*: Ensures TensorRT libraries are available for optimized inference.

2. **Set Up TensorRT Environment**
   - In the `yolov5` Conda environment:
     ```bash
     pip install tensorrt pycuda
     ```
     *Purpose*: Installs TensorRT Python bindings and PyCUDA for engine execution.

3. **Convert YOLOv5 to TensorRT**
   - Navigate to the YOLOv5 directory:
     ```bash
     cd yolov5
     ```
   - Export the YOLOv5 model to ONNX format:
     ```bash
     python export.py --weights yolov5s.pt --include onnx
     ```
     *Purpose*: Converts PyTorch model to ONNX for TensorRT compatibility.
   - Convert ONNX to TensorRT engine with FP16 precision:
     ```bash
     trtexec --onnx=yolov5s.onnx --saveEngine=yolov5s.trt --fp16
     ```
     *Purpose*: Builds a TensorRT engine optimized for Jetson Xavier NX GPU.
     *Options*:
       - `--fp16`: Uses 16-bit floating-point precision for faster inference with minimal accuracy loss.
       - `--int8`: Uses 8-bit integer precision for maximum speed but requires calibration data for accuracy (not included here; see [NVIDIA TensorRT Docs](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#int8-calibration)).
       - `--fp32`: Uses 32-bit floating-point precision for highest accuracy but slower inference (default if no flag specified).

4. **Run TensorRT Inference**
   - Run inference using the TensorRT engine:
     ```bash
     python detect.py --weights yolov5s.trt --img 640 --conf 0.25 --source data/images/
     ```
     *Purpose*: Executes optimized inference, saving results in `runs/detect/expX/`.
   - Fix bounding box scaling issues by updating the postprocess function:
     ```python
     # In detect.py, update postprocess function
     scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
     ```
     *Purpose*: Ensures bounding boxes align with original image dimensions.
   - Average inference time: ~30-35 ms per image (faster than PyTorch).

5. **Troubleshooting**
   - If predictions skip large bounding boxes, verify `--img 640` matches input image size.
   - Rebuild the TensorRT engine if errors occur:
     ```bash
     rm yolov5s.trt
     trtexec --onnx=yolov5s.onnx --saveEngine=yolov5s.trt --fp16
     ```
     *Purpose*: Regenerates the engine to resolve compatibility issues.
   - For INT8, prepare a calibration dataset and use `--int8 --calib=<calibration_file>` with `trtexec`.

## Notes
- Use a 12V 2A (or higher, up to 16V) power supply. The included 19V supply may not be compatible.
- For NVMe SSD booting, follow [this guide](https://developer.ridgerun.com/wiki/index.php?title=How_to_flash_and_boot_a_Jetson_from_NVMe_SSD).
- Refer to [NVIDIA Jetson Xavier NX Getting Started](https://developer.nvidia.com/embedded/learn/get-started-jetson-xavier-nx-devkit) for setup details.
- For YOLOv5 and TensorRT issues, consult [YOLOv5 GitHub](https://github.com/ultralytics/yolov5) and [NVIDIA TensorRT Docs](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html).
- Monitor GPU usage with `nvidia-smi` to ensure efficient resource allocation.