import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import cv2
import os
import time

# Logger for TensorRT
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

# YOLOv5 parameters
CONF_THRESH = 0.1  # Confidence threshold
IOU_THRESH = 0.4   # IoU threshold for NMS
CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
    "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush"
]
NUM_CLASSES = len(CLASSES)

def load_engine(engine_path):
    """Load TensorRT engine from file."""
    with open(engine_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

def preprocess_batch(image_paths, input_shape=(1, 3, 640, 640)):
    """Preprocess a batch of images into a single tensor."""
    batch_images = []
    original_shapes = []
    img_resized_list = []

    for img_path in image_paths:
        if not os.path.exists(img_path):
            print(f"Warning: Image {img_path} not found, skipping.")
            continue
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Failed to load image: {img_path}, skipping.")
            continue
        original_shapes.append(img.shape[:2])  # (height, width)
        img_resized = cv2.resize(img, (input_shape[2], input_shape[3]))
        img = img_resized.transpose(2, 0, 1).astype(np.float32) / 255.0  # HWC to CHW, normalize
        batch_images.append(img)
        img_resized_list.append(img_resized)

    if len(batch_images) == 0:
        raise ValueError("No images were successfully preprocessed.")
    # Stack images into a batch with actual batch size
    batch_tensor = np.stack(batch_images, axis=0)
    return batch_tensor, original_shapes, img_resized_list

def allocate_buffers(engine):
    """Allocate input/output buffers for TensorRT inference."""
    binding_names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
    input_binding = binding_names[0]
    output_binding = binding_names[1]

    input_shape = engine.get_tensor_shape(input_binding)
    output_shape = engine.get_tensor_shape(output_binding)
    print(f"Input binding: {input_binding}, Shape: {input_shape}")
    print(f"Output binding: {output_binding}, Shape: {output_shape}")

    # Adjust batch size dynamically based on preprocessed batch, but cap at engine's max
    max_batch_size = 1  # Default to 1 until dynamic engine is confirmed
    input_shape = list(input_shape)
    input_shape[0] = max_batch_size
    input_shape = tuple(input_shape)

    # Manual dtype mapping to avoid np.bool deprecation
    dtype_map = {
        trt.float32: np.float32,
        trt.float16: np.float16,
        trt.int8: np.int8,
        trt.int32: np.int32,
        trt.bool: np.bool_
    }
    input_dtype = engine.get_tensor_dtype(input_binding)
    output_dtype = engine.get_tensor_dtype(output_binding)
    input_size = trt.volume(input_shape) * dtype_map[input_dtype]().itemsize
    output_size = trt.volume(output_shape) * dtype_map[output_dtype]().itemsize

    d_input = cuda.mem_alloc(int(input_size))
    d_output = cuda.mem_alloc(int(output_size))
    
    bindings = [int(d_input), int(d_output)]
    return d_input, d_output, bindings, input_shape, output_shape, input_binding, output_binding

def do_inference(context, bindings, d_input, d_output, stream, input_data, output_shape, input_binding):
    """Perform inference using TensorRT."""
    cuda.memcpy_htod_async(d_input, input_data, stream)
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    output = np.empty(output_shape, dtype=np.float32)
    cuda.memcpy_dtoh_async(output, d_output, stream)
    stream.synchronize()
    print(f"Inference output shape: {output.shape}")
    return output

def postprocess(output, original_shapes, input_shape, conf_thres=CONF_THRESH, iou_thres=IOU_THRESH):
    """Post-process YOLOv5 output for each image in the batch."""
    results = []
    input_h = input_shape[-2]  # Height
    input_w = input_shape[-1]  # Width
    batch_size = output.shape[0]

    for i in range(batch_size):
        if i >= len(original_shapes):  # Handle cases where fewer images were processed
            results.append(([], [], []))
            continue
        boxes = output[i, ..., :4]  # x, y, w, h for current image
        obj_conf = output[i, ..., 4]  # Objectness score
        cls_conf = output[i, ..., 5:]  # Class probabilities

        scores = obj_conf[..., None] * cls_conf  # (num_boxes, num_classes)
        max_scores = np.max(scores, axis=-1)  # Max score per box
        max_classes = np.argmax(scores, axis=-1)  # Class ID with max score

        mask = max_scores > conf_thres
        boxes = boxes[mask]
        scores = max_scores[mask]
        class_ids = max_classes[mask]

        if len(boxes) == 0:
            results.append(([], [], []))
            continue

        # Scaling to original image dimensions with bounds checking
        h, w = original_shapes[i]
        scale_x = w / input_w
        scale_y = h / input_h
        boxes[:, [0, 2]] *= scale_x  # Scale x_center and width
        boxes[:, [1, 3]] *= scale_y  # Scale y_center and height

        # Convert to (x1, y1, x2, y2) with adjusted scaling to prevent overflow
        boxes_xyxy = np.zeros_like(boxes)
        boxes_xyxy[:, 0] = np.clip(boxes[:, 0] - boxes[:, 2] / 2, 0, w - 1)  # x1
        boxes_xyxy[:, 1] = np.clip(boxes[:, 1] - boxes[:, 3] / 2, 0, h - 1)  # y1
        boxes_xyxy[:, 2] = np.clip(boxes[:, 0] + boxes[:, 2] / 2, 0, w - 1)  # x2
        boxes_xyxy[:, 3] = np.clip(boxes[:, 1] + boxes[:, 3] / 2, 0, h - 1)  # y2

        # Apply NMS
        indices = cv2.dnn.NMSBoxes(
            boxes_xyxy.tolist(),
            scores.tolist(),
            conf_thres,
            iou_thres
        )
        if len(indices) > 0:
            indices = indices if isinstance(indices, np.ndarray) else np.array(indices)
            boxes = boxes_xyxy[indices]
            scores = scores[indices]
            class_ids = class_ids[indices]
        else:
            boxes = np.array([])
            scores = np.array([])
            class_ids = np.array([])

        results.append((boxes, scores, class_ids))

    return results
def draw_boxes(image, boxes, scores, class_ids):
    """Draw bounding boxes, class labels, and confidence scores on the image."""
    print(f"Drawing on image of shape: {image.shape}, boxes: {len(boxes)}")
    for box, score, class_id in zip(boxes, scores, class_ids):
        x1, y1, x2, y2 = map(int, box)
        print(f"Drawing box: ({x1}, {y1}, {x2}, {y2}), score: {score}, class: {CLASSES[class_id]}")
        if x1 < 0 or y1 < 0 or x2 > image.shape[1] or y2 > image.shape[0]:
            print(f"Skipping invalid box: ({x1}, {y1}, {x2}, {y2})")
            continue
        label = f"{CLASSES[class_id]}: {score:.2f}"
        color = (0, 255, 0)  # Green for bounding boxes
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return image

def main(engine_path, image_dir, output_dir="output_images"):
    """Main function to run YOLOv5 inference on a batch of 5 images."""
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # List of image paths (assuming sequential naming, e.g., image1.jpg to image5.jpg)
    image_paths = [os.path.join(image_dir, f"image{i}.jpg") for i in range(1, 6)]
    if not any(os.path.exists(path) for path in image_paths):
        raise ValueError("No images found. Ensure at least one of image1.jpg to image5.jpg exists in the directory.")

    engine = load_engine(engine_path)
    context = engine.create_execution_context()
    
    d_input, d_output, bindings, input_shape, output_shape, input_binding, output_binding = allocate_buffers(engine)
    
    # Preprocess all images
    try:
        input_data, original_shapes, img_resized_list = preprocess_batch(image_paths, input_shape)
    except ValueError as e:
        print(f"Error preprocessing images: {e}")
        return

    # Process images sequentially since engine supports only batch size 1
    for i in range(len(image_paths)):
        if i >= len(img_resized_list):
            print(f"Warning: No data for image {i+1}, skipping.")
            continue
        
        # Set input shape for single image
        input_tensor_name = engine.get_tensor_name(0)
        context.set_input_shape(input_tensor_name, (1, 3, 640, 640))
        output_shape = list(output_shape)
        output_shape[0] = 1
        output_shape = tuple(output_shape)

        stream = cuda.Stream()
        
        start_time = time.time()
        output = do_inference(context, bindings, d_input, d_output, stream, input_data[i:i+1].ravel(), output_shape, input_binding)
        inference_time = (time.time() - start_time) * 1000
        
        # Post-process single image result
        results = postprocess(output, [original_shapes[i]], input_shape)
        
        # Draw and save results
        if results and len(results[0][0]) > 0:  # Check if there are valid boxes
            boxes, scores, class_ids = results[0]
    
    
    
            output_image = draw_boxes(img_resized_list[i], boxes, scores, class_ids)
            output_path = os.path.join(output_dir, f"output_image{i+1}.jpg")
            cv2.imwrite(output_path, output_image)
            print(f"Output image saved to: {output_path}")
            print(f"Image {i+1} - Inference time: {inference_time:.2f} ms")
            print(f"Image {i+1} - Detected {len(boxes)} objects:")
            for j, (box, score, class_id) in enumerate(zip(boxes, scores, class_ids)):
                print(f"  {j+1}. {CLASSES[class_id]} (Confidence: {score:.2f}) at {box.astype(int)}")
        else:
            print(f"Warning: No valid results for image {i+1}, skipping save.")

    stream.synchronize()
    d_input.free()
    d_output.free()

if __name__ == "__main__":
    engine_path = "yolov5.trt"
    image_dir = "./images"  # Directory containing image1.jpg to image5.jpg
    main(engine_path, image_dir)
