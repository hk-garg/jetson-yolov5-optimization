import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import cv2
import time

# Logger for TensorRT
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

# YOLOv5 parameters
CONF_THRESH = 0.25  # Match PyTorch's confidence threshold
IOU_THRESH = 0.45   # Match PyTorch's IOU threshold
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

def preprocess_image(image_path, input_shape=(1, 3, 384, 640)):  # Changed to 384x640
    """Preprocess input image to match model input requirements."""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")
    original_shape = img.shape[:2]  # (height, width)
    img_resized = cv2.resize(img, (input_shape[3], input_shape[2]))  # Width x Height
    img = img_resized.transpose(2, 0, 1).astype(np.float32) / 255.0  # HWC to CHW, normalize
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img, original_shape, img_resized

def allocate_buffers(engine):
    """Allocate input/output buffers for TensorRT inference."""
    binding_names = [engine.get_tensor_name(i) for i in range(engine.num_bindings)]
    input_binding = binding_names[0]
    output_binding = binding_names[1]

    input_shape = engine.get_tensor_shape(input_binding)
    output_shape = engine.get_tensor_shape(output_binding)
    print(f"Input binding: {input_binding}, Shape: {input_shape}")
    print(f"Output binding: {output_binding}, Shape: {output_shape}")

    input_size = trt.volume(input_shape) * engine.get_tensor_dtype(input_binding).itemsize
    output_size = trt.volume(output_shape) * engine.get_tensor_dtype(output_binding).itemsize

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

def postprocess(output, original_shape, input_shape, conf_thres=CONF_THRESH, iou_thres=IOU_THRESH):
    """Post-process YOLOv5 output to extract bounding boxes, scores, and classes."""
    print(f"Postprocess input shape: {output.shape}")
    boxes = output[0, ..., :4]  # x, y, w, h
    obj_conf = output[0, ..., 4]  # Objectness score
    cls_conf = output[0, ..., 5:]  # Class probabilities

    scores = obj_conf[..., None] * cls_conf  # (num_boxes, num_classes)
    max_scores = np.max(scores, axis=-1)  # Max score per box
    max_classes = np.argmax(scores, axis=-1)  # Class ID with max score

    mask = max_scores > conf_thres
    boxes = boxes[mask]
    scores = max_scores[mask]
    class_ids = max_classes[mask]

    if len(boxes) == 0:
        return [], [], []

    # Extract height and width from input_shape dynamically
    input_dims = input_shape
    input_h = input_dims[-2]  # Height
    input_w = input_dims[-1]  # Width
    h, w = original_shape

    # Precise scaling with aspect ratio preservation
    scale_x = w / input_w
    scale_y = h / input_h
    boxes[:, [0, 2]] *= scale_x  # Scale x_center and width
    boxes[:, [1, 3]] *= scale_y  # Scale y_center and height

    # Convert to (x1, y1, x2, y2) with bounds checking
    boxes_xyxy = np.zeros_like(boxes)
    boxes_xyxy[:, 0] = np.clip(boxes[:, 0] - boxes[:, 2] / 2, 0, w)  # x1
    boxes_xyxy[:, 1] = np.clip(boxes[:, 1] - boxes[:, 3] / 2, 0, h)  # y1
    boxes_xyxy[:, 2] = np.clip(boxes[:, 0] + boxes[:, 2] / 2, 0, w)  # x2
    boxes_xyxy[:, 3] = np.clip(boxes[:, 1] + boxes[:, 3] / 2, 0, h)  # y2

    # Debug: Print raw boxes, scores, and class IDs before NMS
    print(f"Raw boxes before NMS: {boxes_xyxy}")
    print(f"Raw scores before NMS: {scores}")
    print(f"Raw class IDs before NMS: {class_ids}")

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

    return boxes, scores, class_ids

def draw_boxes(image, boxes, scores, class_ids):
    """Draw bounding boxes, class labels, and confidence scores on the image."""
    for box, score, class_id in zip(boxes, scores, class_ids):
        x1, y1, x2, y2 = map(int, box)
        label = f"{CLASSES[class_id]}: {score:.2f}"
        color = (0, 255, 0)  # Green for bounding boxes
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return image

def main(engine_path, image_path, output_image_path="output.jpg"):
    """Main function to run YOLOv5 inference and save the output image."""
    engine = load_engine(engine_path)
    context = engine.create_execution_context()
    
    d_input, d_output, bindings, input_shape, output_shape, input_binding, output_binding = allocate_buffers(engine)
    
    input_data, original_shape, img_resized = preprocess_image(image_path, input_shape)
    
    stream = cuda.Stream()
    
    start_time = time.time()
    output = do_inference(context, bindings, d_input, d_output, stream, input_data.ravel(), output_shape, input_binding)
    inference_time = (time.time() - start_time) * 1000
    
    boxes, scores, class_ids = postprocess(output, original_shape, input_shape)
    
    output_image = draw_boxes(img_resized, boxes, scores, class_ids)
    
    cv2.imwrite(output_image_path, output_image)
    print(f"Output image saved to: {output_image_path}")
    print(f"Inference time: {inference_time:.2f} ms")
    print(f"Detected {len(boxes)} objects:")
    for i, (box, score, class_id) in enumerate(zip(boxes, scores, class_ids)):
        print(f"  {i+1}. {CLASSES[class_id]} (Confidence: {score:.2f}) at {box.astype(int)}")
    
    stream.synchronize()
    d_input.free()
    d_output.free()

if __name__ == "__main__":
    engine_path = "yolov5.trt"
    image_path = "sample.jpg"
    output_image_path = "output.jpg"
    main(engine_path, image_path, output_image_path)
