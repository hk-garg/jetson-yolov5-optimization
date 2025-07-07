# All imports and constants same as before
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import cv2
import os
import time

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
CONF_THRESH = 0.1
IOU_THRESH = 0.4

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

def load_engine(engine_path):
    with open(engine_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

def preprocess_batch(image_paths, input_shape=(1, 3, 640, 640)):
    batch_images, original_shapes, img_resized_list = [], [], []
    for img_path in image_paths:
        if not os.path.exists(img_path):
            print(f"Warning: Image {img_path} not found, skipping.")
            continue
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Failed to load image: {img_path}, skipping.")
            continue
        original_shapes.append(img.shape[:2])
        img_resized = cv2.resize(img, (input_shape[2], input_shape[3]))
        img = img_resized.transpose(2, 0, 1).astype(np.float32) / 255.0
        batch_images.append(img)
        img_resized_list.append(img_resized)
    if len(batch_images) == 0:
        raise ValueError("No images were successfully preprocessed.")
    batch_tensor = np.stack(batch_images, axis=0)
    return batch_tensor, original_shapes, img_resized_list

def allocate_buffers(engine):
    binding_names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
    input_binding = binding_names[0]
    output_binding = binding_names[1]

    input_shape = engine.get_tensor_shape(input_binding)
    output_shape = engine.get_tensor_shape(output_binding)

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
    cuda.memcpy_htod_async(d_input, input_data, stream)
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    output = np.empty(output_shape, dtype=np.float32)
    cuda.memcpy_dtoh_async(output, d_output, stream)
    stream.synchronize()
    return output

def postprocess(output, original_shapes, input_shape, conf_thres=CONF_THRESH, iou_thres=IOU_THRESH):
    results = []
    input_h, input_w = input_shape[-2], input_shape[-1]
    batch_size = output.shape[0]

    for i in range(batch_size):
        if i >= len(original_shapes):
            results.append(([], [], []))
            continue
        boxes = output[i, ..., :4]
        obj_conf = output[i, ..., 4]
        cls_conf = output[i, ..., 5:]

        scores = obj_conf[..., None] * cls_conf
        max_scores = np.max(scores, axis=-1)
        max_classes = np.argmax(scores, axis=-1)

        mask = max_scores > conf_thres
        boxes = boxes[mask]
        scores = max_scores[mask]
        class_ids = max_classes[mask]

        if len(boxes) == 0:
            results.append(([], [], []))
            continue

        h, w = original_shapes[i]
        scale_x = w / input_w
        scale_y = h / input_h
        boxes[:, [0, 2]] *= scale_x
        boxes[:, [1, 3]] *= scale_y

        boxes_xyxy = np.zeros_like(boxes)
        boxes_xyxy[:, 0] = np.clip(boxes[:, 0] - boxes[:, 2] / 2, 0, w - 1)
        boxes_xyxy[:, 1] = np.clip(boxes[:, 1] - boxes[:, 3] / 2, 0, h - 1)
        boxes_xyxy[:, 2] = np.clip(boxes[:, 0] + boxes[:, 2] / 2, 0, w - 1)
        boxes_xyxy[:, 3] = np.clip(boxes[:, 1] + boxes[:, 3] / 2, 0, h - 1)

        indices = cv2.dnn.NMSBoxes(boxes_xyxy.tolist(), scores.tolist(), conf_thres, iou_thres)
        if len(indices) > 0:
            indices = np.array(indices).flatten()
            boxes = boxes_xyxy[indices]
            scores = scores[indices]
            class_ids = class_ids[indices]
        else:
            boxes, scores, class_ids = np.array([]), np.array([]), np.array([])

        results.append((boxes, scores, class_ids))
    return results

def draw_boxes(image, boxes, scores, class_ids):
    for box, score, class_id in zip(boxes, scores, class_ids):
        x1, y1, x2, y2 = map(int, box)
        if x1 < 0 or y1 < 0 or x2 > image.shape[1] or y2 > image.shape[0]:
            continue
        label = f"{CLASSES[class_id]}: {score:.2f}"
        color = (0, 255, 0)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return image

def main(engine_path, image_dir):
#    image_paths = [os.path.join(image_dir, f"image{i}.jpg") for i in range(1, 6)]
    image_paths = sorted([
    os.path.join(image_dir, f)
    for f in os.listdir(image_dir)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
	])

    
    if not any(os.path.exists(path) for path in image_paths):
        raise ValueError("No images found in the directory.")

    engine = load_engine(engine_path)
    context = engine.create_execution_context()
    d_input, d_output, bindings, input_shape, output_shape, input_binding, output_binding = allocate_buffers(engine)

    try:
        input_data, original_shapes, img_resized_list = preprocess_batch(image_paths, input_shape)
    except ValueError as e:
        print(f"Error preprocessing images: {e}")
        return

    for i in range(len(image_paths)):
        if i >= len(img_resized_list):
            continue

        input_tensor_name = engine.get_tensor_name(0)
        context.set_input_shape(input_tensor_name, (1, 3, 640, 640))
        output_shape = list(output_shape)
        output_shape[0] = 1
        output_shape = tuple(output_shape)

        stream = cuda.Stream()
        start_time = time.time()
        output = do_inference(context, bindings, d_input, d_output, stream, input_data[i:i+1].ravel(), output_shape, input_binding)
        inference_time = (time.time() - start_time) * 1000
        results = postprocess(output, [original_shapes[i]], input_shape)

        print(f"\n=== Image {i+1} ===")
        print(f"Inference time: {inference_time:.2f} ms")
        if results and len(results[0][0]) > 0:
            boxes, scores, class_ids = results[0]
            for j, (box, score, class_id) in enumerate(zip(boxes, scores, class_ids)):
                print(f"{j+1}. Class: {CLASSES[class_id]}, Confidence Score: {score:.2f}, at: {box.astype(int)}")
        else:
            print("No valid detections.")

    stream.synchronize()
    d_input.free()
    d_output.free()

if __name__ == "__main__":
    engine_path = "yolov5.trt"
    image_dir = "./images"
    main(engine_path, image_dir)

