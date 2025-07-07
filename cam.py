import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import cv2
import time

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

CONF_THRESH = 0.3
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

def allocate_buffers(engine):
    input_binding = engine.get_tensor_name(0)
    output_binding = engine.get_tensor_name(1)
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

def preprocess(img):
    img_resized = cv2.resize(img, (640, 640))
    img_input = img_resized.transpose(2, 0, 1).astype(np.float32) / 255.0
    return np.expand_dims(img_input, axis=0), img_resized

def do_inference(context, bindings, d_input, d_output, stream, input_data, output_shape, input_binding):
    cuda.memcpy_htod_async(d_input, input_data, stream)
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    output = np.empty(output_shape, dtype=np.float32)
    cuda.memcpy_dtoh_async(output, d_output, stream)
    stream.synchronize()
    return output

def postprocess(output, original_shape, input_shape, conf_thres=CONF_THRESH, iou_thres=IOU_THRESH):
    boxes = output[0, :, :4]
    obj_conf = output[0, :, 4]
    cls_conf = output[0, :, 5:]

    scores = obj_conf[..., None] * cls_conf
    max_scores = np.max(scores, axis=-1)
    class_ids = np.argmax(scores, axis=-1)

    mask = max_scores > conf_thres
    boxes = boxes[mask]
    scores = max_scores[mask]
    class_ids = class_ids[mask]

    if boxes.shape[0] == 0:
        return [], [], []

    h, w = original_shape
    scale_x = w / input_shape[3]
    scale_y = h / input_shape[2]

    boxes[:, [0, 2]] *= scale_x
    boxes[:, [1, 3]] *= scale_y

    boxes_xyxy = np.zeros_like(boxes)
    boxes_xyxy[:, 0] = np.clip(boxes[:, 0] - boxes[:, 2] / 2, 0, w - 1)
    boxes_xyxy[:, 1] = np.clip(boxes[:, 1] - boxes[:, 3] / 2, 0, h - 1)
    boxes_xyxy[:, 2] = np.clip(boxes[:, 0] + boxes[:, 2] / 2, 0, w - 1)
    boxes_xyxy[:, 3] = np.clip(boxes[:, 1] + boxes[:, 3] / 2, 0, h - 1)

    indices = cv2.dnn.NMSBoxes(boxes_xyxy.tolist(), scores.tolist(), conf_thres, iou_thres)
    if len(indices) > 0:
        indices = indices.flatten() if isinstance(indices, np.ndarray) else np.array(indices)
        return boxes_xyxy[indices], scores[indices], class_ids[indices]
    else:
        return [], [], []

def draw_boxes(image, boxes, scores, class_ids):
    for box, score, class_id in zip(boxes, scores, class_ids):
        x1, y1, x2, y2 = map(int, box)
        label = f"{CLASSES[class_id]}: {score:.2f}"
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return image

def main():
    engine_path = "yolov5.trt"
    engine = load_engine(engine_path)
    context = engine.create_execution_context()

    d_input, d_output, bindings, input_shape, output_shape, input_binding, output_binding = allocate_buffers(engine)
    context.set_input_shape(input_binding, (1, 3, 640, 640))
    output_shape = (1, *output_shape[1:])

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Unable to access USB camera")

    print("Press 'q' to quit...")
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        input_data, img_resized = preprocess(frame)
        stream = cuda.Stream()
        output = do_inference(context, bindings, d_input, d_output, stream, input_data.ravel(), output_shape, input_binding)
        boxes, scores, class_ids = postprocess(output, frame.shape[:2], input_shape)

        output_image = draw_boxes(frame, boxes, scores, class_ids)
        cv2.imshow("YOLOv5 TensorRT Detection", output_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    d_input.free()
    d_output.free()

if __name__ == "__main__":
    main()

