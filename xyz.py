import torch
from yolov5.models.experimental import attempt_load  # Adjust path if needed

# Load pre-trained model (no map_location)
model = attempt_load('yolov5s.pt')  # Load model
model.eval()

# Move to CPU manually if needed
model = model.to('cpu')

# Define dummy input with dynamic batch
dummy_input = torch.randn(1, 3, 640, 640)  # Initial input
dynamic_axes = {'images': {0: 'batch_size'}}  # Dynamic batch dimension

# Export to ONNX
torch.onnx.export(model, dummy_input, "yolov5s_dynamic.onnx", input_names=['images'], output_names=['output'], dynamic_axes=dynamic_axes, opset_version=11, verbose=True)
print("ONNX model exported to yolov5s_dynamic.onnx")
