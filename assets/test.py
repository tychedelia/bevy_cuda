import torch
from ultralytics import YOLO
import numpy as np

# Global model instance
global_model = None

def load_yolo_model():
    global global_model
    if global_model is None:
        print("Loading YOLO segmentation model...")
        global_model = YOLO("yolo11s-seg.pt")  # Load segmentation model
        global_model.to("cuda")
    return global_model

def process_image(cuda_array):
    try:
        # Create tensor from CUDA array (ensure it's 8-bit unsigned integer format)
        tensor = torch.as_tensor(cuda_array, device='cuda', dtype=torch.uint8)
        tensor_float = tensor.float() / 255.0  # Normalize to 0-1 range

        # Ensure tensor is in RGB format (strip alpha if present)
        if tensor.shape[-1] == 4:
            tensor_float = tensor_float[:, :, :3]  # Remove alpha channel

        # Get original dimensions
        h, w, c = tensor_float.shape
        target_size = 640

        # Compute cropping region for best fit
        scale = min(w / target_size, h / target_size)
        new_w, new_h = int(target_size * scale), int(target_size * scale)
        start_x, start_y = (w - new_w) // 2, (h - new_h) // 2
        cropped_tensor = tensor_float[start_y:start_y + new_h, start_x:start_x + new_w]

        # Resize to 640x640 in 8-bit format
        resized_tensor = torch.nn.functional.interpolate(
            cropped_tensor.permute(2, 0, 1).unsqueeze(0),
            size=(target_size, target_size),
            mode='bilinear',
            align_corners=False
        ).squeeze(0).permute(1, 2, 0)

        # Convert back to 8-bit uint format
        resized_tensor = (resized_tensor * 255).byte()

        # Convert tensor to NCHW format for YOLO
        img_tensor = resized_tensor.permute(2, 0, 1).unsqueeze(0).contiguous()

        # Load YOLO model
        model = load_yolo_model()

        # Run inference on GPU
        results = model.predict(img_tensor.float() / 255.0, device='cuda', verbose=False)[0]

        # Extract detections
        boxes = results.boxes.xyxy  # Bounding boxes (x1, y1, x2, y2) on GPU
        masks = results.masks.data if results.masks is not None else None  # Segmentation masks

        # Draw detections on the tensor (GPU operations only)
        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes[i].int()

            # Draw bounding box in-place
            resized_tensor[y1:y2, x1:x1+2, 0] = 255  # Red vertical line (left)
            resized_tensor[y1:y2, x2-2:x2, 0] = 255  # Red vertical line (right)
            resized_tensor[y1:y1+2, x1:x2, 0] = 255  # Red horizontal line (top)
            resized_tensor[y2-2:y2, x1:x2, 0] = 255  # Red horizontal line (bottom)

            # Apply white mask over detected object
            if masks is not None and i < masks.shape[0]:
                mask = masks[i].unsqueeze(-1).expand(-1, -1, 3)  # Ensure it's 3-channel
                resized_tensor[mask > 0.5] = 255  # Apply white mask

        # Resize back to original cropped region
        restored_tensor = torch.nn.functional.interpolate(
            resized_tensor.permute(2, 0, 1).unsqueeze(0).float(),
            size=(new_h, new_w),
            mode='bilinear',
            align_corners=False
        ).squeeze(0).permute(1, 2, 0).byte()

        # Ensure the original tensor and restored tensor have matching channel count
        if tensor.shape[-1] == 4:
            restored_tensor = torch.cat([
                restored_tensor, torch.full((new_h, new_w, 1), 255, dtype=torch.uint8, device='cuda')
            ], dim=-1)

        # Write back into the original tensor region
        tensor[start_y:start_y + new_h, start_x:start_x + new_w] = restored_tensor

        return True
    except Exception as e:
        import traceback
        print(f"Error in YOLO segmentation script: {e}")
        traceback.print_exc()
        return False