def process_image(cuda_array):
    try:
        print(f"Processing image with shape: {cuda_array.shape}")

        import torch
        import os
        from datetime import datetime
        from PIL import Image
        import numpy as np

        # Create output directory
        output_dir = "filtered_images"
        os.makedirs(output_dir, exist_ok=True)

        # Create tensor from CUDA array
        tensor = torch.as_tensor(cuda_array, device='cuda')

        # Convert to float for processing (0-1 range)
        tensor_float = tensor.float() / 255.0

        # Apply a cool filter: Blue-shift nighttime effect
        filtered = tensor_float.clone()

        # Reduce red channel, enhance blue
        filtered[:, :, 0] *= 0.7  # Reduce red
        filtered[:, :, 2] *= 1.3  # Enhance blue

        # Add a subtle vignette effect
        h, w, _ = filtered.shape
        y = torch.linspace(-1, 1, h, device='cuda')[:, None]
        x = torch.linspace(-1, 1, w, device='cuda')[None, :]
        radius = torch.sqrt(x.pow(2) + y.pow(2))
        vignette = torch.clamp(1.2 - radius, 0.6, 1.0)

        # Apply vignette to RGB channels
        for c in range(3):
            filtered[:, :, c] *= vignette

        # Clamp values to valid range
        filtered = torch.clamp(filtered, 0.0, 1.0)

        # Write filtered image back to shared memory
        tensor[:] = (filtered * 255.0).byte()

        # Save a copy of the processed image to disk
        output_image = (filtered.cpu().numpy() * 255).astype(np.uint8)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        Image.fromarray(output_image).save(f"{output_dir}/blue_night_{timestamp}.png")

        # Synchronize CUDA operations
        torch.cuda.synchronize()

        print(f"Image processed and saved to {output_dir}/blue_night_{timestamp}.png")
        return True

    except Exception as e:
        import traceback
        print(f"Error in filter script: {e}")
        traceback.print_exc()
        return False