import torch
from torchvision import transforms
from PIL import Image
import os
import time
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from src.models import Generator  # assumes models.py defines Generator

# Paths
FILE="woman"
IMAGE_PATH = f"input/{FILE}.png"  # Change this!
WEIGHTS_PATH = "weights/swift_srgan_4x.pth"
SR_PATH = f"output/sr_image_{FILE}.png"
HR_PATH = f"output/hr_image_{FILE}.png"  # Optional: save the original image
LR_PATH = f"output/lr_image_{FILE}.png"  # Optional: save the blurred image
dim = 1024
# Optional: save the blurred image

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
generator = Generator().to(device)
checkpoint = torch.load(WEIGHTS_PATH, map_location=device)
generator.load_state_dict(checkpoint["model"])
generator.eval()

# Load and preprocess input image
transform = transforms.Compose([
    transforms.CenterCrop((dim//4, dim//4)),  # or the original LR input size
    transforms.ToTensor(),
    #transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # match training
])

to_pil = transforms.ToPILImage()

image = Image.open(IMAGE_PATH).convert("RGB")
(width, height) = image.size
left = int((width - dim)/2)
right = left + dim
top = int((height - dim)/2)
bottom = top + dim

hr_image = image.crop((left,top,right,bottom))  # Resize to match the model input size
hr_image.save(HR_PATH)  # Save the original image if needed

# left_ = int((dim - dim//2)/2)
# right_ = left_ + dim//2
# top_ = int((dim - dim//2)/2)
# bottom_ = top + dim//2

down_image = hr_image.resize((dim//4, dim//4), Image.BICUBIC)  # Resize to match the model input size
down_image.save(LR_PATH)  # Save the blurred image if needed
lr_tensor = transform(down_image).unsqueeze(0).to(device)  # shape: [1, 3, H, W]


start_time = time.time()
# Inference
with torch.no_grad():
    sr_tensor = generator(lr_tensor)
end_time = time.time()
print(f"🕒 Inference time: {end_time - start_time:.2f} seconds")
# Post-process output
sr_tensor = sr_tensor.squeeze(0).to(device)

# Ensure the tensor is in the range [0, 1]
sr_tensor = torch.clamp(sr_tensor, 0, 1)

# Convert to image and save

sr_image = to_pil(sr_tensor)
# Make output directory if it doesn't exist
os.makedirs(os.path.dirname(SR_PATH), exist_ok=True)
sr_image.save(SR_PATH)

sr_image_np = np.array(sr_image)  # Super-resolved image
hr_image_np = np.array(hr_image)  # High-resolution ground truth image

# Calculate PSNR
psnr_value = psnr(hr_image_np, sr_image_np, data_range=255)
print(f"🔍 PSNR: {psnr_value:.2f} dB")

# Calculate SSIM
ssim_value = ssim(hr_image_np, sr_image_np, multichannel=True, data_range=255,win_size=3)
print(f"🔍 SSIM: {ssim_value:.4f}")

print(f"✅ Super-resolved image saved at {SR_PATH}")




