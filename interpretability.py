import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import slic
from lime import lime_image
from src.datasets.dataset import UnderwaterDataset
from src.models.baseline_cnn import BaselineCNN
import os

# ------------------------------
# 1️⃣ Device and model
# ------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load trained model
model = BaselineCNN().to(device)
model.load_state_dict(torch.load("models/baseline_cnn.pth", map_location=device))
model.eval()

# ------------------------------
# 2️⃣ Load a sample validation image
# ------------------------------
val_dataset = UnderwaterDataset(
    input_dir=r"R:\underwater project\archive\valid\input\images",
    target_dir=r"R:\underwater project\archive\valid\target\images",
    augment=False
)

input_img, target_img = val_dataset[0]  # first sample
input_img = input_img.unsqueeze(0).to(device)  # add batch dimension

# ------------------------------
# 3️⃣ Grad-CAM
# ------------------------------
last_conv_layer = model.model[2]  # Last Conv2d layer

input_img.requires_grad = True
output = model(input_img)
grads = torch.autograd.grad(output.sum(), last_conv_layer.weight, retain_graph=True)[0]

weights = grads.mean(dim=(2,3), keepdim=True)
cam = (weights * last_conv_layer.weight).sum(dim=1, keepdim=True)
cam = F.relu(cam)
cam = cam - cam.min()
cam = cam / cam.max()
cam_np = cam.detach().cpu().numpy()[0,0]

# ------------------------------
# 4️⃣ LIME
# ------------------------------
explainer = lime_image.LimeImageExplainer()

def predict_fn(images):
    imgs = torch.tensor(images).permute(0,3,1,2).float().to(device)
    with torch.no_grad():
        outputs = model(imgs)
    return outputs.cpu().numpy().reshape(len(images), -1)

explanation = explainer.explain_instance(
    np.array(input_img.squeeze().permute(1,2,0)),
    classifier_fn=predict_fn,
    segmentation_fn=lambda x: slic(x, n_segments=50, compactness=10),
    top_labels=1
)

temp, mask = explanation.get_image_and_mask(
    label=0, positive_only=True, hide_rest=False
)

# ------------------------------
# 5️⃣ Visualization and saving
# ------------------------------
save_dir = r"outputs/interpretability"
os.makedirs(save_dir, exist_ok=True)

fig, axs = plt.subplots(2,2, figsize=(12,12))

# Original input
axs[0,0].imshow(input_img.squeeze().permute(1,2,0).cpu())
axs[0,0].set_title("Input Underwater Image")
axs[0,0].axis("off")

# Ground truth
axs[0,1].imshow(target_img.permute(1,2,0))
axs[0,1].set_title("Ground Truth Enhanced Image")
axs[0,1].axis("off")

# Grad-CAM
axs[1,0].imshow(input_img.squeeze().permute(1,2,0).cpu())
axs[1,0].imshow(cam_np, cmap='jet', alpha=0.5)
axs[1,0].set_title("Grad-CAM Heatmap")
axs[1,0].axis("off")

# LIME
axs[1,1].imshow(temp)
axs[1,1].set_title("LIME Superpixels")
axs[1,1].axis("off")

plt.tight_layout()
save_path = os.path.join(save_dir, "interpretability_sample.png")
plt.savefig(save_path)
print(f"Interpretability figure saved to {save_path}")
plt.close()
