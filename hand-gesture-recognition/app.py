import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image
import json
from models import build_resnet18
from utils import load_image_tensor


# Sidebar
st.sidebar.title("Hand Gesture Recognition")
st.sidebar.write("Upload an image of a hand gesture and let the model classify it.")


# Load model + labels
@st.cache_resource
def load_model(weights_path, labels_path, device="cpu"):
	with open(labels_path, "r", encoding="utf-8") as f:
		class_names = json.load(f)
	model = build_resnet18(num_classes=len(class_names), pretrained=False)
	model.load_state_dict(torch.load(weights_path, map_location=device))
	model.eval()
	return model, class_names


weights = "artifacts/best_model.pth"
labels = "artifacts/class_names.json"
device = "cuda" if torch.cuda.is_available() else "cpu"
model, class_names = load_model(weights, labels, device)


# Upload image
uploaded_file = st.file_uploader("Upload a hand gesture image", type=["jpg", "jpeg", "png"])


if uploaded_file is not None:
	image = Image.open(uploaded_file).convert("RGB")
	st.image(image, caption="Uploaded Image", use_column_width=True)

	x = load_image_tensor(uploaded_file, img_size=224, device=device)
	with torch.no_grad():
		logits = model(x)
		probs = F.softmax(logits, dim=1).cpu().numpy()[0]

	top_idx = int(probs.argmax())
	st.markdown(f"### Prediction: **{class_names[top_idx]}** ({probs[top_idx]:.2f})")

	st.write("#### Top-5 Probabilities")
	for i in probs.argsort()[::-1][:5]:
		st.write(f"{class_names[i]}: {probs[i]:.3f}")