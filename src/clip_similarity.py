import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import argparse

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "openai/clip-vit-large-patch14"

model = CLIPModel.from_pretrained(MODEL_NAME).to(DEVICE)
processor = CLIPProcessor.from_pretrained(MODEL_NAME)

def image_similarity(img1_path, img2_path):
    img1 = Image.open(img1_path).convert("RGB")
    img2 = Image.open(img2_path).convert("RGB")

    inputs = processor(images=[img1, img2], return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        features = model.get_image_features(**inputs)
        features = features / features.norm(dim=-1, keepdim=True)

    similarity = torch.matmul(features[0], features[1]).item()
    return similarity

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute CLIP similarity between two images")
    parser.add_argument("img1", type=str, help="Path to first image")
    parser.add_argument("img2", type=str, help="Path to second image")
    args = parser.parse_args()

    sim = image_similarity(args.img1, args.img2)
    print(f"CLIP similarity: {sim:.4f}")
