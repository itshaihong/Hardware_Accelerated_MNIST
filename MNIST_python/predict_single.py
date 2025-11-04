import os
import time
import argparse
import numpy as np
from PIL import Image, ImageOps
import torch
import torch.nn.functional as F
from lenet5 import LeNet5

# Open image robustly and composite transparency onto white background if needed
def open_grayscale_robust(image_path):
    img = Image.open(image_path)
    if img.mode == "RGBA":
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[-1])
        img = bg.convert("L")
    elif img.mode == "LA":
        l, a = img.split()
        bg = Image.new("L", img.size, 255)
        bg.paste(l, mask=a)
        img = bg
    elif img.mode == "P":
        img = img.convert("RGBA")
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[-1])
        img = bg.convert("L")
    else:
        img = img.convert("L")
    return img

# Preprocess: ensure 28x28, optional invert, normalize as training
def load_and_preprocess(image_path, invert=False, auto_invert=True, save_preprocessed=None):
    img = open_grayscale_robust(image_path)

    if img.size != (28, 28):
        img = img.resize((28, 28), resample=Image.BILINEAR)

    arr01 = np.array(img).astype(np.float32) / 255.0

    if auto_invert and not invert:
        mean_intensity = float(arr01.mean())
        if mean_intensity > 0.5:
            img = ImageOps.invert(img)
            arr01 = 1.0 - arr01

    if invert and not auto_invert:
        img = ImageOps.invert(img)
        arr01 = 1.0 - arr01

    if save_preprocessed:
        os.makedirs(os.path.dirname(save_preprocessed) or ".", exist_ok=True)
        img.save(save_preprocessed)

    mean, std = 0.1307, 0.3081
    arr_norm = (arr01 - mean) / std
    arr_norm = arr_norm[None, None, :, :]
    x = torch.from_numpy(arr_norm).float()

    print(f"Preprocess stats: min={arr01.min():.3f}, max={arr01.max():.3f}, mean={arr01.mean():.3f}")
    return x

def predict_single(image_path, model_path="weights/lenet5.pth",
                   invert=False, auto_invert=True,
                   save_preprocessed=None, topk=3):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    device = torch.device("cpu")
    x = load_and_preprocess(
        image_path,
        invert=invert,
        auto_invert=auto_invert,
        save_preprocessed=save_preprocessed
    ).to(device)

    model = LeNet5().to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    with torch.no_grad():
        t0 = time.perf_counter()
        logits = model(x)
        t1 = time.perf_counter()

        probs = F.softmax(logits, dim=1).cpu().numpy()[0]
        pred = int(np.argmax(probs))
        elapsed_ms = (t1 - t0) * 1000.0

        top_idx = np.argsort(-probs)[:topk]
        print(f"Prediction: {pred}")
        print("Top-{}: {}".format(topk, ", ".join([f"{i}:{probs[i]:.4f}" for i in top_idx])))
        print(f"Inference time: {elapsed_ms:.3f} ms")
    return pred, elapsed_ms

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict a single 28x28 MNIST-like grayscale image with LeNet-5 (CPU)")
    parser.add_argument("--image", type=str, required=True, help="Path to a 28x28 grayscale PNG/JPG")
    parser.add_argument("--model", type=str, default="weights/lenet5.pth", help="Path to saved model .pth")
    parser.add_argument("--invert", action="store_true", help="Force invert (white background -> black background)")
    parser.add_argument("--no_auto_invert", action="store_true", help="Disable auto-invert by mean intensity")
    parser.add_argument("--save_preprocessed", type=str, default="", help="Optional path to save the preprocessed 28x28 image")
    parser.add_argument("--topk", type=int, default=3, help="Show top-K probabilities")
    args = parser.parse_args()

    predict_single(
        image_path=args.image,
        model_path=args.model,
        invert=args.invert,
        auto_invert=(not args.no_auto_invert),
        save_preprocessed=(args.save_preprocessed if args.save_preprocessed else None),
        topk=args.topk
    )