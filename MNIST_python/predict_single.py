import argparse
import time
from typing import Tuple

import numpy as np
from PIL import Image, ImageOps

import torch
import torch.nn.functional as F

from lenet5 import LeNet5

# Consistent normalization with training
MEAN = 0.1307
STD = 0.3081


def load_and_preprocess_image(
    path: str,
    invert: str = "auto",
    save_preprocessed: str = None
) -> torch.Tensor:
    """
    Load a single image, convert to 28x28 grayscale, optionally invert,
    normalize to match training, and return a tensor of shape (1, 1, 28, 28).
    - invert: "auto" | "yes" | "no"
    - save_preprocessed: optional path to save the 28x28 preprocessed PNG (uint8 0..255)
    """
    # Load image
    img = Image.open(path).convert("L")  # grayscale

    # Resize to 28x28 (keep digit crisp)
    if img.size != (28, 28):
        img = img.resize((28, 28), resample=Image.NEAREST)

    # Convert to numpy float in [0,1]
    arr = np.array(img, dtype=np.uint8)
    arr_f = arr.astype(np.float32) / 255.0

    # Decide inversion
    do_invert = False
    if invert == "yes":
        do_invert = True
    elif invert == "no":
        do_invert = False
    else:  # auto
        # If background is bright (mean > 0.5), invert to get white digit on black background
        if arr_f.mean() > 0.5:
            do_invert = True

    if do_invert:
        arr_f = 1.0 - arr_f

    # Optionally save the preprocessed image (uint8 0..255)
    if save_preprocessed:
        to_save = (arr_f * 255.0).clip(0, 255).astype(np.uint8)
        Image.fromarray(to_save, mode="L").save(save_preprocessed)

    # Normalize using training parameters: (x - mean) / std
    arr_norm = (arr_f - MEAN) / STD

    # To tensor shape (1, 1, 28, 28)
    x = torch.from_numpy(arr_norm).unsqueeze(0).unsqueeze(0).contiguous()
    x = x.to(dtype=torch.float32)
    return x


def run_inference(model_path: str, x: torch.Tensor, topk: int = 3) -> Tuple[int, np.ndarray, float]:
    """
    Load model weights, run a single forward pass on CPU, and return:
    - predicted class (int)
    - top-k probabilities array of shape (k, 2): [(class, prob), ...]
    - forward pass time in milliseconds
    """
    device = torch.device("cpu")
    model = LeNet5(num_classes=10)
    state = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    with torch.no_grad():
        start = time.perf_counter()
        logits = model(x.to(device))
        end = time.perf_counter()
        forward_ms = (end - start) * 1000.0

        probs = F.softmax(logits, dim=1).cpu().numpy()[0]
        pred = int(np.argmax(probs))

        # Top-k
        k = max(1, min(topk, probs.shape[0]))
        top_idx = probs.argsort()[-k:][::-1]
        top = np.array([[int(i), float(probs[i])] for i in top_idx], dtype=object)

    return pred, top, forward_ms


def main():
    parser = argparse.ArgumentParser(description="Single-image inference for LeNet-5 (CPU)")
    parser.add_argument("--image", type=str, required=True, help="Path to 28x28 grayscale image (PNG recommended)")
    parser.add_argument("--model", type=str, default="weights/lenet5.pth", help="Path to model weights")
    parser.add_argument("--invert", type=str, choices=["auto", "yes", "no"], default="auto",
                        help="Invert image colors: auto (default), yes, or no")
    parser.add_argument("--topk", type=int, default=3, help="Number of top probabilities to display")
    parser.add_argument("--save_preprocessed", type=str, default=None,
                        help="Optional path to save the 28x28 preprocessed PNG")
    args = parser.parse_args()

    # Load and preprocess
    x = load_and_preprocess_image(args.image, invert=args.invert, save_preprocessed=args.save_preprocessed)

    # Run inference
    pred, top, forward_ms = run_inference(args.model, x, topk=args.topk)

    # Output
    print(f"Prediction: {pred}")
    print("Top-{} probabilities:".format(len(top)))
    for cls, prob in top:
        print(f"  class {cls}: {prob:.4f}")
    print(f"Single inference time (forward pass): {forward_ms:.3f} ms")


if __name__ == "__main__":
    main()