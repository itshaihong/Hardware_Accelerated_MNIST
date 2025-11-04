import os
import numpy as np
import torch
from lenet5 import LeNet5

# Export trained LeNet-5 weights to float32 binary files for OpenCL GPU inference
def export_weights(model_path="weights/lenet5.pth", out_dir="weights_fp32"):
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device("cpu")

    model = LeNet5().to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # Conv1: [6,1,5,5]
    w = model.conv1.weight.detach().cpu().numpy().astype(np.float32)
    b = model.conv1.bias.detach().cpu().numpy().astype(np.float32)
    w.tofile(os.path.join(out_dir, "conv1_weight.bin"))
    b.tofile(os.path.join(out_dir, "conv1_bias.bin"))

    # Conv2: [16,6,5,5]
    w = model.conv2.weight.detach().cpu().numpy().astype(np.float32)
    b = model.conv2.bias.detach().cpu().numpy().astype(np.float32)
    w.tofile(os.path.join(out_dir, "conv2_weight.bin"))
    b.tofile(os.path.join(out_dir, "conv2_bias.bin"))

    # FC1: [120, 400]
    w = model.fc1.weight.detach().cpu().numpy().astype(np.float32)
    b = model.fc1.bias.detach().cpu().numpy().astype(np.float32)
    w.tofile(os.path.join(out_dir, "fc1_weight.bin"))
    b.tofile(os.path.join(out_dir, "fc1_bias.bin"))

    # FC2: [84, 120]
    w = model.fc2.weight.detach().cpu().numpy().astype(np.float32)
    b = model.fc2.bias.detach().cpu().numpy().astype(np.float32)
    w.tofile(os.path.join(out_dir, "fc2_weight.bin"))
    b.tofile(os.path.join(out_dir, "fc2_bias.bin"))

    # FC3: [10, 84]
    w = model.fc3.weight.detach().cpu().numpy().astype(np.float32)
    b = model.fc3.bias.detach().cpu().numpy().astype(np.float32)
    w.tofile(os.path.join(out_dir, "fc3_weight.bin"))
    b.tofile(os.path.join(out_dir, "fc3_bias.bin"))

    with open(os.path.join(out_dir, "shapes.txt"), "w") as f:
        f.write("conv1_weight: (6,1,5,5)\n")
        f.write("conv1_bias: (6,)\n")
        f.write("conv2_weight: (16,6,5,5)\n")
        f.write("conv2_bias: (16,)\n")
        f.write("fc1_weight: (120,400)\n")
        f.write("fc1_bias: (120,)\n")
        f.write("fc2_weight: (84,120)\n")
        f.write("fc2_bias: (84,)\n")
        f.write("fc3_weight: (10,84)\n")
        f.write("fc3_bias: (10,)\n")

    print(f"Exported weights to {out_dir}")

if __name__ == "__main__":
    export_weights()