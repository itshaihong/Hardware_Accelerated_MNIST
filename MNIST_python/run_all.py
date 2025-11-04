import os
import subprocess
import sys

# Simple orchestrator to train then evaluate on idx files
def run_all():
    os.makedirs('weights', exist_ok=True)

    # Train model
    print("Step 1: Training LeNet-5 on CPU...")
    ret = subprocess.call([sys.executable, "train_lenet5.py"])
    if ret != 0:
        print("Error: Training failed")
        return

    # Evaluate on idx test files
    print("Step 2: Evaluating on idx test files...")
    ret = subprocess.call([sys.executable, "infer_idx.py", 
                           "--images", "t10k-images.idx3-ubyte",
                           "--labels", "t10k-labels.idx1-ubyte",
                           "--model", "weights/lenet5.pth",
                           "--batch_size", "1000"])
    if ret != 0:
        print("Error: Evaluation failed")
        return

    print("Done.")

if __name__ == "__main__":
    run_all()