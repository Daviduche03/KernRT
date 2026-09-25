"""Train a real 784->256->10 MLP on MNIST with MLX and export it for KernRT.

Produces, under `target/`:
  mnist.safetensors       fc1.weight, fc1.bias, fc2.weight, fc2.bias
  mnist_test.safetensors  input [N,784], label [N], logits [N,10] (MLX reference)

Run with the project's MLX venv, e.g.:
  SDKROOT=<xcode sdk> /path/to/mlxvenv/bin/python scripts/train_mnist_mlp.py
"""

import gzip
import struct
import urllib.request
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import mlx.utils
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "target" / "mnist_raw"
MODEL_OUT = ROOT / "target" / "mnist.safetensors"
TEST_OUT = ROOT / "target" / "mnist_test.safetensors"

BASE = "https://storage.googleapis.com/cvdf-datasets/mnist"
FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}
HIDDEN = 256
EPOCHS = 8
BATCH = 128
TEST_SAMPLES = 1000


def download() -> dict[str, Path]:
    DATA.mkdir(parents=True, exist_ok=True)
    paths = {}
    for key, name in FILES.items():
        path = DATA / name
        if not path.exists():
            url = f"{BASE}/{name}"
            print(f"downloading {url}")
            urllib.request.urlretrieve(url, path)
        paths[key] = path
    return paths


def load_images(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        magic, count, rows, cols = struct.unpack(">IIII", f.read(16))
        assert magic == 2051, magic
        data = np.frombuffer(f.read(), dtype=np.uint8).reshape(count, rows * cols)
    return data.astype(np.float32) / 255.0


def load_labels(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        magic, count = struct.unpack(">II", f.read(8))
        assert magic == 2049, magic
        return np.frombuffer(f.read(), dtype=np.uint8).astype(np.int32)


class Mlp(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.fc1 = nn.Linear(784, hidden)
        self.fc2 = nn.Linear(hidden, 10)

    def __call__(self, x):
        return self.fc2(mx.maximum(self.fc1(x), 0.0))


def loss_fn(model, x, y):
    return nn.losses.cross_entropy(model(x), y, reduction="mean")


def main():
    paths = download()
    x_train = load_images(paths["train_images"])
    y_train = load_labels(paths["train_labels"])
    x_test = load_images(paths["test_images"])
    y_test = load_labels(paths["test_labels"])
    print(f"train {x_train.shape} test {x_test.shape}")

    model = Mlp(HIDDEN)
    mx.eval(model.parameters())
    optimizer = optim.Adam(learning_rate=1e-3)
    loss_and_grad = nn.value_and_grad(model, loss_fn)

    n = x_train.shape[0]
    for epoch in range(EPOCHS):
        order = np.random.default_rng(epoch).permutation(n)
        total = 0.0
        for start in range(0, n, BATCH):
            idx = order[start : start + BATCH]
            xb = mx.array(x_train[idx])
            yb = mx.array(y_train[idx])
            loss, grads = loss_and_grad(model, xb, yb)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)
            total += loss.item()
        print(f"epoch {epoch + 1}/{EPOCHS} loss {total / (n // BATCH):.4f}")

    # Reference accuracy on the whole test set.
    test_logits = model(mx.array(x_test))
    mx.eval(test_logits)
    test_pred = np.array(mx.argmax(test_logits, axis=1))
    print(f"full test accuracy: {(test_pred == y_test).mean() * 100:.2f}%")

    # Export the model exactly as MLX names it: fc1.weight is [out, in].
    weights = dict(mlx.utils.tree_flatten(model.parameters()))
    mx.save_safetensors(str(MODEL_OUT), weights)
    for name, arr in weights.items():
        print(f"  {name}: {arr.shape} {arr.dtype}")

    # Export a test slice with the reference logits for Rust to match.
    x_slice = mx.array(x_test[:TEST_SAMPLES])
    logits = model(x_slice)
    mx.eval(logits)
    mx.save_safetensors(
        str(TEST_OUT),
        {
            "input": x_slice,
            "label": mx.array(y_test[:TEST_SAMPLES].astype(np.float32)),
            "logits": logits,
        },
    )
    print(f"wrote {MODEL_OUT} and {TEST_OUT}")


if __name__ == "__main__":
    main()
