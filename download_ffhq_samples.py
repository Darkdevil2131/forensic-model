from datasets import load_dataset
import os
import random

# -----------------------
# Settings
# -----------------------
NUM_IMAGES = 30
SAVE_DIR = "Benchmark/Original"

os.makedirs(SAVE_DIR, exist_ok=True)

print("Loading FFHQ dataset...")

dataset = load_dataset(
    "student/FFHQ",
    split="train",
    streaming=True
)

dataset = list(dataset.take(500))

random.shuffle(dataset)

print(f"Downloading {NUM_IMAGES} images...\n")

saved = 0

for sample in dataset:
    try:
        img = sample["image"]

        filename = os.path.join(
            SAVE_DIR,
            f"face_{saved+1:03d}.png"
        )

        img.save(filename)

        print(f"Saved {filename}")

        saved += 1

        if saved >= NUM_IMAGES:
            break

    except Exception:
        continue

print("\nDone!")
print(f"Saved {saved} images.")