## 🖼️ Background Remover (U²-Net)

A simple, command-line Python tool to **remove image backgrounds** using the [U²-Net](https://github.com/xuebinqin/U-2-Net) deep-learning model.
It supports single images or entire folders and outputs PNGs with transparent backgrounds.

---

### 🚀 Features

* 🔹 Works on **any common image format** (JPG, PNG, BMP, TIFF, etc.).
* 🔹 Handles **single files or folders**.
* 🔹 Automatically **downloads the U²-Net model** if missing.
* 🔹 Saves clean cut-outs as PNGs with transparency.
* 🔹 Gracefully skips unsupported or corrupted files.
* 🔹 Runs on **GPU (CUDA)** if available, else falls back to CPU.

---

### 📦 Requirements

Install dependencies using:

```bash
pip install -r requirements.txt
```

**requirements.txt**

```txt
torch>=2.0.0
torchvision>=0.15.0
Pillow>=10.0.0
numpy>=1.24.0
requests>=2.31.0
tqdm>=4.66.0
```

---

### ⚙️ Setup

1. Clone or download this project.
2. (Optional) Create and activate a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate   # on macOS/Linux
   venv\Scripts\activate      # on Windows
   ```
3. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

When first run, the script will **automatically download** the pre-trained `u2net.pth` model (~176 MB) from the official source.

---

### 🧠 Usage

#### 🖼️ Single image

```bash
python remove_bg.py --input path/to/image.jpg --output path/to/output_folder
```

#### 📁 Whole folder

```bash
python remove_bg.py --input path/to/input_folder --output path/to/output_folder
```

All processed images will be saved as **.png** files (to preserve transparency) in the specified output directory.

---

### 🧩 Example

```bash
python remove_bg.py --input ./photos --output ./results
```

**Input:**
`photos/dog.jpg`

**Output:**
`results/dog.png` (dog cut out with transparent background)

---

### ⚡ Notes

* The script uses **U²-Net (salient object detection)**, which performs well on most subjects (people, animals, objects).
* For faster but smaller models, you can swap in **U²-Netp (lightweight)** if you download `u2netp.pth`.
* Works best on images with clear foreground/background separation.
* You can extend it to use **U²-Net-human-seg** for portraits.

---

### 🧰 Credits

* **Model:** [U²-Net – Qin et al., Pattern Recognition 2020](https://github.com/xuebinqin/U-2-Net)
* **Implementation:** based on PyTorch, with image handling via Pillow.
