# GAN Anime Faces — README

## 📌 Overview
This project implements a **DCGAN-style Generative Adversarial Network (GAN)** in PyTorch to generate anime faces. The GAN learns from thousands of anime face images and produces new faces that resemble the dataset.

The workflow includes:
1. Preparing the Anime Face dataset.
2. Training the GAN model (Generator + Discriminator).
3. Generating new anime face images from the trained Generator.

---

## 📂 Project Structure
```
GAN_Anime_Faces_Project/
│
├── gan_anime_faces.py        # Main script (training + sampling)
├── README.md                 # Documentation (this file)
│
├── animefaces/               # Dataset folder (unzipped images go here)
│   ├── 0.jpg
│   ├── 1.jpg
│   ├── 2.jpg
│   └── ...
│
├── output/                   # Auto-created during training
│   ├── samples/              # Generated sample images during training
│   │   ├── sample_000000.png
│   │   ├── sample_005000.png
│   │   └── ...
│   ├── samples_manual/       # Images generated manually after training
│   ├── checkpoint_epoch_1.pth
│   ├── checkpoint_epoch_2.pth
│   ├── generator_final.pth
│   └── discriminator_final.pth
│
└── requirements.txt          # Dependencies (optional)
```

---

## 📦 Requirements
- Python 3.8+
- PyTorch (>=1.10)
- torchvision
- Pillow
- tqdm

Install with:
```bash
pip install torch torchvision pillow tqdm
```

(Optional for evaluation)
```bash
pip install pytorch-fid
```

---

## 📥 Dataset
- **Source**: [Anime Face Dataset (Kaggle)](https://www.kaggle.com/splcher/animefacedataset)
- **Steps**:
  1. Download `animefacedataset.zip` from Kaggle.
  2. Extract it into the `animefaces/` folder.
     - Example: `animefaces/0.jpg, animefaces/1.jpg, ...`
  3. Make sure the dataset path looks like this (⚠️ no nested subfolder):
     ```
     animefaces/
       ├── 0.jpg
       ├── 1.jpg
       └── ...
     ```

---

## 🏋️ Training the GAN
Run the following command to start training:

```bash
python gan_anime_faces.py --data_dir ./animefaces --epochs 60 --batch_size 64 --output_dir ./output
```

- `--data_dir` → path to your dataset folder.
- `--epochs` → number of training epochs (default: 50).
- `--batch_size` → how many images per training step.
- `--output_dir` → where models/checkpoints/samples will be saved.

Training will produce:
- Intermediate checkpoints: `checkpoint_epoch_X.pth`
- Final models: `generator_final.pth`, `discriminator_final.pth`
- Generated training samples: `output/samples/`

---

## 🎨 Generating New Anime Faces
After training is complete, you can generate new faces using the trained Generator.

### Example command:
```bash
python gan_anime_faces.py --mode sample --data_dir ./animefaces --gen_path ./output/generator_final.pth --output_dir ./output --sample_size 1
```

- `--mode sample` → tells the script to run in sampling mode.
- `--gen_path` → path to your trained generator model.
- `--sample_size` → number of faces to generate (default: 1).
- Output will be saved in: `output/samples_manual/sample_manual.png`.

👉 This produces **one grid image** containing 1 anime faces.

---

## 📊 Evaluation (Optional)
Two common evaluation metrics:
- **FID (Frechet Inception Distance)** → lower is better.
- **Inception Score (IS)** → higher is better.

Example FID calculation (after installing `pytorch-fid`):
```bash
# Generate 10k images into ./gen and compare with real dataset ./animefaces
pytorch-fid ./animefaces ./gen
```

---

## 🚀 Improvements You Can Try
- Increase image resolution to 128×128.
- Add Spectral Normalization or WGAN-GP for stability.
- Use progressive growing GANs.
- Experiment with different learning rates and optimizers.

---

## ✅ Deliverables
- `gan_anime_faces.py` → training & sampling script.
- `README.md` → documentation (this file).
- `output/` → generated samples and trained models.

---

## ⚠️ Notes
- The **FutureWarning** during sampling (`weights_only=False`) is safe for your own models. To future-proof, you can modify the script to use:
  ```python
  torch.load(args.gen_path, map_location=device, weights_only=True)
  ```
- Always visually inspect generated samples during training to monitor progress.

