# TB Chest X-ray Classifier

A binary image classifier that detects Tuberculosis from chest X-ray images, built with transfer learning (MobileNetV2) in TensorFlow/Keras.

## Results

| | Precision | Recall |
|---|---|---|
| Baseline (frozen base) | 0.98 | 0.91 |
| Fine-tuned | 1.00 | 0.94 |

Final test set (420 held-out images): 350/350 Normal correctly classified, 66/70 Tuberculosis cases caught, **0 false positives**.

## Dataset

[Tuberculosis (TB) Chest X-ray Database](https://www.kaggle.com/datasets/tawsifurrahman/tuberculosis-tb-chest-xray-dataset) — 3,500 Normal and 700 Tuberculosis chest X-ray images.

Not included in this repo (too large, not mine to redistribute). To run this notebook yourself:

1. Download the dataset from the Kaggle link above.
2. Extract it so the folder structure looks like this:

```
data/
└── TB_Chest_Radiography_Database/
    ├── Normal/
    └── Tuberculosis/
```

## Approach

- **Model:** MobileNetV2 pretrained on ImageNet, used as a frozen feature extractor with a new classification head, then fine-tuned by unfreezing the last 30 layers at a low learning rate.
- **Class imbalance** (4:1 Normal:TB) handled via `class_weight='balanced'` rather than resampling.
- **Data augmentation** (flip/rotation/zoom/contrast) applied to training images only, to reduce overfitting on a relatively small dataset.
- **Classification threshold** chosen by maximizing F1 on the validation set (rather than using the default 0.5), and re-derived after fine-tuning since the model's output probabilities shift.
- **Single train/val/test split** (80/10/10, stratified) built once and reused throughout, with the test set never touched until final evaluation — so the reported numbers reflect genuine held-out performance.


### Citation
This project uses a Tuberculosis Chest X-ray dataset provided by Tawsifur Rahman et al. (2020).
If you use this dataset, please cite:

Tawsifur Rahman, Amith Khandakar, Muhammad A. Kadir, Khandaker R. Islam, 
Khandaker F. Islam, Zaid B. Mahbub, Mohamed Arselene Ayari, 
Muhammad E. H. Chowdhury. (2020). 

"Reliable Tuberculosis Detection using Chest X-ray with Deep Learning, 
Segmentation and Visualization". IEEE Access, Vol. 8, pp. 191586–191601.  
DOI: 10.1109/ACCESS.2020.3031384.

## Usage

1. Install dependencies: `pip install -r requirements.txt`
2. Download and place the dataset as described above.
3. Open `tb_chest_notebook.ipynb` and run all cells top to bottom.

