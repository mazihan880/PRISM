# Training and Inference Guide for PRISM

This guide demonstrates how to train the model and perform inference using the provided scripts. And the **[Supplementary Material](./supplementar_material.pdf)** is provided in the repository.

---

## Environment Setup

To ensure the environment is ready for training and inference, install the necessary dependencies:

```bash
pip install torch torchvision tqdm numpy pandas scikit-learn
```

### Recommended Setup

1. Use GPUs for faster training (`--device cuda`).
2. Set random seeds for reproducibility.

---

## Model Training

The training process consists of training a news classifier and a guided diffusion-based generator for news recommendation. The pipeline is controlled by `main.py`.

### Configuration

Adjust the training configuration in `args.json`. Key parameters include:

- **Training Settings**:
  - `batch_size`: Batch size during training.
  - `epochs`: Total number of training epochs.
  - `max_len`: Maximum length of user historical sequences.

- **Diffusion Model Settings**:
  - `timesteps`: Number of diffusion steps (e.g., 200).
  - `hidden_dim`: Size of hidden layers.
  - `fusion_mode`: Fusion mode (`clsattn` or `seqattn`).

- **Optimization Settings**:
  - `cls_optimizer` & `diff_optimizer`: Optimizer types (`adam`, `sgd`, etc.).
  - `lr` & `lr_cls`: Learning rates for the diffuser and classifier.

#### Example `args.json`:
```json
{
    "batch_size": 32,
    "epochs": 100,
    "timesteps": 200,
    "max_len": 50,
    "hidden_dim": 128,
    "fusion_mode": "clsattn",
    "cls_optimizer": "adam",
    "lr": 0.001,
    "lr_cls": 0.0005
}
```

### Start Training

Run the following command to start training:

```bash
python main.py --args_file args.json
```

**Training Process**:
1. **Classifier Training** (before `class_epoch`): Optimizes the news authenticity classifier.
2. **Diffusion Generator Training** (after `class_epoch`): Trains the generator using the frozen classifier.

Model weights will automatically be saved to the specified directory (`args.model_dir`).

---

## Model Inference

Run inference on the test set to evaluate the model's performance.

### 1. Pretrained Weights

Ensure the pretrained weights are saved in the directory specified by `--model_dir`:
- Classifier weights: `YourExperiment_classifier.pth`
- Diffusion model weights: `YourExperiment_diffusion_model_bestwith_hr5_XXX_fake_XXX.pth`

### 2. Start Inference

Use `inference.py` to perform inference. Example command:

```bash
python inference.py --args_file args.json --model_dir ./model_dir --random_seed 0
```

**Performance Metrics**:
1. **Accuracy Metrics**:
   - HR@K: Hit Rate for Top-K recommendations.
   - NDCG@K: Normalized Discounted Cumulative Gain for Top-K.
2. **Authenticity Metrics**:
   - RT@K: Ratio of real news in the top-K recommendations.
   - FNSR@K: Suppression ratio for fake news.

---

## Example Outputs

Logs will display the performance of the model on the test set, e.g.:

```
HR@5       NDCG@5     Recall@5    MRR@5   
0.432100   0.321500   0.451200   0.315400 
Real News Top-K Ratio: [0.85, 0.80, 0.75] Real News First Ratio: 0.55 Weighted Fake News Suppression: [0.90, 0.85, 0.80]
```

## Dataset Availability

Due to certain privacy and licensing restrictions, the full dataset used in this work cannot currently be uploaded. However:
- **All code is fully open source**, and we have provided a **sample dataset** for reference.
- **Preprocessing methods** will be included in future updates, so you can recreate the required dataset from raw data.
