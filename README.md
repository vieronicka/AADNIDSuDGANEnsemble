# AADNIDSuDGANEnsemble

**Adversarially Aware Deep Network Intrusion Detection System using Dynamic GAN Ensemble**

## Overview

This repository implements a novel adversarially robust Network Intrusion Detection System (NIDS) that leverages a Dynamic GAN Ensemble approach to defend against adversarial attacks. The system combines multiple Generative Adversarial Network architectures to augment training data and improve model robustness against evasion attacks.

## Key Features

- **Dynamic GAN Ensemble Defense**: Intelligently combines three GAN architectures (Vanilla GAN, WGAN-GP, CGAN) with adaptive weighting based on performance
- **Adversarial Robustness**: Defense against common evasion attacks including FGSM, PGD, DeepFool, and others
- **Baseline Comparison**: Includes Random Forest baseline for performance benchmarking
- **Comprehensive Evaluation**: Uses Adversarial Robustness Toolbox (ART) for attack evaluation
- **Real-world Dataset**: Trained on MachineLearningCVE dataset with focus on Infiltration and Botnet attack detection

## Architecture

### GAN Ensemble Components

1. **Vanilla GAN**
   - Uses Binary Cross-Entropy (BCE) loss
   - Fast training baseline
   - Generates synthetic network flow patterns

2. **WGAN-GP (Wasserstein GAN with Gradient Penalty)**
   - Wasserstein distance as loss function
   - Gradient penalty for stability
   - Avoids mode collapse
   - Best for training stability

3. **CGAN (Conditional GAN)**
   - Label-conditional generation
   - Learns per-class patterns
   - Generates class-specific synthetic samples

### Dynamic Weighting

The ensemble dynamically weights each GAN based on F1 scores:
- 70% weight on Infiltration detection
- 30% weight on Botnet detection
- Adapts weights based on validation performance

## Project Structure

```
├── VivaAADNIDSDGE_v6_(6advgantrainig) (3).ipynb          # Main implementation notebook
├── Another_copy_of_grok_V6_baseline_GANensemble.ipynb    # Baseline experiments
├── Copy_of_Another__for_cumulative_sum_important_copy_of_grok_V6_baseline_GANensemble.ipynb  # Cumulative analysis
└── README.md                                              # This file
```

## Workflow

### 1. Environment Setup
- Import required libraries (TensorFlow, scikit-learn, ART, etc.)
- Configure GPU/CPU settings

### 2. Dataset Loading and Preprocessing
- Load MachineLearningCVE dataset (8 CSV files)
- 79 standard network flow features
- Handle missing values and normalization

### 3. Feature Selection and Encoding
- Chi-squared feature selection
- Label encoding for categorical features
- MinMax/Standard scaling

### 4. Baseline Random Forest Training
- Train baseline RF model
- Evaluate performance metrics
- Establish comparison baseline

### 5. Dynamic GAN Ensemble Defense Training
- Train all three GAN architectures
- Generate synthetic adversarial examples
- Augment training data
- Calculate dynamic weights based on F1 scores

### 6. Adversarial Robustness Evaluation
- Install ART (Adversarial Robustness Toolbox) and GOPATA
- Train DNN surrogate model
- Execute various adversarial attacks:
  - FGSM (Fast Gradient Sign Method)
  - PGD (Projected Gradient Descent)
  - DeepFool
  - C&W (Carlini & Wagner)
- Calculate Attack Success Rate (ASR)
- Generate comprehensive evaluation reports

## Requirements

```python
- Python 3.7+
- TensorFlow 2.x
- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn
- adversarial-robustness-toolbox (ART)
- ctgan
```

## Installation

```bash
# Clone the repository
git clone https://github.com/vieronicka/AADNIDSuDGANEnsemble.git
cd AADNIDSuDGANEnsemble

# Install dependencies
pip install tensorflow scikit-learn pandas numpy matplotlib seaborn
pip install adversarial-robustness-toolbox ctgan
```

## Usage

### Running the Main Notebook

1. Open `VivaAADNIDSDGE_v6_(6advgantrainig) (3).ipynb` in Jupyter Notebook or Google Colab
2. Mount your Google Drive (if using Colab)
3. Update dataset path to your MachineLearningCVE dataset location
4. Run cells sequentially to:
   - Preprocess data
   - Train baseline model
   - Train GAN ensemble
   - Evaluate adversarial robustness

### Dataset Setup

```python
# Update the path to your dataset
path = '/path/to/MachineLearningCVE'
all_files = glob.glob(path + "/*.csv")
dataset = pd.concat((pd.read_csv(f) for f in all_files))
```

## Performance Metrics

The system evaluates models using:
- **Accuracy**: Overall classification accuracy
- **Precision**: Per-class precision
- **Recall**: Per-class recall
- **F1 Score**: Harmonic mean of precision and recall
- **Attack Success Rate (ASR)**: Percentage of successful adversarial attacks
- **Confusion Matrix**: Detailed classification breakdown

## GAN Architecture Details

### Generator
- Input: Random noise vector (latent dimension)
- Output: Synthetic network flow features
- Architecture: Dense layers with LeakyReLU activation

### Discriminator
- Input: Real or fake network flow features
- Output: Probability score (0 = fake, 1 = real)
- Architecture: Dense layers with dropout for regularization

### Training Parameters
- Optimizer: Adam
- Learning Rate: 0.0002 (Vanilla GAN, CGAN), 0.0001 (WGAN-GP)
- Gradient Penalty (WGAN-GP): λ = 10.0
- Batch Size: Configurable
- Epochs: Adaptive based on convergence

## Research Focus

This implementation focuses on:
- **Adversarial Defense**: Improving robustness against evasion attacks
- **Data Augmentation**: Using GANs to generate realistic adversarial training samples
- **Ensemble Learning**: Combining multiple GAN architectures for better generalization
- **Network Security**: Detecting sophisticated intrusion attempts (Infiltration, Botnet)

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{aadnidsudganensemble2025,
  title={Adversarially Aware Deep Network Intrusion Detection System using Dynamic GAN Ensemble},
  author={Your Name},
  year={2025},
  publisher={GitHub},
  url={https://github.com/vieronicka/AADNIDSuDGANEnsemble}
}
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

<!-- This project is licensed under the MIT License - see the LICENSE file for details. -->

## Acknowledgments

- Adversarial Robustness Toolbox (ART) by IBM Research
- MachineLearningCVE Dataset
- CTGAN for tabular data generation
- TensorFlow and scikit-learn communities

## Contact

For questions or collaboration opportunities, please open an issue on GitHub.

---

**Note**: This is a research implementation. For production deployment, additional security hardening and testing are recommended.