# Face Generation using GANs

A deep learning project implementing a Generative Adversarial Network (GAN) to generate realistic synthetic faces using the CelebA dataset.

## Project Overview

This project trains a DCGAN (Deep Convolutional GAN) to generate novel, realistic images of human faces. The generator learns to synthesize 64x64 RGB images that fool a discriminator network, while the discriminator learns to distinguish between real and generated images.

### Key Features

- **Custom Dataset Pipeline**: Efficient data loading from the CelebA dataset with image normalization
- **DCGAN Architecture**: Deep convolutional layers for both generator and discriminator
- **Adversarial Training**: Alternating optimization of generator and discriminator networks
- **Training Visualization**: Real-time monitoring of generated images during training
- **Loss Analysis**: Visualization of discriminator and generator losses over time

## Architecture

### Generator
- Takes a latent vector of dimension `[batch_size, 128, 1, 1]`
- Outputs 64x64x3 RGB images through transposed convolutions
- Uses batch normalization and ReLU activations (except final Tanh layer)
- Architecture: 5-layer deconvolutional network with progressive upsampling

### Discriminator
- Takes 64x64x3 RGB images as input
- Outputs a single score (0-1) indicating real vs. fake classification
- Uses convolutional layers with LeakyReLU activations and batch normalization
- Architecture: 5-layer convolutional network with progressive downsampling

## Dataset

**CelebFaces Attributes Dataset (CelebA)**
- 32,600 pre-processed celebrity face images (64x64x3)
- Images are cropped and resized to focus on faces
- Normalized to range [-1, 1] for training stability

[Download CelebA Dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)

## Requirements

```
matplotlib==3.4.3
numpy==1.21.4
Pillow==9.0.0
torch==1.10.0
torchvision==0.11.1
```

### Installation

```bash
pip install -r requirements.txt
```

Ensure you have PyTorch installed with CUDA support for GPU acceleration:
```bash
# For CUDA 11.0
pip install torch==1.10.0 torchvision==0.11.1 torchaudio==0.10.0 -f https://download.pytorch.org/whl/cu110/torch_stable.html
```

## Usage

### Running the Project

Open the Jupyter notebook:
```bash
jupyter notebook face_generation.ipynb
```

The notebook contains the following sections:

1. **Data Pipeline** - Load and preprocess CelebA images
2. **Model Implementation** - Define Generator and Discriminator
3. **Training** - Train both networks with adversarial loss
4. **Visualization** - Display generated faces and training losses

### Key Parameters

```python
latent_dim = 128          # Dimension of latent vector
n_epochs = 20             # Number of training epochs
batch_size = 64           # Batch size for training
learning_rate = 0.0002    # Adam optimizer learning rate
```

## Training Strategy

The training loop implements several best practices:

1. **Alternating Updates**: Discriminator and generator trained sequentially per batch
2. **Label Smoothing**: Real images labeled as 0.9 instead of 1.0 to prevent discriminator overconfidence
3. **Gradient Detachment**: Fake images detached when training discriminator to avoid updating generator twice
4. **Fixed Latent Vector**: Same latent vector used throughout training to visualize generator progression

## Results

The trained model generates recognizable human faces at 64x64 resolution. Training typically shows:
- Discriminator loss converging around 0.5 (balanced performance)
- Generator loss gradually decreasing
- Progressive improvement in face realism over epochs

### Current Limitations
- 64x64 resolution limits fine details (hair texture, iris patterns)
- Training instability common in standard GANs
- Mode collapse potential (generator producing limited face variations)

### Recommended Improvements

1. **Advanced Architectures**
   - Implement ProGAN for progressive high-resolution generation (up to 1024x1024)
   - Use WGAN-GP (Wasserstein GAN with Gradient Penalty) for better training stability
   - Explore StyleGAN for style-based generation

2. **Training Enhancements**
   - Extend training to 50-100 epochs for better convergence
   - Add spectral normalization for discriminator stability
   - Implement learning rate scheduling
   - Use gradient penalty to prevent discriminator from becoming too powerful

3. **Dataset Improvements**
   - Increase dataset size and diversity
   - Augment with face rotation, lighting variations
   - Balance demographic representation

4. **Loss Functions**
   - Replace BCE loss with Wasserstein loss to prevent vanishing gradients
   - Implement hinge loss variants
   - Add perceptual loss for improved realism

## Project Structure

```
.
├── face_generation.ipynb    # Main notebook with complete pipeline
├── requirements.txt         # Python dependencies
├── tests.py                # Unit tests for dataset, discriminator, generator
├── processed_celeba_small/  # CelebA dataset directory
└── README.md               # This file
```

## Implementation Details

### Data Transforms
Images are normalized using ImageNet statistics shifted to [-1, 1]:
```python
Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
```

### Optimizers
Both networks use Adam optimizer:
- Learning rate: 0.0002
- Beta1 (momentum): 0.5
- Beta2 (adaptive learning): 0.999

### Loss Functions
- **Generator**: BCE Loss targeting discriminator to label fakes as real
- **Discriminator**: BCE Loss on real/fake classification with label smoothing

## References

- Radford, A., Metz, L., & Chintala, S. (2015). "Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks" ([arXiv:1511.06434](https://arxiv.org/abs/1511.06434))
- CelebA Dataset: [mmlab.ie.cuhk.edu.hk/projects/CelebA.html](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
- PyTorch Documentation: [pytorch.org](https://pytorch.org)

## Acknowledgments

This project was completed as part of the Udacity Deep Learning Nanodegree Program.

**Note**: GPU acceleration is highly recommended for training. Training on CPU will be significantly slower. Adjust `device = 'cuda'` to `device = 'cpu'` if GPU is not available, but expect ~10-20x slower training.
