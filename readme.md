# FNOpt: Resolution-Agnostic, Self-Supervised Cloth Simulation using Meta-Optimization with Fourier Neural Operators

This is the official repository for [FNOpt: Resolution-Agnostic, Self-Supervised Cloth Simulation using Meta-Optimization with Fourier Neural Operators](https://arxiv.org/abs/2512.05762).


# Abstract
We present FNOpt, a self-supervised cloth simulation framework that formulates time integration as an optimization problem and trains a resolution-agnostic neural optimizer parameterized by a Fourier neural operator (FNO). Prior neural simulators often rely on extensive ground truth data or sacrifice fine-scale detail, and generalize poorly across resolutions and motion patterns. In contrast, FNOpt learns to simulate physically plausible cloth dynamics and achieves stable and accurate rollouts across diverse mesh resolutions and motion patterns without retraining. Trained only on a coarse grid with physics-based losses, FNOpt generalizes to finer resolutions, capturing fine-scale wrinkles and preserving rollout stability. Extensive evaluations on a benchmark cloth simulation dataset demonstrate that FNOpt outperforms prior learning-based approaches in out-of-distribution settings in both accuracy and robustness. These results position FNO-based meta-optimization as a compelling alternative to previous neural simulators for cloth, thus reducing the need for curated data and improving cross-resolution reliability.


# Running the code
## Requirements

- Python 3.9
- PyTorch 2.4.1
- CUDA (recommended for training and fast inference)

## Installation

### 1. Create conda environment and install dependencies

In the PyTorch ecosystem, packages that compile C++/CUDA extensions (like `torch_scatter` and `pytorch3d`) require PyTorch to be installed *first*. Please run these commands sequentially:

```bash
conda create -n fnopt python=3.9 -y
conda activate fnopt

# 1. Install PyTorch base
pip install torch==2.4.1 torchvision==0.19.1

# 2. Install PyTorch extensions
pip install torch_scatter "git+https://github.com/facebookresearch/pytorch3d.git" --no-build-isolation

# 3. Install remaining dependencies
pip install -r requirements.txt
```

### 2. Install the `neuraloperator` package

This project depends on a previous version of the [neuraloperator](https://github.com/neuraloperator/neuraloperator) library, which is included as a submodule in the `neuraloperator/` directory.

```bash
cd neuraloperator
pip install -e .
cd ..
```


## Pretrained Model

The trained FNOpt model is available for download at: **[link]**

After downloading, place the model checkpoint in the `Logger/` directory, preserving the folder structure.


## Training

To train the model:

```bash
python train.py --config_file fno_vertex.yaml --config_name e11
```


## Inference

To run inference:

```bash
python main_clothnet_inference.py --config_file fno_vertex.yaml --config_name handle_e11
```

