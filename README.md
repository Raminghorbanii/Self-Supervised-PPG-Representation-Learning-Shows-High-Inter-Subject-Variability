# Self-Supervised PPG Representation Learning Shows High Inter-Subject Variability

## Overview
This repository is the implementation of the study presented in the paper "*Self-Supervised PPG Representation Learning Shows High Inter-Subject Variability.*" This work introduces a Self-Supervised Learning (SSL) framework for PPG (photoplethysmography) signal representation learning to overcome the challenges of label scarcity in PPG data analysis, focusing on a task like activity recognition. It demonstrates how SSL can leverage signal reconstruction as a pretext task to learn informative, generalized PPG representations. However, there is still high inter-subject variability in these learned representations, which makes working with this data more challenging when labeled data is scarce.

- [Read the full paper here](https://dl.acm.org/doi/pdf/10.1145/3589883.3589902)

## Dataset
Our experiments are conducted using the PPG-DaLiA dataset. For access and more information, refer to the [PPG-DaLiA dataset paper](https://www.mdpi.com/1424-8220/19/14/3079).

## Installation Instructions

### Clone the Repository
To get started, clone this repository to your local machine:

```bash
git clone https://github.com/Raminghorbanii/Self-Supervised-PPG-Representation-Learning-Shows-High-Inter-Subject-Variability.git SSL_HISV
cd SSL_HISV
```

### Set Up the Environment
Create and activate a new Conda environment:

```bash
conda create --name SSL_HISV python=3.8
conda activate SSL_HISV
```

### Install Dependencies
Install the required Python packages via:

```bash
pip install -r requirements.txt
```

## Running the Framework

### Pretext Task
Begin with the pretext task. Adjust '**Pretext_config.json**' for your setup, especially adding the '**data_directory**' and '**saved_model_directory**' paths. These directories are already created and available in the main directory. 

```bash
cd Pretext_task
python Pretext_main.py
```

### Downstream Task
Proceed to the downstream task, updating '**Downstream_config.json**' with the correct '**data_directory**' and '**saved_model_directory**' paths, the same as the Pretext task. 


```bash
cd ../Downstream_task
python Downstream_main.py
```

## Citation
If you find this work useful for your research, please consider citing our paper:

```bash
@inproceedings{ghorbani2023self,
  title={Self-supervised ppg representation learning shows high inter-subject variability},
  author={Ghorbani, Ramin and Reinders, Marcel JT and Tax, David MJ},
  booktitle={Proceedings of the 2023 8th International Conference on Machine Learning Technologies},
  pages={127--132},
  year={2023}
}
```


