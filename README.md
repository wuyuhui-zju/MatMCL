# MatMCL: A versatile multimodal learning framework bridging multi-scale knowledge for material design

This is the official implementation of **MatMCL**, a versatile multimodal learning framework bridging multi-scale knowledge for material design.

## System requirements
This package has been tested on the following system:
- Linux: Ubuntu 22.04
- CUDA 12.4

## Environment configuration
Clone this repository:
```
git clone https://github.com/wuyuhui-zju/MatMCL.git
```
 Create the environment with Anaconda in few minutes:
```
cd MatMCL
conda env create
```
Activate the created environment:
```
conda activate matmcl
```

## Project structure
The major folders and their functions in MatMCL are organized as follows:
```
MatMCL/
├── datasets/       # Datasets for SGPT, Mechanical property prediction, cross-modal retrieval and structure generation
├── models/         # Model checkpoints
├── generated/      # Output files of conditional structure generation modulue
├── scripts/        # Scripts for training and application
├── src/            # Source code of MatMCL (model architectures, training pipelines, utils)
├── environment.yml # Conda environment file
├── README.md       # Project introduction and usage guide
```

## Quick start
You can download the trained model checkpoints and datasets from the [link](https://figshare.com/s/0cad763a26f928b70840) for a quick start.
Please place the downloaded `datasets/` and `models/` directories directly under the project root.

Then, navigate to the `scripts/` directory and run the following command to execute the example tasks (including mechanical property prediction and structure generation):
```
python main.py
```
The generated structures will be saved in the `generated/` directory.

## 1. Structure-guided pre-training
We provide a script to pre-train the model:
```
bash train_sgpt.sh
```
- Use `--save` to enable saving model checkpoints.
- Specify checkpoint save path with `--save_path`.

## 2. Mechanical property prediction
After structure-guided pre-training (SGPT), fine-tune the model to predict the mechanical properties of nanofibers:
```
bash train_mech_model.sh
```

Inside the script, you can specify hyperparameters such as:
- `--lr`, `--dropout`, `--weight_decay`: Optimized learning rate, dropout ratio, weight decay.

## 3. Cross-modal retrieval
MatMCL can be used to retrieve either structural images based on processing parameters, 
or retrieve corresponding parameters based on a given structure. 
You can try the following examples:
```
bash retrieve_conditions.sh
bash retrieve_structures.sh
```
Inside the script, you can specify options:
- `--mode`: Select the retrieval mode. Use `retrieve_struct` to retrieve structural images based on input parameters, or `retrieve_cond` to retrieve conditions from a given structure.
- `--params`: A list of input processing parameters (e.g., flow rate, concentration, voltage, etc.). Required when `--mode` is `retrieve_struct`.
- `--filename`: The filename of the input structure image. Required when `--mode` is `retrieve_cond`.

## 4. Conditional structure generation
MatMCL enables the generation of structures conditioned on specific processing parameters by a prior model and a diffusion decoder.
Run the training script as follows:
```
bash train_prior.sh
bash train_decoder.sh
```
- Use `--save` to enable saving, and specify the save path with `--save_path`.

## Citation
If you find MatMCL helpful, please cite our work. (The official paper will be available soon.)
