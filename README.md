# ISTS-PLM

This is an offical implementation of *Unleashing The Power of Pre-Trained Language Models for Irregularly Sampled Time Series*.


## Setup

   ### Requirements

   Your local system should have the following executables:

   - [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)
   - Python 3.9 or later
   - CUDA Version: 12.2
   - git

   ### Create conda environment

   All instructions below should be executed from a terminal.

   1. clone this repository and run 

   ```bash
cd ISTS-PLM
   ```

   2. create an environment ```ISTS-PLM``` and activate it.

   ```bash
conda create -n ISTS-PLM python=3.9
conda activate ISTS-PLM
   ```

   3. install the required Python modules using file [requirements.txt](requirements.txt).

   ```bash
pip install -r requirement.txt
   ```


**Notices**: After creating the conda environment, you are supposed to find the ```gpt2``` and ```bert``` of ```transformer``` and copy the files in [model_wope](model_wope) to ```gpt2``` and ```bert```, respectively. The path of the ```gpt2``` and ```bert``` may look like this:

```bash
.conda/envs/ISTS-PLM/lib/python3.9/site-packages/transformers/models/gpt2
.conda/envs/ISTS-PLM/lib/python3.9/site-packages/transformers/models/bert
```

## Preparation of dataset

### Datasets for classification

Download the processed datasets:

**(1)** P19 (PhysioNet Sepsis Early Prediction Challenge 2019) https://doi.org/10.6084/m9.figshare.19514338.v1

**(2)** P12 (PhysioNet Mortality Prediction Challenge 2012) https://doi.org/10.6084/m9.figshare.19514341.v1

**(3)** PAM (PAMAP2 Physical Activity Monitoring) https://doi.org/10.6084/m9.figshare.19514347.v1



### Datasets for extrapolation and interpolation

For *Physionet* and *Human Activity*, we have provided the processed datasets for download.

For *MIMIC*, because of the [PhysioNet Credentialed Health Data License](https://physionet.org/content/mimiciii/view-dua/1.4/), you need to apply the processed dataset at [MIMIC-III-Ext-tPatchGNN](https://physionet.org/content/mimic-iii-ext-tpatchgnn/1.0.0/) for experiments replication.

## Run the Model

The scripts for all the experiments are in the [scripts](./scripts). 



### Classification tasks

For example, if you want to reproduce the classification experiment on P12 dataset, please run the following command.

   ```bash
bash scripts/scripts_classification/P12.sh
   ```


   Example:

   ```bash
python classification.py --task {task_name} \
               --log {log_path} \
               --save_path {save_path}
               --seed {seed} \
               --lr 0.001 \
               --batch {batch_size} \
               --model {model_name} \
               --n_te_gptlayer {num of layers for ts} \
               --n_st_gptlayer {num of layers for var} \
               --collate {collate_name}\
               --semi_freeze
   ```

   - ```task```: the classification task name, select from ```[P12, P19, PAM]```.
   - ```seed```: the seed for parameter initialization.
   - ```collate``` the collate function to represent the irregular time series, select from ```[indseq, vector]```.
   - ```n_te_gptlayer```&```n_st_gptlayer```: the number of PLM layers uses for the time series modeling and variable correlation modeling.
   - ```semi_freeze```: whether or not to fine-tune the ```LayerNorm``` of the PLM.



### Extrapolation and interpolation tasks

For example, if you want to reproduce the extrapolation experiment on physionet dataset, please run the following command.

   ```bash
bash scripts/scripts_extrapolation/physionet.sh
   ```


   Example:

   ```bash
python regression.py --dataset {dataset_name} \
		    --task {task_name} \
               --log {log_path} \
               --save_path {save_path}
               --seed {seed} \
               --lr 0.001 \
               --batch {batch_size} \
               --model {model_name} \
               --n_te_gptlayer {num of layers for ts} \
               --n_st_gptlayer {num of layers for var} \
               --collate {collate_name}\
               --semi_freeze
   ```

   - ```dataset```: the regression dataset name, select from ```[physionet, mimic, activity]```.
   - ```task```: the regression task name, select from ```[forecasting, imputation]```.
   - ```seed```: the seed for parameter initialization.
   - ```collate``` the collate function to represent the irregular time series, select from ```[indseq, vector]```.
   - ```n_te_gptlayer```&```n_st_gptlayer```: the number of PLM layers uses for the time series modeling and variable correlation modeling.
   - ```semi_freeze```: whether or not to fine-tune the ```LayerNorm``` of the PLM.


   For more details, please refer to [classification.py](./classification.py) and [regression.py](./regression.py).

## License

The original [MIMIC database](https://mimic.mit.edu/docs/iii/) is hosted and maintained on [PhysioNet](https://physionet.org/about/) under [PhysioNet Credentialed Health Data License 1.5.0](https://physionet.org/content/mimiciii/view-license/1.4/), and is publicly accessible at [here](https://physionet.org/content/mimiciii/1.4/).

Our code in this repository is licensed under the [MIT license](./LICENSE).
