# ASIMOW XAI Paper Checkliste

## Install Python Environment
```bash
uv sync
```


## Datasets
### CNC Machining Data - [Github](https://github.com/boschresearch/CNC_Machining)
The dataset provided is a collection of real-world industrial vibration data collected from a brownfield CNC milling machine. The acceleration has been measured using a tri-axial accelerometer (Bosch CISS Sensor) mounted inside the machine. The X- Y- and Z-axes of the accelerometer have been recorded using a sampling rate equal to 2 kHz. Thereby normal as well as anomoulous data have been collected for 6 different timeframes, each lasting 6 months from October 2018 until August 2021 and labelled accordingly. It can be used to investigate the scalability of models and research process variations as the anomaly impact differs. In total there is data from three different CNC milling machines each executing 15 processes. For a detailed description of the data and experimental set-up, please refer to the paper. 
> Tnani, Mohamed-Ali; Feil, Michael; Diepold, Klaus. Smart Data Collection System for Brownfield CNC Milling Machines: A New Benchmark Dataset for Data-Driven Machine Monitoring. Procedia CIRP2022,107, 131–136.

### Welding - [Zenodo](https://doi.org/10.5281/zenodo.15101072)
The dataset provides multivariate time series from arc welding processes, focusing on quality prediction. It contains synchronously sampled current and voltage signals at 100 kHz, labeled as either "substandard" (43%) or "satisfactory" (57%) welds.
> Hahn, Y., Maack, R., Tercan, H., Buchholz, G., Purrio, M., Angerhausen, M., Meyes, R., & Meisen, T. (2025). Metal Arc Welding [Data set].  [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15101072.svg)](https://doi.org/10.5281/zenodo.15101072)


### ECG - [Kaggle](https://www.kaggle.com/datasets/shayanfazeli/heartbeat/data)

The ECG dataset is a combination of the MIT-BIH Arrhythmia and PTB Diagnostic ECG databases, preprocessed and segmented into individual heartbeats. This univariate dataset classifies heartbeats into five categories, including normal and various arrhythmia types.
> George B Moody and Roger G Mark. 2001. The impact of the MIT-BIH arrhythmia database. IEEE engineering in medicine and biology magazine 20, 3 (2001), 45–50.

> Ralf Bousseljot, Dieter Kreiseler, and Allard Schnabel. 1995. Nutzung der EKG-Signaldatenbank CARDIODAT der PTB über das Internet. (1995).

## Results

## Best Hyperparameter
| model_name         | dataset_name   |   batch_size |   gradient_clip_val |   learning_rate |   res_dropout |   n_head |   d_model |   n_hidden_layers |   dropout_p |   use_layer_norm |   gen_epochs |   finetune_epochs |   prob_unk_token |   epoch_iter |   num_embeddings |   embedding_dim |   hidden_dim |   n_resblocks |   moving_avg |   kernel_size |   individual |   top_k |   d_ff |
|:-------------------|:---------------|-------------:|--------------------:|----------------:|--------------:|---------:|----------:|------------------:|------------:|-----------------:|-------------:|------------------:|-----------------:|-------------:|-----------------:|----------------:|-------------:|--------------:|-------------:|--------------:|-------------:|--------:|-------:|
| DLinear            | CNC_Machining  |         1024 |                0.7  |          0.0001 |        nan    |      nan |       nan |               nan |      nan    |              nan |          nan |               nan |           nan    |          nan |              nan |             nan |          nan |           nan |            9 |             7 |            1 |     nan |    nan |
| DLinear            | ECG            |         1024 |                0.8  |          0.01   |        nan    |      nan |       nan |               nan |      nan    |              nan |          nan |               nan |           nan    |          nan |              nan |             nan |          nan |           nan |            5 |             7 |          nan |     nan |    nan |
| DLinear            | Welding        |         1024 |                0.5  |          0.001  |        nan    |      nan |       nan |               nan |      nan    |              nan |          nan |               nan |           nan    |          nan |              nan |             nan |          nan |           nan |            9 |             9 |          nan |     nan |    nan |
| DVAE               | CNC_Machining  |          512 |                1.5  |          0.001  |        nan    |      nan |       nan |               nan |        0.1  |              nan |          nan |               nan |           nan    |          nan |              512 |             512 |          512 |             8 |          nan |           nan |          nan |     nan |    nan |
| DVAE               | ECG            |          512 |                0.25 |          0.001  |        nan    |      nan |       nan |               nan |        0.1  |              nan |          nan |               nan |           nan    |          nan |              128 |             128 |          512 |             4 |          nan |           nan |          nan |     nan |    nan |
| DVAE               | Welding        |          512 |                0.25 |          0.001  |        nan    |      nan |       nan |               nan |        0.1  |              nan |          nan |               nan |           nan    |          nan |              128 |             128 |          128 |             4 |          nan |           nan |          nan |     nan |    nan |
| DVAE_MLP           | CNC_Machining  |         2048 |                1    |          0.001  |        nan    |      nan |       128 |                 5 |        0    |                0 |          nan |               nan |           nan    |          nan |              nan |             nan |          nan |           nan |          nan |           nan |          nan |     nan |    nan |
| DVAE_MLP           | ECG            |          256 |                1.5  |          0.01   |        nan    |      nan |       256 |                 3 |        0.15 |                1 |          nan |               nan |           nan    |          nan |              nan |             nan |          nan |           nan |          nan |           nan |          nan |     nan |    nan |
| DVAE_MLP           | Welding        |         1024 |                0.8  |          0.001  |        nan    |      nan |        64 |                 4 |        0.15 |                0 |          nan |               nan |           nan    |          nan |              nan |             nan |          nan |           nan |          nan |           nan |          nan |     nan |    nan |
| DVAE_Transformer   | CNC_Machining  |           64 |                0.9  |          0.001  |          0.1  |      nan |      1024 |               nan |      nan    |              nan |           15 |                20 |             0.1  |            2 |              nan |             nan |          nan |           nan |          nan |           nan |          nan |     nan |    nan |
| DVAE_Transformer   | ECG            |          128 |                1.5  |          0.001  |          0.15 |      nan |       128 |               nan |      nan    |              nan |           10 |                30 |             0.2  |            4 |              nan |             nan |          nan |           nan |          nan |           nan |          nan |     nan |    nan |
| DVAE_Transformer   | Welding        |          128 |                1.5  |          0.001  |          0.1  |      nan |       512 |               nan |      nan    |              nan |           15 |                10 |             0.15 |            3 |              nan |             nan |          nan |           nan |          nan |           nan |          nan |     nan |    nan |
| MLP                | CNC_Machining  |         1024 |                1    |          0.0001 |        nan    |      nan |      1024 |                 3 |        0.3  |                1 |          nan |               nan |           nan    |          nan |              nan |             nan |          nan |           nan |          nan |           nan |          nan |     nan |    nan |
| MLP                | ECG            |          512 |                0.9  |          0.001  |        nan    |      nan |       256 |                 3 |        0.15 |                0 |          nan |               nan |           nan    |          nan |              nan |             nan |          nan |           nan |          nan |           nan |          nan |     nan |    nan |
| MLP                | Welding        |         1024 |                1.5  |          0.0001 |        nan    |      nan |      1024 |                 3 |        0.15 |                1 |          nan |               nan |           nan    |          nan |              nan |             nan |          nan |           nan |          nan |           nan |          nan |     nan |    nan |
| TS_Transformer     | CNC_Machining  |           64 |                1.5  |          0.001  |          0.05 |        4 |       512 |               nan |      nan    |              nan |          nan |               nan |           nan    |          nan |              nan |             nan |          nan |           nan |          nan |           nan |          nan |     nan |    nan |
| TS_Transformer     | ECG            |          256 |                1.5  |          0.001  |          0.2  |        8 |      1024 |               nan |      nan    |              nan |          nan |               nan |           nan    |          nan |              nan |             nan |          nan |           nan |          nan |           nan |          nan |     nan |    nan |
| TS_Transformer     | Welding        |           64 |                0.8  |          0.001  |          0.1  |        8 |       512 |               nan |      nan    |              nan |          nan |               nan |           nan    |          nan |              nan |             nan |          nan |           nan |          nan |           nan |          nan |     nan |    nan |
| TimesNet           | CNC_Machining  |          128 |                0.9  |          0.0001 |        nan    |      nan |        64 |                 4 |        0.1  |              nan |          nan |               nan |           nan    |          nan |              nan |             nan |          nan |           nan |          nan |           nan |          nan |       2 |     64 |
| VQ-VAE             | CNC_Machining  |          512 |                0.8  |          0.001  |        nan    |      nan |       nan |               nan |        0.1  |              nan |          nan |               nan |           nan    |          nan |              128 |               2 |          128 |             4 |          nan |           nan |          nan |     nan |    nan |
| VQ-VAE             | ECG            |          512 |                0.5  |          0.001  |        nan    |      nan |       nan |               nan |        0.1  |              nan |          nan |               nan |           nan    |          nan |              256 |              16 |          512 |             8 |          nan |           nan |          nan |     nan |    nan |
| VQ-VAE             | Welding        |          512 |                2    |          0.001  |        nan    |      nan |       nan |               nan |        0.1  |              nan |          nan |               nan |           nan    |          nan |              512 |             256 |          512 |             8 |          nan |           nan |          nan |     nan |    nan |
| VQ-VAE_MLP         | CNC_Machining  |          512 |                0.5  |          0.0001 |        nan    |      nan |      1024 |                 3 |        0    |                0 |          nan |               nan |           nan    |          nan |              nan |             nan |          nan |           nan |          nan |           nan |          nan |     nan |    nan |
| VQ-VAE_MLP         | ECG            |          512 |                0.5  |          0.0001 |        nan    |      nan |      1024 |                 2 |        0.2  |                1 |          nan |               nan |           nan    |          nan |              nan |             nan |          nan |           nan |          nan |           nan |          nan |     nan |    nan |
| VQ-VAE_MLP         | Welding        |         1024 |                0.8  |          0.0001 |        nan    |      nan |        32 |                 3 |        0.1  |                0 |          nan |               nan |           nan    |          nan |              nan |             nan |          nan |           nan |          nan |           nan |          nan |     nan |    nan |
| VQ-VAE_Transformer | CNC_Machining  |           64 |                0.8  |          0.001  |          0.1  |      nan |       512 |               nan |      nan    |              nan |           20 |                40 |             0.15 |            2 |              nan |             nan |          nan |           nan |          nan |           nan |          nan |     nan |    nan |
| VQ-VAE_Transformer | ECG            |           64 |                1.5  |          0.001  |          0.2  |      nan |       512 |               nan |      nan    |              nan |           10 |                30 |             0.15 |            3 |              nan |             nan |          nan |           nan |          nan |           nan |          nan |     nan |    nan |
| VQ-VAE_Transformer | Welding        |          128 |                0.7  |          0.001  |          0.2  |      nan |        64 |               nan |      nan    |              nan |           15 |                20 |             0.05 |            4 |              nan |             nan |          nan |           nan |          nan |           nan |          nan |     nan |    nan |


## Classification Results
### CNC_Machining
| model_name         | f1_score      | accuracy      |
|:-------------------|:--------------|:--------------|
| DLinear            | 0.502 ± 0.000 | 0.957 ± 0.000 |
| DVAE_MLP           | 0.745 ± 0.027 | 0.965 ± 0.004 |
| DVAE_Transformer   | 0.538 ± 0.067 | 0.957 ± 0.001 |
| MLP                | 0.819 ± 0.010 | 0.977 ± 0.001 |
| SAX_MLP            | 0.594 ± 0.021 | 0.949 ± 0.007 |
| TS_Transformer     | 0.489 ± 0.000 | 0.957 ± 0.000 |
| TimesNet           | 0.891 ± 0.010 | 0.984 ± 0.001 |
| VQ-VAE_MLP         | 0.755 ± 0.005 | 0.969 ± 0.001 |
| VQ-VAE_Transformer | 0.701 ± 0.008 | 0.965 ± 0.002 |


### ECG
| model_name         | f1_score      | accuracy      |
|:-------------------|:--------------|:--------------|
| DLinear            | 0.641 ± 0.023 | 0.912 ± 0.003 |
| DVAE_MLP           | 0.775 ± 0.005 | 0.952 ± 0.002 |
| DVAE_Transformer   | 0.517 ± 0.136 | 0.914 ± 0.026 |
| MLP                | 0.913 ± 0.008 | 0.983 ± 0.002 |
| SAX_MLP            | 0.684 ± 0.013 | 0.924 ± 0.002 |
| TS_Transformer     | 0.537 ± 0.037 | 0.914 ± 0.006 |
| TimesNet           | 0.931 ± 0.005 | 0.988 ± 0.001 |
| VQ-VAE_MLP         | 0.816 ± 0.007 | 0.965 ± 0.002 |
| VQ-VAE_Transformer | 0.775 ± 0.023 | 0.956 ± 0.003 |


### Welding
| model_name         | f1_score      | accuracy      |
|:-------------------|:--------------|:--------------|
| DLinear            | 0.731 ± 0.024 | 0.744 ± 0.015 |
| DVAE_MLP           | 0.780 ± 0.001 | 0.786 ± 0.001 |
| DVAE_Transformer   | 0.761 ± 0.004 | 0.765 ± 0.005 |
| MLP                | 0.802 ± 0.002 | 0.807 ± 0.002 |
| SAX_MLP            | 0.749 ± 0.002 | 0.754 ± 0.002 |
| TS_Transformer     | 0.728 ± 0.008 | 0.733 ± 0.012 |
| TimesNet           | 0.822 ± 0.002 | 0.825 ± 0.002 |
| VQ-VAE_MLP         | 0.789 ± 0.001 | 0.794 ± 0.001 |
| VQ-VAE_Transformer | 0.785 ± 0.006 | 0.792 ± 0.003 |


## Results XAI
# Combined results for all datasets

*The table is structured with AUC metrics in the top rows and XAI methods as subcolumns within each model. For each dataset, the highest values of AUC<sub>XAI-RND</sub>, AUC<sub>RND</sub>, and Cosine Similarity (CS) Agreement are highlighted in bold.*

| Dataset | Model Name | SM | IG | RISE | LIME | ATM | RND | CS Agreement |
|---------|------------|----|----|------|------|-----|-----|--------------|
| **CNC** | DLinear | 0.0 ± 0.0 | 0.0 ± 0.0 | -0.01 ± 0.0 | -0.0 ± 0.0 | - | 0.49 ± 0.0 | 0.84 ± 0.0 |
|         | DVAE MLP | 0.03 ± 0.03 | -0.03 ± 0.05 | 0.06 ± 0.03 | 0.03 ± 0.02 | - | 0.54 ± 0.06 | 0.88 ± 0.01 |
|         | DVAE Tr | -0.01 ± 0.01 | 0.0 ± 0.0 | 0.0 ± 0.0 | -0.0 ± 0.0 | -0.01 ± 0.02 | 0.5 ± 0.01 | 0.78 ± 0.01 |
|         | MLP | -0.0 ± 0.01 | 0.03 ± 0.0 | -0.0 ± 0.01 | 0.01 ± 0.01 | - | **0.65 ± 0.0** | 0.85 ± 0.0 |
|         | TS Tr | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.49 ± 0.0 | 0.83 ± 0.01 |
|         | TimesNet | 0.0 ± 0.01 | 0.04 ± 0.01 | 0.01 ± 0.0 | 0.02 ± 0.01 | - | 0.63 ± 0.02 | 0.75 ± 0.02 |
|         | VQ-VAE MLP | **0.08 ± 0.03** | -0.03 ± 0.05 | 0.06 ± 0.01 | -0.02 ± 0.02 | - | 0.55 ± 0.07 | 0.8 ± 0.01 |
|         | SAX MLP | 0.01 ± 0.01 | 0.01 ± 0.01 | 0.01 ± 0.01 | 0.04 ± 0.01 | - | 0.54 ± 0.01 | **0.89 ± 0.01** |
|         | VQ-VAE Tr | -0.05 ± 0.01 | 0.01 ± 0.01 | 0.0 ± 0.01 | -0.02 ± 0.0 | -0.01 ± 0.05 | 0.55 ± 0.01 | 0.78 ± 0.02 |
| **ECG** | DLinear | 0.06 ± 0.01 | 0.03 ± 0.01 | 0.12 ± 0.01 | 0.06 ± 0.0 | - | 0.25 ± 0.01 | 0.69 ± 0.0 |
|         | DVAE MLP | 0.11 ± 0.01 | 0.03 ± 0.03 | 0.14 ± 0.0 | 0.17 ± 0.01 | - | 0.43 ± 0.01 | 0.87 ± 0.01 |
|         | DVAE Tr | 0.06 ± 0.01 | 0.07 ± 0.01 | 0.07 ± 0.02 | 0.08 ± 0.02 | 0.02 ± 0.06 | 0.29 ± 0.03 | 0.73 ± 0.05 |
|         | MLP | 0.19 ± 0.0 | **0.21 ± 0.01** | 0.2 ± 0.01 | 0.12 ± 0.0 | - | 0.43 ± 0.02 | 0.71 ± 0.02 |
|         | TS Tr | 0.09 ± 0.02 | 0.12 ± 0.02 | 0.04 ± 0.03 | 0.01 ± 0.03 | 0.0 ± 0.12 | 0.33 ± 0.01 | 0.7 ± 0.06 |
|         | TimesNet | 0.07 ± 0.01 | 0.09 ± 0.01 | 0.11 ± 0.0 | 0.01 ± 0.01 | - | 0.39 ± 0.01 | 0.69 ± 0.0 |
|         | VQ-VAE MLP | 0.15 ± 0.05 | 0.04 ± 0.07 | 0.12 ± 0.02 | 0.17 ± 0.02 | - | **0.44 ± 0.03** | **0.9 ± 0.02** |
|         | SAX MLP | 0.07 ± 0.01 | 0.02 ± 0.01 | 0.07 ± 0.0 | 0.07 ± 0.01 | - | 0.4 ± 0.01 | 0.87 ± 0.01 |
|         | VQ-VAE Tr | 0.11 ± 0.05 | 0.08 ± 0.01 | 0.14 ± 0.02 | 0.11 ± 0.02 | 0.06 ± 0.04 | 0.36 ± 0.02 | 0.76 ± 0.01 |
| **Welding** | DLinear | 0.12 ± 0.01 | 0.12 ± 0.01 | **0.25 ± 0.01** | 0.02 ± 0.0 | - | 0.52 ± 0.0 | 0.69 ± 0.0 |
|         | DVAE MLP | 0.07 ± 0.03 | 0.01 ± 0.01 | 0.1 ± 0.02 | 0.12 ± 0.03 | - | 0.64 ± 0.03 | 0.87 ± 0.01 |
|         | DVAE Tr | 0.04 ± 0.04 | 0.03 ± 0.01 | 0.11 ± 0.01 | 0.16 ± 0.02 | 0.07 ± 0.04 | **0.65 ± 0.02** | 0.78 ± 0.04 |
|         | MLP | 0.04 ± 0.02 | 0.03 ± 0.02 | 0.22 ± 0.02 | 0.03 ± 0.01 | - | 0.64 ± 0.01 | 0.82 ± 0.0 |
|         | TS Tr | 0.13 ± 0.04 | 0.14 ± 0.03 | 0.03 ± 0.02 | -0.03 ± 0.06 | 0.07 ± 0.05 | 0.57 ± 0.05 | 0.65 ± 0.09 |
|         | TimesNet | 0.05 ± 0.02 | 0.03 ± 0.01 | 0.21 ± 0.01 | -0.01 ± 0.04 | - | 0.59 ± 0.02 | 0.66 ± 0.03 |
|         | VQ-VAE MLP | 0.09 ± 0.06 | 0.05 ± 0.03 | 0.14 ± 0.01 | 0.09 ± 0.02 | - | 0.6 ± 0.03 | 0.87 ± 0.01 |
|         | SAX MLP | 0.02 ± 0.02 | 0.0 ± 0.02 | **0.1 ± 0.01** | 0.1 ± 0.03 | - | 0.63 ± 0.02 | **0.88 ± 0.01** |
|         | VQ-VAE Tr | 0.02 ± 0.03 | 0.03 ± 0.01 | 0.11 ± 0.02 | 0.09 ± 0.01 | 0.05 ± 0.04 | 0.63 ± 0.03 | 0.76 ± 0.01 |
