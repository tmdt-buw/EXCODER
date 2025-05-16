# ASIMOW XAI Paper Checkliste




## Environment

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
| model_name                | f1_score      | accuracy      |
|:--------------------------|:--------------|:--------------|
| DLinear                   | 0.502 ± 0.000 | 0.957 ± 0.000 |
| MLP                       | 0.819 ± 0.010 | 0.977 ± 0.001 |
| TS_Transformer            | 0.489 ± 0.000 | 0.957 ± 0.000 |
| TimesNet                  | 0.891 ± 0.010 | 0.984 ± 0.001 |
| VQ-VAE_DVAE_MLP           | 0.962 ± 0.002 | 0.973 ± 0.002 |
| VQ-VAE_DVAE_Transformer   | 0.959 ± 0.002 | 0.971 ± 0.002 |
| VQ-VAE_VQ-VAE_MLP         | 0.965 ± 0.001 | 0.975 ± 0.001 |
| VQ-VAE_VQ-VAE_Transformer | 0.957 ± 0.001 | 0.970 ± 0.001 |


### ECG
| model_name                | f1_score      | accuracy      |
|:--------------------------|:--------------|:--------------|
| DLinear                   | 0.641 ± 0.023 | 0.912 ± 0.003 |
| MLP                       | 0.913 ± 0.008 | 0.983 ± 0.002 |
| TS_Transformer            | 0.542 ± nan   | 0.914 ± nan   |
| VQ-VAE_DVAE_MLP           | 0.823 ± 0.009 | 0.966 ± 0.002 |
| VQ-VAE_DVAE_Transformer   | 0.676 ± 0.182 | 0.948 ± 0.026 |
| VQ-VAE_VQ-VAE_MLP         | 0.806 ± 0.015 | 0.965 ± 0.006 |
| VQ-VAE_VQ-VAE_Transformer | 0.787 ± 0.011 | 0.962 ± 0.001 |


### Welding
| model_name                | f1_score      | accuracy      |
|:--------------------------|:--------------|:--------------|
| DLinear                   | 0.731 ± 0.024 | 0.744 ± 0.015 |
| MLP                       | 0.802 ± 0.002 | 0.807 ± 0.002 |
| TS_Transformer            | 0.727 ± 0.008 | 0.730 ± 0.011 |
| VQ-VAE_DVAE_MLP           | 0.774 ± 0.003 | 0.782 ± 0.003 |
| VQ-VAE_DVAE_Transformer   | 0.789 ± 0.003 | 0.795 ± 0.002 |
| VQ-VAE_VQ-VAE_MLP         | 0.793 ± 0.001 | 0.800 ± 0.001 |
| VQ-VAE_VQ-VAE_Transformer | 0.792 ± 0.003 | 0.797 ± 0.005 |


## Results XAI
