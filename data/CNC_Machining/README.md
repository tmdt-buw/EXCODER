# CNC Machining Dataset Setup Instructions

## Overview
This directory contains setup instructions and preprocessing scripts for the CNC Machining dataset used in our XAI research paper. The dataset provides real-world industrial vibration data collected from brownfield CNC milling machines.

## Source
**Original Repository:** [Github - boschresearch/CNC_Machining](https://github.com/boschresearch/CNC_Machining)

**Research Paper:**
> Tnani, Mohamed-Ali; Feil, Michael; Diepold, Klaus. Smart Data Collection System for Brownfield CNC Milling Machines: A New Benchmark Dataset for Data-Driven Machine Monitoring. Procedia CIRP2022,107, 131–136.

**Paper DOI:** [CIRP CMS](https://doi.org/10.1016/j.procir.2022.04.022)


## Dataset Description
The dataset contains real-world industrial vibration data collected from brownfield CNC milling machines with the following characteristics:

- **Sensor:** Tri-axial accelerometer (Bosch CISS Sensor) mounted inside the machine
- **Axes:** X, Y, and Z-axes recorded
- **Sampling Rate:** 2 kHz
- **Time Periods:** 6 different timeframes, each lasting 6 months from October 2018 to August 2021
- **Machines:** 3 different CNC milling machines
- **Processes:** 15 processes per machine
- **Data Types:** Normal and anomalous data labeled accordingly

## Setup Instructions

### Step 1: Download the Dataset
1. **Clone the original repository:**
   ```bash
   git clone https://github.com/boschresearch/CNC_Machining.git
   ```

2. **Navigate to the cloned repository:**
   ```bash
   cd CNC_Machining
   ```

3. **Copy the data files to your project directory:**
   ```bash
   cp -r CNC_milling_data/* /path/to/your/project/data/CNC_Machining/
   ```
   
   **Alternative: Direct download and extraction**
   If the repository contains compressed data files, extract them directly into `data/CNC_Machining/`.

### Step 2: Verify Directory Structure
After copying the data, your `data/CNC_Machining/` directory should contain:
```
data/CNC_Machining/
├── README.md                 # This file
├── preprocess_ds.py         # Preprocessing script
├── test_idx.npy            # Test indices
├── val_idx.npy             # Validation indices
└── [Raw data directories]   # Your downloaded dataset files (.h5 format)
```

The raw data should be organized in the following structure:
```
Machine_X/
├── TimeFrame_Y/
│   ├── good/
│   │   └── *.h5 files
│   └── bad/
│       └── *.h5 files
```

### Step 3: Run Preprocessing
Execute the preprocessing script to convert and prepare the data:

```bash
cd data/CNC_Machining
python preprocess_ds.py
```

**What the preprocessing does:**
1. **Convert H5 to NumPy:** Converts `.h5` files to `.npy` format for faster loading
2. **Moving Average:** Applies moving average smoothing (window_size=300, step_size=150)
3. **Sequence Normalization:** Scales all sequences to a fixed length of 2000 timesteps
4. **Window Splitting:** Splits each sequence into 10 smaller windows for data augmentation
5. **Label Processing:** Creates binary labels (1 for 'good', 0 for 'bad')

**Output files:**
- `mvg_data_scaled.npy`: Preprocessed vibration data
- `mvg_labels.npy`: Corresponding binary labels

### Step 4: Verify Preprocessing Success
After preprocessing, check that the output files were created:

```bash
ls -la *.npy
```

You should see:
- `mvg_data_scaled.npy` - Processed vibration data
- `mvg_labels.npy` - Binary labels
- `test_idx.npy` - Test set indices (pre-existing)
- `val_idx.npy` - Validation set indices (pre-existing)
