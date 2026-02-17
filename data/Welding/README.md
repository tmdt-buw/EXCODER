# Welding Dataset
## Source
[Zenodo](https://doi.org/10.5281/zenodo.15101072)

The dataset provides multivariate time series from arc welding processes, focusing on quality prediction. It contains synchronously sampled current and voltage signals at 100 kHz, labeled as either "substandard" (43%) or "satisfactory" (57%) welds.
> Hahn, Y., Maack, R., Tercan, H., Buchholz, G., Purrio, M., Angerhausen, M., Meyes, R., & Meisen, T. (2025). Metal Arc Welding [Data set]. [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15497262.svg)](https://doi.org/10.5281/zenodo.15497262)


### Setup Instructions

1. Download the dataset from the link above
2. Extract and place the dataset files in the `data/Welding` directory
3. Run the preprocessing script once to prepare the data:

```bash
python data/Welding/preprocess_data.py
```

**Note:** The preprocessing step is required before training any models.

