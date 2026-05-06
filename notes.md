### Links and Resources

Kaggle Link
https://www.kaggle.com/datasets/vipullrathod/fish-market/data

Direct CSV
https://raw.githubusercontent.com/twiradinata/datasets/refs/heads/main/fish-weight.csv

### Feature Analysis
The three length measurements capture different aspects of fish body shape:
- **Length1 vs Length2 vs Length3**: Multiple linear dimensions help the model learn species-specific body proportions
- **Length3** shows highest correlation with Weight (0.923), suggesting it best captures the dimensional growth pattern
- These measurements likely represent different fish orientation captures (vertical, diagonal, horizontal)

**Note**: The exact measurement protocol isn't documented in the source. The labels were chosen by whoever prepared the dataset.

### Data Quality Notes
- No missing values in dataset
- One outlier: Weight=0 for Roach (row 40) - kept in training data