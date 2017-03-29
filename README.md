# Accidents
Estimation of car accidents importance

We have several features about accidents such as: year, day, month, road surface, GPS coordinates, traffic index, etc. We have to estimate wether level of importance for each accident is high or not.

Dataset: Expedientes.mat  
- Rows: samples
- Columns: features

Results: 
- results.p: two pandas Series [Acc test and AUC] for 100 different subsamples.
- Acc: test accuracy
- Acc_zoom: test accuracy zoom
