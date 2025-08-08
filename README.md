## Proton64_Analysis

Scripts for running the proton64 analysis.

### Analysis 

The script `proton64_analysis.py` does the full analysis on a batch of data. The functionality is controlled by variables defined within the script defined at the top in the parameters section. The variable `data_dir` points to the directory containing the sample events. It expects files named `batch_X.npy` where X is the batch number. The analysis results will be saved in the directory specified by `analysis_dir` using the `run_name` as a prefix for the various output files. 

`single_proton64_analysis.py` is a helper module for running the proton64 analysis on a single event. Mostly used to correct the dE/dx curves. 

### Plotting 

The aptly named `analysis_plotting.py` script is used to visualize the results of the analysis. For now it can plot the results of up to four different samples at once. Modify the variables in the config section `analysis_dir` is the location of the analysis files saved from `proton64_analysis.py` and `plot_dir` is the directory to save the histograms. 
To plot a specific run set: 
- `plot_runX` = True 
- `runX_name` = name of sample (same as `proton64_analysis.py`)  
- `runX_legend` = string to display in the plot legend

The proton dE/dx curves are plotted using `dedx_plotting.py` follows the same conventions. The visualizations often require tweaking, so the code below is messy. 