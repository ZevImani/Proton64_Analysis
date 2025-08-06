## Proton64_Analysis

Scripts for running the proton64 analysis. This repo is in need of refactoring and the documentation is WiP.

Note: this code is currently optimized for Zev's VS Code workflow.

### Analysis 

The script `proton64_analysis.py` does the full analysis on a batch of data. The functionality is controlled by variables defined within the script defined at the top in the parameters section. The variable `data_dir` points to the directory containing the sample events. It expects files named `batch_X.npy` where X is the batch number. The analysis results will be saved in the directory specified by `save_dir` using the `run_name` as a prefix for the various output files. 

`single_proton64_analysis.py` is a helper module for running the proton64 analysis on a single event. Mostly used to correct the dE/dx curves. 

### Plotting 

The aptly named `analysis_plotting.py` script is used to visualize the results of the analysis. For now it can plot the results of up to four different samples at once. Modify the variables in the config section `dir` is the location of the analysis files (`save_dir` from above) and `plot_dir` is the directory to save the histograms. For each of the desired runs requires `runX_name` to be the name of the run as used in the analysis script (`run_name`). The variable `runX_legend` is the text that will appear in the legend of the plots. For now the only method of specifying which of the four runs to plot is by commenting out appropriate `plt.hist` lines. 

The proton dE/dx curves are plotted using `dedx_plotting.py`. The code is a messier version of the analysis plotting script, so it follows the same conventions. 