from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import numpy as np


run_single_event = False 
if run_single_event: 
	event_name = "sample1.npy"
	plot_title = "Edep-Sim Sample1"
	single_event = np.load("single_event_samples/"+event_name, allow_pickle=True).item()
	# print(single_event)
	event = single_event['image']
	event_mom = single_event['momentum']

	from single_proton64_analysis import load_reco_model, analyse_event

	reco_model = load_reco_model() 
	length, width, rad_angle, dEdx, reco_mom = analyse_event(reco_model, event)

	# print("Length:", length, "Width:", width, "Angle:", rad_angle)
	# print("dE/dx shape:", dEdx.shape)
	# # print('Reco Mom:', reco_mom*500)
	# x,y,mag = reco_mom
	# z = np.sqrt(np.clip(mag**2-x**2-y**2, a_min=0, a_max=None))
	# print("Reco Mom:", [x*500,y*500,z*500])
	# print("True Mom:", event_mom)
else:
	plot_title = "Validation Dataset"

dir = "./data_stats/"
plot_dir = "./plots/"

run1_run = "edep_ldm_sample1"
run1_lengths = np.load(dir+run1_run+"_lengths.npy") 
run1_dEdx = np.load(dir+run1_run+"_dEdxs.npy")
run1_name = "generated_sample1"
run1_name += " ("+str(len(run1_lengths))+")"

run2_run = "edep_sample1_v3"
run2_lengths = np.load(dir+run2_run+"_lengths.npy") 
run2_dEdx = np.load(dir+run2_run+"_dEdxs.npy")
run2_name = "simulation_sample1"
run2_name += " ("+str(len(run2_lengths))+")"




##### Plotting #####

# plt.title("Proton dE/dx - "+plot_title)
plt.title("Generated Sample1 dE/dx")
energies = run1_dEdx[0,:]
distances = run1_dEdx[1,:]
plt.hist2d(distances, energies, 
		   bins=(32,100), 
		#    bins=(70,100),
		   range=[[0,32], [0,80]], 
		#    range=[[0,50], [0,320]],
		   density=False, 
		#    norm=LogNorm(), 
		   cmin=0)
plt.colorbar()
plt.xlabel("Residual Distance")
plt.ylabel('Energy')

if run_single_event: 
	true_energies = dEdx[0,:]
	true_distances = dEdx[1,:] 

	plt.plot(true_distances, true_energies, 
		  linestyle='-', marker='x', c='w', 
		  markersize=3, alpha=0.8, label="True dE/dx")

	# plt.hist2d(true_distances, true_energies, 
	# 	   bins=(32,100), 
	# 	   range=[[0,32], [0,80]], 
	# 	   density=False, 
	# 	   cmap='Greys', 
	# 	   alpha=0.7,
	# 	   label="True dE/dx",
	# 	   cmin=1)

	plt.legend() 


plt.tight_layout()
plt.savefig(plot_dir+"hist_run1_dEdx.png")
# plt.clf()

print("Saved dE/dx plot to", plot_dir)


# Calculate and plot run2 average bin values as white line
run2_energies = run2_dEdx[0,:]
run2_distances = run2_dEdx[1,:]

run2_energies[run2_energies > 50] = 0  # Ensure no negative energies
run2_distances[run2_energies > 50] = 0  


# Define the same bins as the histogram
distance_bins = np.linspace(0, 32, 33)  # 32 bins from 0 to 32
energy_bins = np.linspace(0, 80, 101)   # 100 bins from 0 to 80

# Calculate 2D histogram for run2 to get bin counts
hist2d_run2, _, _ = np.histogram2d(run2_distances, run2_energies, 
								   bins=[distance_bins, energy_bins])

# Calculate average energy for each distance bin
avg_energies = []
bin_centers = []
min_counts_threshold = 100  # Minimum counts required for reliable average

for i in range(len(distance_bins)-1):
	# Get the center of the distance bin
	bin_center = (distance_bins[i] + distance_bins[i+1]) / 2
	bin_centers.append(bin_center)
	
	# Find the weighted average energy for this distance bin
	energy_centers = (energy_bins[:-1] + energy_bins[1:]) / 2
	bin_counts = hist2d_run2[i, :]
	
	print(np.sum(bin_counts))

	if np.sum(bin_counts) >= min_counts_threshold:
		avg_energy = np.average(energy_centers, weights=bin_counts)
		avg_energies.append(avg_energy)
	else:
		avg_energies.append(np.nan)  # Use NaN for bins with insufficient data

# Convert to arrays for easier handling
bin_centers = np.array(bin_centers)
avg_energies = np.array(avg_energies)

# hack to fix the average line for some reaons (TODO)
print(avg_energies)
print(bin_centers)
# avg_energies[bin_centers == 15.5] = 5.7
# avg_energies[bin_centers > 16] = np.nan


# Plot the average line with markers, skipping NaN values
plt.plot(bin_centers, avg_energies, 
		 color='white', linewidth=2, alpha=0.5, 
		 marker='o', markersize=4, markerfacecolor='white', 
		 markeredgecolor='black', markeredgewidth=0.5,
		 label=f"Simulation Average dE/dx (sample1)")

plt.xlabel("Residual Distance (pixels)")
plt.ylabel('Energy')

if run_single_event: 
	true_energies = dEdx[0,:]
	true_distances = dEdx[1,:] 

	plt.plot(true_distances, true_energies, 
		  linestyle='-', marker='x', c='w', 
		  markersize=3, alpha=0.9, label="True dE/dx")

	# plt.hist2d(true_distances, true_energies, 
	# 	   bins=(32,100), 
	# 	   range=[[0,32], [0,80]], 
	# 	   density=False, 
	# 	   cmap='Greys', 
	# 	   alpha=0.7,
	# 	   label="True dE/dx",
	# 	   cmin=1)

plt.legend() 

plt.tight_layout()
plt.savefig(plot_dir+"hist_run1_dEdx_with_run2_avg.png")
plt.clf()

print("Saved dE/dx plot with run2 average line to", plot_dir)

exit() 