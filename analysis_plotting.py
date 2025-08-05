from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import numpy as np

use_single_event = False
if use_single_event: 
	event_name = "sample1.npy"
	single_event = np.load("single_event_samples/"+event_name, allow_pickle=True).item()
	# print(single_event)
	event = single_event['image']
	event_mom = single_event['momentum']

	from single_proton64_analysis import load_reco_model, analyse_event

	reco_model = load_reco_model() 
	length, width, rad_angle, dEdx, reco_mom = analyse_event(reco_model, event)
	single_event_color = 'purple'


dir = "./data_stats/"
plot_dir = "./plots/"

run1_run = "edep_sample1_v3"
run1_lengths = np.load(dir+run1_run+"_lengths.npy") 
run1_widths = np.load(dir+run1_run+"_widths.npy") 
run1_angles = np.load(dir+run1_run+"_angles.npy")
run1_name = "simulation_sample1"
run1_name += " ("+str(len(run1_lengths))+")"

run2_run = "edep_ldm_sample1"
run2_lengths = np.load(dir+run2_run+"_lengths.npy") 
run2_widths = np.load(dir+run2_run+"_widths.npy") 
run2_angles = np.load(dir+run2_run+"_angles.npy")
run2_name = "generated_sample1"
run2_name += " ("+str(len(run2_lengths))+")"

run3 = "edep_sample1_v3"
run3_lengths = np.load(dir+run3+"_lengths.npy") 
run3_widths = np.load(dir+run3+"_widths.npy") 
run3_angles = np.load(dir+run3+"_angles.npy")
run3_name = "Sim_sample1_v3_5x"
run3_name += " ("+str(len(run3_lengths))+")"

run4_dir = "edep_sample1_v2"
run4_lengths = np.load(dir+run4_dir+"_lengths.npy") 
run4_widths = np.load(dir+run4_dir+"_widths.npy") 
run4_angles = np.load(dir+run4_dir+"_angles.npy")
run4_name = "sim_sample1_v2"
run4_name += " ("+str(len(run4_lengths))+")"

n_bins = 50
normalize = True 

y_label = "Number of events"
if normalize: 
	y_label = "Fraction of events"

plt.title("Proton Length")
plt.hist(run1_lengths, n_bins, color='black', label=run1_name, density=normalize, histtype='step', stacked=False, fill=False)
plt.hist(run2_lengths, n_bins, color='red', label=run2_name, density=normalize, histtype='step', stacked=False, fill=False)
# plt.hist(run3_lengths, n_bins, color='blue', label=run3_name, density=normalize, histtype='step', stacked=False, fill=False)
# plt.hist(run4_lengths, n_bins, color='green', label=run4_name, density=normalize, histtype='step', stacked=False, fill=False)
if use_single_event:
	ymin, ymax = plt.ylim()
	plt.hist([length], n_bins, color=single_event_color, label="True Length", density=False, histtype='step', stacked=False, fill=False)
	plt.ylim(ymin, ymax)
plt.xlabel("Length (pixels)")
plt.ylabel(y_label)
plt.legend()
plt.tight_layout()
plt.savefig(plot_dir+"hist_length.png")
plt.clf()

n_bins = 50

# print(np.sum(run3_widths > 15))
# print(max(run3_widths))

plt.title("Proton Width")
plt.hist(run1_widths, n_bins, color='black', label=run1_name, density=normalize, histtype='step', stacked=False, fill=False)
plt.hist(run2_widths, n_bins, color='red', label=run2_name, density=normalize, histtype='step', stacked=False, fill=False)
# plt.hist(run3_widths, n_bins, color='blue', label=run3_name, density=normalize, histtype='step', stacked=False, fill=False)
# plt.hist(run4_widths, n_bins, color='green', label=run4_name, density=normalize, histtype='step', stacked=False, fill=False)
if use_single_event:
	ymin, ymax = plt.ylim()
	plt.hist([width], n_bins, color=single_event_color, label="True Width", density=True, histtype='step', stacked=False, fill=False)
	plt.ylim(ymin, ymax)
plt.xlabel("Width (pixels)")
plt.ylabel(y_label)
plt.xlim(0,10)
plt.legend()
plt.tight_layout()
plt.savefig(plot_dir+"hist_width.png")
plt.clf()

n_bins = 50 

run1_angles = np.rad2deg(run1_angles)
run2_angles = np.rad2deg(run2_angles)
run3_angles = np.rad2deg(run3_angles)
run4_angles = np.rad2deg(run4_angles)


plt.title("Proton Angle")
# plt.hist(run1_angles[run1_angles>50], n_bins, color='black', label=run1_name, density=normalize, histtype='step', stacked=False, fill=False)
# plt.hist(run2_angles[run2_angles>50], n_bins, color='red', label=run2_name, density=normalize, histtype='step', stacked=False, fill=False)

plt.hist(run1_angles, n_bins, color='black', label=run1_name, density=normalize, histtype='step', stacked=False, fill=False)
plt.hist(run2_angles, n_bins, color='red', label=run2_name, density=normalize, histtype='step', stacked=False, fill=False)
# plt.hist(run3_angles, n_bins, color='blue', label=run3_name, density=normalize, histtype='step', stacked=False, fill=False)
# plt.hist(run4_angles, n_bins, color='green', label=run4_name, density=normalize, histtype='step', stacked=False, fill=False)
if use_single_event:
	ymin, ymax = plt.ylim()
	plt.hist([np.rad2deg(rad_angle)], n_bins, color=single_event_color, label="True Angle", density=False, histtype='step', stacked=False, fill=False)
	plt.ylim(ymin, ymax)
plt.xlabel("Angle (degrees) from Horizontal")
plt.ylabel(y_label)
# plt.yscale('log')
plt.legend()
plt.tight_layout()
plt.savefig(plot_dir+"hist_angle.png")
plt.clf()

print("Saved hists")