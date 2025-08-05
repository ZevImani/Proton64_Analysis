from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np 
import glob
import sys


def bresenham_line(x0, y0, x1, y1):
	"""Generate points of a line using Bresenham's algorithm."""
	points = []
	dx = abs(x1 - x0)
	dy = abs(y1 - y0)
	sx = 1 if x0 < x1 else -1
	sy = 1 if y0 < y1 else -1
	err = dx - dy

	while True:
		points.append((x0, y0))
		if x0 == x1 and y0 == y1:
			break
		e2 = 2 * err
		if e2 > -dy:
			err -= dy
			x0 += sx
		if e2 < dx:
			err += dx
			y0 += sy

	return points


zlab = "/n/holystore01/LABS/iaifi_lab/Users/zimani/datasets"

### Configs ###
save_dir = './data_stats/'
plot_events = False
save_stats = True 
plot_batch = 0
###############

### Meta parameters ### 
background_threshold = 5e-2 ## CAREFUL - hand picked
minimum_pixels = 5
minimum_length = 0
#######################

cnt, min_cnt = 0, 0
row,col = 0, 0
fontsize = 20
lengths,widths,angles,dEdxs  = [], [], [], []

use_reco_dedx = True
if use_reco_dedx:
	# Special imports (careful with hardcoded paths)
	import torch
	sys.path.append('/n/home11/zimani/reco_model/')
	from ResNet.ResNet import ResNet50 # slightly modified ResNet50

	# Load model and weights 
	model = ResNet50(num_classes=3, channels=1, norm='batch')
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	model.to(device)
	checkpoint_name = '/n/home11/zimani/reco_model/checkpoints/ResNet50_edep/ResNet50_epoch38.pt'
	model.load_state_dict(torch.load(checkpoint_name, weights_only=True)['model_state_dict'])
	model.eval() 


# data_dir = zlab + "/protons64_div10_v2/val/"
# run_name = "protons64_v2_val"

# data_dir = "/n/home11/zimani/latent-diffusion/one_mom_sample/full_sample1/"
# data_dir = "/n/home11/zimani/make_data/protons64_one_mom_sample1/train/"
# data_dir = "./event_samples/protons64_cond_attn_x_e10/"
# run_name = "full_sample1_thresh10x" 

# data_dir = zlab + "/edep_data/sample1_v3/"
data_dir = "/n/home11/zimani/latent-diffusion/one_mom_sample/edep_ldm_sample1/"

run_name = "edep_ldm_sample1"

# background_threshold *= 5 


data_range = len(glob.glob(data_dir+"*.npy"))
if data_range == 0:
	print("No data found")
	print(data_dir)
	exit()

## Don't iterate momentum files (if they exist)
if len(glob.glob(data_dir+"*mom*.npy")) > 0: 
	data_range = data_range // 2 

for batch_num in tqdm(range(data_range)): 

	try: 
		batch = np.load(data_dir+"batch_"+str(batch_num)+".npy")
	except FileNotFoundError as e:
		continue

	## Pre-processing 
	batch[batch < background_threshold] = 0

	if plot_events and batch_num == plot_batch: 
		rows = 4
		cols = 4
		fig, axes = plt.subplots(rows, cols, figsize=(8, 9))
		# fig.suptitle("SampleA: Length | Width | Angle", fontsize=fontsize) 
		fig.suptitle("Sample1 Threshx10", fontsize=fontsize)

	# Reco Momentum for Batch 
	if use_reco_dedx:
		model_input = torch.tensor(batch).unsqueeze(1).to(device)  # Add batch and channel dimensions
		with torch.no_grad():
			pred = model(model_input)
		reco_mom = pred.squeeze().cpu().numpy() 

	for i, track in enumerate(batch):
	# for i in range(1): 

		# Event counter (starts at 1)
		cnt += 1

		# Get nonzero point coordinates  
		coords = np.column_stack(np.nonzero(track))

		# Require minimum number of pixels in event  
		if len(coords) < minimum_pixels: 
			min_cnt += 1 
			continue 

		# DBScan to keep only the largest cluster 
		labels = DBSCAN(eps=3, min_samples=1).fit_predict(coords) 
		try: 
			unique_labels, counts = np.unique(labels[labels != -1], return_counts=True)
			largest_cluster_label = unique_labels[np.argmax(counts)]
			coords = coords[labels == largest_cluster_label] 
		except: 
			print("Failed clustering on track #"+str(cnt))
			continue 

		# Apply PCA to the coordinates
		pca = PCA(n_components=2)
		try:
			pca.fit(coords)
		except:
			print("Failed PCA on track #"+str(cnt))
			continue 

		# Find distances along PCA axes 
		projected_coords = pca.transform(coords)
		pca_lengths = projected_coords.max(axis=0) - projected_coords.min(axis=0)
		length = pca_lengths[0]
		width = pca_lengths[1]

		# Minimum length cut
		if length < minimum_length: 
			min_cnt += 1 
			continue 

		lengths.append(length)
		widths.append(width)
		

		# Calculate two points that define the PCA line
		axis_line = pca.components_[0] * pca_lengths[0]/2
		x1, y1 = int(pca.mean_[1] - axis_line[1]), int(pca.mean_[0] - axis_line[0])
		x2, y2 = int(pca.mean_[1] + axis_line[1]), int(pca.mean_[0] + axis_line[0])

		# Find start and end 
		dist1 = np.sqrt((32 - x1)**2 + (32 - y1)**2)
		dist2 = np.sqrt((32 - x2)**2 + (32 - y2)**2)
		if dist1 > dist2: 
			_x1, _y1 = x1, y1
			_x2, _y2 = x2, y2
			x1, y1 = _x2, _y2
			x2, y2 = _x1, _y1

		# Get angle from PCA line 
		dy = y2 - y1
		dx = x2 - x1 
		rad_angle = np.arctan2(dy,dx)
		deg_angle = np.rad2deg(rad_angle)
		angles.append(rad_angle)

		# Get the list of points using Bresenham's line algorithm
		line_points = bresenham_line(x1, y1, x2, y2)

		## Find endpoint (farthest from 32,32) 
		distances = np.linalg.norm(line_points - np.array([32,32]), axis=1)
		endpoint = line_points[np.argmax(distances)]
		min_dist = np.min(distances)

		
		## Traverse points and sum around to get dE/dx
		dEs = []
		dxs = []  
		for x,y in line_points:  
			# Edge aware 5x5 box
			y_min, y_max = max(0, y - 2), min(track.shape[0], y + 3)
			x_min, x_max = max(0, x - 2), min(track.shape[1], x + 3)
			dE = np.sum(np.abs(track[y_min:y_max, x_min:x_max])) # absolute value for qt data
			dx = np.sqrt((x-endpoint[0])**2 + (y-endpoint[1])**2) #- min_dist
			dEs.append(dE) 
			dxs.append(dx) 

		# Find projection from reco momentum
		proj_scale = 1.0
		if use_reco_dedx: 
			x,y,mag = reco_mom[i]
			# z = np.sqrt(np.clip(mag**2-x**2-y**2, a_min=0, a_max=None))
			proj_scale = np.sqrt(x**2 + y**2) / mag

		# Save dE and dx with scaling (ugly method)
		dEs = np.array(dEs)
		dxs = np.array(dxs) * proj_scale
		try: 
			energies = np.hstack([energies, dEs])
			dists = np.hstack([dists, dxs])
		except NameError as e: 
			energies = dEs.copy()
			dists = dxs.copy()

		## Plot event with PCAs drawn 
		# plt.clf()
		# plt.imshow(track, cmap='gray')
		# plt.scatter(pca.mean_[1], pca.mean_[0], color='red', label='Mean')
		# for i, (component, length) in enumerate(zip(pca.components_, pca_lengths)):
		# 	axis_line = component * length
		# 	plt.plot([pca.mean_[1] - axis_line[1], pca.mean_[1] + axis_line[1]],
		# 			[pca.mean_[0] - axis_line[0], pca.mean_[0] + axis_line[0]],
		# 			label=f'PCA {i+1}', linewidth=1)
		# 	# print([pca.mean_[1] - axis_line[1], pca.mean_[1] + axis_line[1]],
		# 	# 		[pca.mean_[0] - axis_line[0], pca.mean_[0] + axis_line[0]])
		# plt.title("Track with PCA")
		# plt.legend()
		# plt.axis('off') 
		# plt.savefig('test.png')
		# exit() 
		
		# if plot_events and (batch_num == plot_batch) and length==0:
		if plot_events: #and width >= 2:

			length = str(np.round(length,1))
			width = str(np.round(width,1))
			angle = str(int(deg_angle)*-1)

			if (row >= rows) or (col >= cols): 
				plt.tight_layout()
				plt.savefig("test.png")
				print("Saved test.png")
				exit()
				# continue

			axes[row, col].imshow(track, cmap='gray', interpolation='none')
			axes[row, col].axis('off')
			# axes[row, col].set_title(length+" | "+width+" | "+angle+"$^o$", fontsize=fontsize-4)
			# axes[row, col].set_title(str(batch_mom[i]), fontsize=fontsize-4)

			# PCA Line
			# for x,y in line_points:
			# 	axes[row,col].scatter(x,y, c='red', alpha=0.5, s=1)

			# Hack to plot anywhere within dataset 
			col += 1
			if col == cols:
				col = 0  
				row += 1 
			

# if save_stats and len(lengths) > 100: 
if save_stats:
	print("Saving", len(lengths), "stats to", save_dir+run_name+"_") 
	np.save(save_dir+run_name+"_lengths.npy", lengths)
	np.save(save_dir+run_name+"_widths.npy", widths)
	np.save(save_dir+run_name+"_angles.npy", angles)
	np.save(save_dir+run_name+"_dEdxs.npy", np.vstack([energies, dists]))

if plot_events: 
	plt.tight_layout()
	plt.savefig("test.png")
	print("Saved test.png")

