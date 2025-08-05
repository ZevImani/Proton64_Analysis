from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
import numpy as np 
import torch
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

def load_reco_model(model_path='/n/home11/zimani/reco_model/', model_checkpoint='/n/home11/zimani/reco_model/checkpoints/ResNet50_edep/ResNet50_epoch38.pt'): 
	
	# Special imports (careful with hardcoded paths)
	import torch
	sys.path.append(model_path)
	from ResNet.ResNet import ResNet50 # slightly modified ResNet50

	# Load model and weights 
	model = ResNet50(num_classes=3, channels=1, norm='batch')
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	model.to(device)
	model.load_state_dict(torch.load(model_checkpoint, weights_only=True)['model_state_dict'])
	model.eval() 

	return model 

def analyse_event(reco_model, event):

	background_threshold = 5e-2 ## CAREFUL - hand picked
	minimum_pixels = 5
	minimum_length = 0

	# background_threshold *= 10

	event[event < background_threshold] = 0

	# Get nonzero point coordinates
	coords = np.column_stack(np.nonzero(event))

	# Require minimum number of pixels in event
	if len(coords) < minimum_pixels: 
		print("Track too small")
		return None, None, None, None, None

	# DBScan to keep only the largest cluster 
	labels = DBSCAN(eps=3, min_samples=1).fit_predict(coords) 
	try: 
		unique_labels, counts = np.unique(labels[labels != -1], return_counts=True)
		largest_cluster_label = unique_labels[np.argmax(counts)]
		coords = coords[labels == largest_cluster_label] 
	except: 
		print("Failed clustering")
		return None, None, None, None, None 

	# Apply PCA to the coordinates
	pca = PCA(n_components=2)
	try:
		pca.fit(coords)
	except:
		print("Failed PCA")
		return None, None, None, None, None  

	# Find distances along PCA axes 
	projected_coords = pca.transform(coords)
	pca_lengths = projected_coords.max(axis=0) - projected_coords.min(axis=0)
	length = pca_lengths[0]
	width = pca_lengths[1]

	# Minimum length 
	if length < minimum_length: 
		print("Track too short")
		return None, None, None, None, None

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

	# Get the list of points using Bresenham's line algorithm
	line_points = bresenham_line(x1, y1, x2, y2)

	## Find endpoint (farthest from 32,32) 
	distances = np.linalg.norm(line_points - np.array([32,32]), axis=1)
	endpoint = line_points[np.argmax(distances)]
	
	## Traverse points and sum around to get dE/dx
	dEs = []
	dxs = []  
	for x,y in line_points:  
		# Edge aware 5x5 box
		y_min, y_max = max(0, y - 2), min(event.shape[0], y + 3)
		x_min, x_max = max(0, x - 2), min(event.shape[1], x + 3)
		dE = np.sum(np.abs(event[y_min:y_max, x_min:x_max])) # absolute value for qt data
		dx = np.sqrt((x-endpoint[0])**2 + (y-endpoint[1])**2) 
		dEs.append(dE) 
		dxs.append(dx) 

	# Find projection from reco momentum
	proj_scale = 1.0 
	if reco_model is not None:
		device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		model_input = torch.tensor(event).unsqueeze(0).unsqueeze(0).to(device)  # Add batch and channel dimensions
		with torch.no_grad():
			reco_mom = reco_model(model_input).squeeze().cpu().numpy()  
		x, y, mag = reco_mom
		# z = np.sqrt(np.clip(mag**2-x**2-y**2, a_min=0, a_max=None))
		proj_scale = np.sqrt(x**2 + y**2) / mag
	else: 
		reco_mom = None 

	# Save dE and dx with 3D momentum scaling 
	dEs = np.array(dEs)
	dxs = np.array(dxs) * proj_scale
	dEdx = np.vstack([dEs, dxs])

	return length, width, rad_angle, dEdx, reco_mom


if __name__ == "__main__": 

	zlab = "/n/holystore01/LABS/iaifi_lab/Users/zimani/datasets"
	data_dir = zlab + "/protons64_div10_v2/train/"
	run_name = "protons64_v2_train"


	batch = np.load(data_dir+"batch_0.npy")

	# for event in batch: 
	event = np.load("bad_event.npy")
	print(event.shape)
	reco_model = load_reco_model()
	length, width, rad_angle, dEdx, reco_mom = analyse_event(reco_model, event)
	print("Length:", length, "Width:", width, "Angle:", rad_angle)
	print("dE/dx shape:", dEdx.shape)
	print('Reco Mom:', reco_mom)
	exit() 