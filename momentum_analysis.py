from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np 
import glob
import sys



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

data_dir = zlab + "/protons64_div10_v2/train/"
# run_name = "protons64_v2_val"

# data_dir = "/n/home11/zimani/latent-diffusion/one_mom_sample/full_sample1/"
# data_dir = "/n/home11/zimani/make_data/protons64_one_mom_sample1/train/"
# data_dir = "./event_samples/protons64_cond_attn_x_e10/"
# run_name = "full_sample1_thresh10x" 

# data_dir = zlab + "/edep_data/sample1_v2/"
# run_name = "edep_sample1_v2_thresh10x"

# background_threshold *= 10 

data_dir = zlab + "/edep_protons64/edep_val/"


data_range = len(glob.glob(data_dir+"*.npy"))
if data_range == 0:
	print("No data found")
	print(data_dir)
	exit()

## Don't iterate momentum files (if they exist)
if len(glob.glob(data_dir+"*mom*.npy")) > 0: 
	data_range = data_range // 2 

xs, ys, zy, mags = [], [], [], []
KEs = []

imgs = [] 
moms = [] 

pixel_sums = []

for batch_num in tqdm(range(data_range)): 

	# try: 
	# 	batch = np.load(data_dir+"batch_"+str(batch_num)+".npy")
	# except FileNotFoundError as e:
	# 	continue

	# ## Pre-processing 
	# batch[batch < background_threshold] = 0

	batch_mom = np.load(data_dir+"batch_mom_"+str(batch_num)+".npy")
	batch = np.load(data_dir+"batch_"+str(batch_num)+".npy")

	for event, mom in zip(batch, batch_mom): 
		x,y,z = mom
		mag = np.round(np.sqrt(x**2 + y**2 + z**2),2)
		
		# px,py,pz = 314.0, -126.4, 249.1

		mag = np.sqrt(x**2 + y**2 + z**2)

		mass = 938.272 ## mass of proton 

		totalE = np.sqrt(mag**2 + mass**2)

		KE = totalE - mass 

		E = np.sqrt(x**2 + y**2 + z**2 + mass**2) - mass 

		if E > 300: 
			imgs.append(event)
			moms.append(np.array([mag, E]))
			print(x,y,z, E)

		xs.append(x)
		ys.append(y)
		zy.append(z)
		mags.append(mag)
		KEs.append(E)

		pixel_sums.append(np.sum(event))

	# 	if len(imgs) > 16: 
	# 		break 

	# if len(imgs)>16: 
	# 	break 

print(len(pixel_sums), "Sums", np.min(pixel_sums), np.mean(pixel_sums), np.max(pixel_sums))
exit()

grid_size = 4
fig, axes = plt.subplots(grid_size, grid_size, figsize=(8, 8))
axes = axes.ravel() # Flatten axes array for easy iteration 

for i in range(len(axes)): 
	axes[i].imshow(imgs[i] , cmap='gray')
	axes[i].axis('off')
	axes[i].set_title(str(np.round(moms[i],1)))
	# axes[i].set_title(str(np.round(moms[i],1)))

plt.savefig("test.png")

exit() 


plt.hist(KEs)
plt.savefig("test.png")

KEs = np.array(KEs)
print(np.min(KEs), np.max(KEs))

exit() 

fig, axes = plt.subplots(4, 1, figsize=(8, 8))

# Flatten the axes array for easy iteration
axes = axes.ravel()

data1 = [xs, ys, zy, mags]
labels = ["px", "py", "pz", "p_mag"]

fig.suptitle("PILArNet Proton64 Training Data Momentum")
for i in range(len(axes)): 
	axes[i].hist(data1[i], label=labels[i], histtype='step')
	# axes[i].set_title(f"{mom1[i][0]:.1f}  {mom1[i][1]:.1f}  {mom1[i][2]:.1f}", fontsize=11)
	# axes[i].axis('off')
	axes[i].legend()
plt.tight_layout()
plt.savefig("test.png")

exit() 

## Edep-Sim energy calculation 

px,py,pz = 314.0, -126.4, 249.1

mag = np.sqrt(px**2 + py**2 + pz**2)

mass = 938.272

totalE = np.sqrt(mag**2 + mass**2)

KE = totalE - mass 

E = np.sqrt(px**2 + py**2 + pz**2 + mass**2) - mass 

print(KE)