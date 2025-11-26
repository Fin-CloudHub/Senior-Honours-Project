import readgadget
import numpy as np
import MAS_library as MASL
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pylab import *
from scipy.ndimage import affine_transform

# input files
snapshot = # Particle data directory eg. '/.../Mn_p/0/snapdir_004/snap_004'
print("snapshot files imported")

#[1](CDM), [2](neutrinos) or [1,2](CDM+neutrinos)
     
ptype_CDM = [1]
ptype_Neutrino = [2]

# read header
header   = readgadget.header(snapshot)
BoxSize  = header.boxsize/1e3  #Mpc/h
Nall     = header.nall         #Total number of particles
Masses   = header.massarr*1e10 #Masses of the particles in Msun/h
Omega_m  = header.omega_m      #value of Omega_m
Omega_l  = header.omega_l      #value of Omega_l
h        = header.hubble       #value of h
redshift = header.redshift     #redshift of the snapshot
Hubble   = 100.0*np.sqrt(Omega_m*(1.0+redshift)**3+Omega_l)#Value of H(z) in km/s/(Mpc/h)

# read positions, velocities and IDs of the particles
pos_CDM = readgadget.read_block(snapshot, "POS ", ptype_CDM)/1e3 #positions in Mpc/h
vel_CDM = readgadget.read_block(snapshot, "VEL ", ptype_CDM)     #peculiar velocities in km/s
ids_CDM = readgadget.read_block(snapshot, "ID  ", ptype_CDM)-1

pos_Neutrino = readgadget.read_block(snapshot, "POS ", ptype_Neutrino)/1e3
vel_Neutrino = readgadget.read_block(snapshot, "VEL ", ptype_Neutrino)
ids_Neutrino = readgadget.read_block(snapshot, "ID  ", ptype_Neutrino)-1

print("Positions & velocities acquired")

#############################To get the halo catalog######################################

f_catalog = # Halo data eg. '/.../Mn_p/halos/0/out_4_pid.list'

# read the halo catalog
data = np.loadtxt(f_catalog)
print("Halo data imported")

pid  = data[:,41]
idx = pid == -1  #ids of the halos
data_halo = data[idx]

#Units: Masses in Msun / h
#Units: Positions in Mpc / h (comoving)
#Units: Velocities in km / s (physical, peculiar)

# retrieve halo properties
halo_pos = data_halo[:,8:11]
halo_vel = data_halo[:,11:14]
halo_mass = data_halo[:,2]

print("Halo positions, velocites, & masses acquired")

halo_filter = 1e14 #Solar Masses
 # Filter out halos with less mass than 1e14 solar masses
idx = np.where(halo_mass > halo_filter)
halo_mass = halo_mass[idx]
halo_pos = halo_pos[idx]
halo_vel = halo_vel[idx]

print(f"Halos with mass less than {halo_filter} solar masses removed!")

#############################Build density field######################################

Grid = 1500 #int(input("Grid size: ")) #density field with have grid^3 voxels
MAS = 'CIC' #Mass-assignment scheme:'NGP', 'CIC', 'TSC', 'PCS'
do_RSD = False # Dont do redshift-space distortions
axis = 0 #axis along which place RSD; not used here
verbose = True #whether to print information about progress

#Print some information
print('BoxSize: %.3f Mpc/h'%BoxSize)
print('Redshift: %.3f'%redshift)


# OVERDENSITY FIELD #

# Compute the effective number of particles/mass in each voxel
delta_pos_Neutrino = MASL.density_field_gadget(snapshot, ptype_Neutrino, Grid, MAS, do_RSD, axis, verbose)
delta_pos_CDM = MASL.density_field_gadget(snapshot, ptype_CDM, Grid, MAS, do_RSD, axis, verbose)

# compute the density contrast
delta_pos_Neutrino /= np.mean(delta_pos_Neutrino, dtype=np.float32); delta_pos_Neutrino -= 1
delta_pos_CDM /= np.mean(delta_pos_CDM, dtype=np.float32); delta_pos_CDM -= 1

# Define voxel size
L = BoxSize / Grid

# make cut out and lists for data
cut_out = 25 #Mpc/h
cut_out_array_n = []
cut_out_array_cdm = []
angles_yz = []
angles_xz = []
angles_xy = []

# function for coords to rotate in each plane
def prepare_grid_yz(N):
    coords = np.arange(N)
    Y, Z = np.meshgrid(coords, coords, indexing='ij')

    cy = (N - 1) / 2
    cz = (N - 1) / 2

    Yc = Y - cy
    Zc = Z - cz

    return Yc, Zc, cy, cz

def prepare_grid_xz(N):
    coords = np.arange(N)
    X, Z = np.meshgrid(coords, coords, indexing='ij')

    cx = (N - 1) / 2
    cz = (N - 1) / 2

    Xc = X - cx
    Zc = Z - cz

    return Xc, Zc, cx, cz

def prepare_grid_xy(N):
    coords = np.arange(N)
    X, Y = np.meshgrid(coords, coords, indexing='ij')

    cx = (N - 1) / 2
    cy = (N - 1) / 2

    Xc = X - cx
    Yc = Y - cy

    return Xc, Yc, cx, cy

# loop through each halo position and make cut outs
for i, (hx, hy, hz) in enumerate(halo_pos):

    ix = int(hx / L)
    iy = int(hy / L)
    iz = int(hz / L)
    cut = int(cut_out / L)

    #skip halos near voxel boundaries
    if (ix < cut) or (iy < cut) or (iz < cut):
        continue
    if (ix >= Grid - cut) or (iy >= Grid - cut) or (iz >= Grid - cut):
        continue
    
    # create mask to cut out regions
    cut_n = delta_pos_Neutrino[ix-cut:ix+cut, iy-cut:iy+cut, iz-cut:iz+cut].astype(np.float32)
    cut_cdm = delta_pos_CDM[ix-cut:ix+cut, iy-cut:iy+cut, iz-cut:iz+cut].astype(np.float32)

    # append to lists
    cut_out_array_n.append(cut_n)
    cut_out_array_cdm.append(cut_cdm)

    # calculate rotation angles in each dimension
    angles_yz.append(np.arctan2(halo_vel[:,1][i], halo_vel[:,2][i]))
    angles_xz.append(np.arctan2(halo_vel[:,0][i], halo_vel[:,2][i]))
    angles_xy.append(np.arctan2(halo_vel[:,0][i], halo_vel[:,1][i]))
    print(f"Cut out: {i}", end='\r')

print('')
print('Finished cutting out')

# stack the cut outs into an array, turn angles into an array
cut_out_array_n = np.stack(cut_out_array_n, axis=0).astype(np.float32)
cut_out_array_cdm = np.stack(cut_out_array_cdm, axis=0).astype(np.float32)
angles_yz = np.array(angles_yz)
angles_xz = np.array(angles_xz)
angles_xy = np.array(angles_xy)
num_halos = len(cut_out_array_n)

# Cube size
N = cut_out_array_n[0].shape[0]
Yc, Zc, cy, cz = prepare_grid_yz(N)
Xc, Zc, cx, cz = prepare_grid_xz(N)
Xc, Yc, cx, cy = prepare_grid_xy(N)
print('cell complete')

# create a function to find all the indices to rotate by
def rotation_indices(angles, Ac, Bc, ca, cb):
    H = len(angles)

    c = np.cos(angles)[:, None, None]
    s = np.sin(angles)[:, None, None]

    Ar = Ac * c - Bc * s + ca
    Br = Ac * s + Bc * c + cb

    Ai = np.rint(Ar).astype(int)
    Bi = np.rint(Br).astype(int)

    N = Yc.shape[0]
    Ai = np.clip(Ai, 0, N-1)
    Bi = np.clip(Bi, 0, N-1)

    return Ai, Bi

# rotate the cubes at once
def rotate_cubes(cubes, Ai, Bi):
    H = cubes.shape[0]
    hidx = np.arange(H)[:, None, None]
    return cubes[hidx, :, Ai, Bi]

# create arrays for rotated cut outs
stacked_n_pos_yz = np.zeros((N, N, N), dtype=np.float32)
stacked_n_pos_xz = np.zeros((N, N, N), dtype=np.float32)
stacked_n_pos_xy = np.zeros((N, N, N), dtype=np.float32)
stacked_cdm_pos_yz = np.zeros((N, N, N), dtype=np.float32)
stacked_cdm_pos_xz = np.zeros((N, N, N), dtype=np.float32)
stacked_cdm_pos_xy = np.zeros((N, N, N), dtype=np.float32)
stack_count_yz = 0 # counters
stack_count_xz = 0
stack_count_xy = 0
batch_size = 500 # batch size (to try and tackle the memory limit)

# loop through the cut out in batch_size steps
for start in range(0, num_halos, batch_size):
    end = min(start + batch_size, num_halos)
    
    # stack only this batch
    batch_n = np.stack(cut_out_array_n[start:end], axis=0).astype(np.float32)
    batch_cdm = np.stack(cut_out_array_cdm[start:end], axis=0).astype(np.float32)
    batch_angles_yz = np.array(angles_yz[start:end], dtype=np.float32)
    batch_angles_xz = np.array(angles_xz[start:end], dtype=np.float32)
    batch_angles_xy = np.array(angles_xy[start:end], dtype=np.float32)
    
    # rotate batch
    Yi_batch, Zi_batch = rotation_indices(batch_angles_yz, Yc, Zc, cy, cz)
    rotated_n_yz = rotate_cubes(batch_n, Yi_batch, Zi_batch)
    rotated_cdm_yz = rotate_cubes(batch_cdm, Yi_batch, Zi_batch)

    Xi_batch, Zi_batch = rotation_indices(batch_angles_xz, Xc, Zc, cx, cz)
    rotated_n_xz = rotate_cubes(batch_n, Xi_batch, Zi_batch)
    rotated_cdm_xz = rotate_cubes(batch_cdm, Xi_batch, Zi_batch)

    Xi_batch, Yi_batch = rotation_indices(batch_angles_xy, Xc, Yc, cx, cy)
    rotated_n_xy = rotate_cubes(batch_n, Xi_batch, Yi_batch)
    rotated_cdm_xy = rotate_cubes(batch_cdm, Xi_batch, Yi_batch)
    
    # accumulate into stacked arrays
    stacked_n_pos_yz += np.sum(rotated_n_yz, axis=0)
    stacked_cdm_pos_yz += np.sum(rotated_cdm_yz, axis=0)

    stacked_n_pos_xz += np.sum(rotated_n_xz, axis=0)
    stacked_cdm_pos_xz += np.sum(rotated_cdm_xz, axis=0)

    stacked_n_pos_xy += np.sum(rotated_n_xy, axis=0)
    stacked_cdm_pos_xy += np.sum(rotated_cdm_xy, axis=0)
    
    stack_count_yz += rotated_n_yz.shape[0] # add a count
    stack_count_xz += rotated_n_xz.shape[0]
    stack_count_xy += rotated_n_xy.shape[0]
    
    print(f"Processed batch {start}–{end}", end='\r')

print('')
print('Finished stacking!')

# Compute the average so not just stacked
stacked_n_pos_yz /= stack_count_yz
stacked_cdm_pos_yz /= stack_count_yz

stacked_n_pos_xz /= stack_count_xz
stacked_cdm_pos_xz /= stack_count_xz

stacked_n_pos_xy /= stack_count_xy
stacked_cdm_pos_xy /= stack_count_xy

# find where slice over central halo is
mid = N // 2
low = int(mid * 0.45)
high = int(mid * 0.55)

# take mean of the data in that slice
proj_yz = np.mean(stacked_n_pos_yz[low:high, :, :], axis=0)
proj_xz = np.mean(stacked_n_pos_xz[:, low:high, :], axis=1)
proj_xy = np.mean(stacked_n_pos_xy[:, :, low:high], axis=2)

stacked_n_pos = (proj_yz + proj_xz + proj_xy) / 3

proj_yz = np.mean(stacked_cdm_pos_yz[low:high, :, :], axis=0)
proj_xz = np.mean(stacked_cdm_pos_xz[:, low:high, :], axis=1)
proj_xy = np.mean(stacked_cdm_pos_xy[:, :, low:high], axis=2)

stacked_cdm_pos = (proj_yz + proj_xz + proj_xy) / 3

# optimised factor according to chi-squared
f_opt = np.sum(stacked_n_pos * stacked_cdm_pos) / np.sum(stacked_n_pos ** 2)

print("Optimal factor =", f_opt)

dipole_field = (stacked_n_pos * f_opt) - stacked_cdm_pos

print('cell complete')

#################################Draw Density Fields#####################################

### NEUTRINO ###
fig, ax, = plt.subplots(figsize=(8,8))
im= ax.imshow(stacked_n_pos, cmap='plasma', origin='lower')
ax_divider = make_axes_locatable(ax)
cax = ax_divider.append_axes("right", size="5%", pad="2%")
cb = fig.colorbar(im, cax=cax, label='\u03C1 / \u03C1\u0304')
ax.set_xlabel("Mpc/h"); ax.set_ylabel("Mpc/h")
ax.set_title(f"Neutrino Overdensity Field, voxel size: {L} Mpc/h, Mn_p")

Ny, Nz = stacked_n_pos.shape
ticks = np.linspace(0, Ny-1, 5)
center = (Ny - 1) / 2 
tick_labels = [f"{(t - center) * L:.1f}" for t in ticks]  # Convert to Mpc/h
ax.set_xticks(ticks); ax.set_xticklabels(tick_labels)
ax.set_yticks(ticks); ax.set_yticklabels(tick_labels)
plt.show()


### CDM ###
fig, ax, = plt.subplots(figsize=(8,8))
im= ax.imshow(stacked_cdm_pos, cmap='plasma', origin='lower')
ax_divider = make_axes_locatable(ax)
cax = ax_divider.append_axes("right", size="5%", pad="2%")
cb = fig.colorbar(im, cax=cax, label='\u03C1 / \u03C1\u0304')
ax.set_xlabel("Mpc/h"); ax.set_ylabel("Mpc/h")
ax.set_title(f"CDM Overdensity Field, voxel size: {L} Mpc/h, Mn_p")

Ny, Nz = stacked_n_pos.shape
ticks = np.linspace(0, Ny-1, 5)
center = (Ny - 1) / 2 
tick_labels = [f"{(t - center) * L:.1f}" for t in ticks]  # Convert to Mpc/h
ax.set_xticks(ticks); ax.set_xticklabels(tick_labels)
ax.set_yticks(ticks); ax.set_yticklabels(tick_labels)
plt.show()

### SUBTRACTION ###
vmax = np.percentile(np.abs(dipole_field), 99)
fig, ax, = plt.subplots(figsize=(8,8))
im= ax.imshow(dipole_field, cmap='plasma', origin='lower', vmin=-vmax, vmax=vmax)
ax_divider = make_axes_locatable(ax)
cax = ax_divider.append_axes("right", size="5%", pad="2%")
cb = fig.colorbar(im, cax=cax, label='\u03C1 / \u03C1\u0304')
ax.set_xlabel("Mpc/h"); ax.set_ylabel("Mpc/h")
ax.set_title(f"Dipole Field, voxel size: {L} Mpc/h, Mn_p")

Ny, Nz = stacked_n_pos.shape
ticks = np.linspace(0, Ny-1, 5)
center = (Ny - 1) / 2 
tick_labels = [f"{(t - center) * L:.1f}" for t in ticks]  # Convert to Mpc/h
ax.set_xticks(ticks); ax.set_xticklabels(tick_labels)
ax.set_yticks(ticks); ax.set_yticklabels(tick_labels)
plt.show()
