
###########################General information###################
#000 ——> z=3
#001 ——> z=2
#002 ——> z=1
#003 ——> z=0.5
#004 ——> z=0

###########################To import the particle data###################
import readgadget
import numpy as np
import MAS_library as MASL

# input files
snapshot = '/Users/finlaysime/Desktop/Senior Honour Project/snapdir_004/snap_004'
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
#pos_CDM = readgadget.read_block(snapshot, "POS ", ptype_CDM)/1e3 #positions in Mpc/h
#vel_CDM = readgadget.read_block(snapshot, "VEL ", ptype_CDM)     #peculiar velocities in km/s
#ids_CDM = readgadget.read_block(snapshot, "ID  ", ptype_CDM)-1

pos_Neutrino = readgadget.read_block(snapshot, "POS ", ptype_Neutrino)/1e3
vel_Neutrino = readgadget.read_block(snapshot, "VEL ", ptype_Neutrino)
ids_Neutrino = readgadget.read_block(snapshot, "ID  ", ptype_Neutrino)-1

print("Positions & velocities acquired")

#############################To get the halo catalog######################################
f_catalog = '/Users/finlaysime/Desktop/Senior Honour Project/halos/1/out_4_pid.list' 

# read the halo catalog
data = np.loadtxt(f_catalog)
print("Halo data imported")

pid  = data[:,41]
idx = pid == -1  #ids of the halos
data_halo = data[idx]

#Units: Masses in Msun / h
#Units: Positions in Mpc / h (comoving)
#Units: Velocities in km / s (physical, peculiar)

halo_pos = data_halo[:,8:11]
halo_vel = data_halo[:,11:14]
halo_mass = data_halo[:,2]

print("Halo positions, velocites, & masses acquired")

halo_filter = float(input("Halo mass filter: "))

idx = np.where(halo_mass > halo_filter) # Filter out halos with less mass than 1e14 solar masses
halo_mass = halo_mass[idx]
halo_pos = halo_pos[idx]
halo_vel = halo_vel[idx]

print(f"Halos with mass less than {halo_filter} solar masses removed!")

#############################Build density field######################################
grid_N = int(input("Grid size Neutrinos: ")) #density field with have grid^3 voxels
MAS = 'CIC' #Mass-assignment scheme:'NGP', 'CIC', 'TSC', 'PCS'
do_RSD = False # Dont do redshift-space distortions
axis = 0 #axis along which place RSD; not used here
verbose = True #whether to print information about progress

#Print some information
print('BoxSize: %.3f Mpc/h'%BoxSize)
print('Redshift: %.3f'%redshift)


# Compute the effective number of particles/mass in each voxel
delta_pos_Neutrino = MASL.density_field_gadget(snapshot, ptype_Neutrino, grid_N, MAS, do_RSD, axis, verbose)

# compute the density contrast
delta_pos_Neutrino /= np.mean(delta_pos_Neutrino, dtype=np.float64); delta_pos_Neutrino -= 1


# Define voxel size
L = BoxSize / grid_N

# Compute integer cell indices for each particle
index_x = np.floor(pos_Neutrino[:, 0] / L).astype(int)
index_y = np.floor(pos_Neutrino[:, 1] / L).astype(int)
index_z = np.floor(pos_Neutrino[:, 2] / L).astype(int)

# Clamp values inside grid range [0, grid_N-1]
index_x = np.clip(index_x, 0, grid_N - 1)
index_y = np.clip(index_y, 0, grid_N - 1)
index_z = np.clip(index_z, 0, grid_N - 1)

# Flatten 3D indices to a single linear index for accumulation
flat_index = np.ravel_multi_index((index_x, index_y, index_z), (grid_N, grid_N, grid_N))

# Prepare arrays to hold data
sum_vx = np.zeros(grid_N**3)
sum_vy = np.zeros(grid_N**3)
sum_vz = np.zeros(grid_N**3)
count  = np.zeros(grid_N**3, dtype=int)

# Accumulate each component separately
np.add.at(sum_vx, flat_index, vel_Neutrino[:, 0])
np.add.at(sum_vy, flat_index, vel_Neutrino[:, 1])
np.add.at(sum_vz, flat_index, vel_Neutrino[:, 2])
np.add.at(count,  flat_index, 1)

# Avoid division by zero
nonzero = count > 0
sum_vx[nonzero] /= count[nonzero]
sum_vy[nonzero] /= count[nonzero]
sum_vz[nonzero] /= count[nonzero]

#Reshape into 3D grids
delta_vel_x = sum_vx.reshape(grid_N, grid_N, grid_N)
delta_vel_y = sum_vy.reshape(grid_N, grid_N, grid_N)
delta_vel_z = sum_vz.reshape(grid_N, grid_N, grid_N)

# Combine if you still want a single array [grid_N, grid_N, grid_N, 3]
delta_vel = np.stack((delta_vel_x, delta_vel_y, delta_vel_z), axis=-1)

print("Non-empty voxels:", np.count_nonzero(nonzero))
print("Mean |v|:", np.mean(np.linalg.norm(delta_vel, axis=-1)))


# Number of slices
size_N = BoxSize / grid_N
print(f"Voxel size in Neutrinos is {size_N}Mpc/h")
while True:
    Slice_N = int(input("No. of slices for Neutrinos: "))
    
    if isinstance(Slice_N, int):
        break
    else:
        print("Invalid input, please enter an integer")

slice_thickness_N = Slice_N * size_N
print(f"Neutrino slice thickness: {slice_thickness_N} Mpc/h")

#Take the first 5 components along the first axis and compute the mean value
mean_density_Neutrino = np.mean(delta_pos_Neutrino[:Slice_N,:,:],axis=0)
mean_vel_Neutrino = np.mean(delta_vel[:Slice_N:,:,:],axis=0)

# Neutrino slice
indexes_Neutrino = np.where(pos_Neutrino[:,0] < slice_thickness_N)
pos_slide_Neutrino = pos_Neutrino[indexes_Neutrino]

# Halo slice
indexes_halo = np.where(halo_pos[:,0] < slice_thickness_N)
pos_slide_halo = halo_pos[indexes_halo]
vel_slide_halo = halo_vel[indexes_halo]


area = halo_mass[indexes_halo]/5e13



#################################Draw Density Fields#####################################
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pylab import *
import os

# create directory to save plots to
output_dir = "halo_centered_plots_with_vectors6"
os.makedirs(output_dir, exist_ok=True)

# Change to grid co-ord system
y_grid = np.arange(grid_N)
z_grid = np.arange(grid_N)
Y, Z = np.meshgrid(y_grid, z_grid)

# Loop over every halo in the slide, extracting its position as a reference
for i, (hx, hy, hz) in enumerate(pos_slide_halo):
    
    # change to voxel co-ords
    cx = int(np.floor(hy/L)%grid_N)
    cy = int(np.floor(hz/L)%grid_N)
    
    # density and velocity field shift of neutrinos to centre halo
    shifted_density_field = np.roll(np.roll(mean_density_Neutrino.T, shift=-cx+grid_N//2, axis=0), 
                                            shift=-cy+grid_N//2, axis=1)
    shifted_neutrino_vel = np.roll(np.roll(mean_vel_Neutrino, shift=-cx+grid_N//2, axis=0),
                                   shift=-cy+grid_N//2, axis=1)
    
    # shift halo position such that it is in the centre and halos that are moved off
    # one edge are located appropriately on the other side
    hy_shift = (pos_slide_halo[:,1]-hy+BoxSize/2)%BoxSize
    hz_shift = (pos_slide_halo[:,2]-hz+BoxSize/2)%BoxSize
    hy_shift = hy_shift//L
    hz_shift = hz_shift//L
    # Same procedure for the halo velocities
    hy_vel = vel_slide_halo[:,1] - halo_vel[i,1]
    hz_vel = vel_slide_halo[:,2] - halo_vel[i,2]
    
    fig, ax = plt.subplots(figsize=(8,8))
    im = ax.imshow(shifted_density_field, cmap='plasma', origin='lower')
    ax_divider = make_axes_locatable(ax)
    cax = ax_divider.append_axes("right", size="5%", pad="2%")
    cb = fig.colorbar(im, cax=cax, label='rho/<rho>')
    ax.quiver(np.arange(grid_N), np.arange(grid_N), shifted_neutrino_vel[:,:,1].T,
              shifted_neutrino_vel[:,:,2].T, color='black')
    ax.scatter(hy_shift, hz_shift, s=area, c='g')
    ax.quiver(hy_shift, hz_shift, hy_vel, hz_vel, color='g')
    ax.set_xlabel("Y voxels")
    ax.set_ylabel("Z voxels")
    ax.set_title(f"Centred on halo {i}, Mass: {halo_mass[i]:.2e} Msun/h")
    ax.set(xlim=(0.35*grid_N, 0.65*grid_N), ylim=(0.35*grid_N, 0.65*grid_N))
    
    plt.savefig(f"{output_dir}/halo_vector{i}.png")
    plt.close(fig)
    print(f"Halo {i} centred and plotted", end='\r')

print('')    
print("Finished plotting!")
    
    
    
