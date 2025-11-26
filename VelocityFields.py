###########################To import the particle data###################

import readgadget
import numpy as np
import MAS_library as MASL
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pylab import *
from scipy.ndimage import rotate

# input files
snapshot = Particle data here eg. '/.../Mn_p/snapdir_004/snap_004'
print("snapshot files imported")

#[1](CDM), [2](neutrinos) or [1,2](CDM+neutrinos)
     
ptype_CDM = [1]
ptype_Neutrino = [1,2]

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

f_catalog = Particle data here eg. '/.../halos/1/out_4_pid.list' 

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

Grid = 200 #int(input("Grid size: ")) #density field with have grid^3 voxels
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
delta_pos_Neutrino /= np.mean(delta_pos_Neutrino, dtype=np.float64); delta_pos_Neutrino -= 1
delta_pos_CDM /= np.mean(delta_pos_CDM, dtype=np.float64); delta_pos_CDM -= 1


# Define voxel size
L = BoxSize / Grid

Grid_vel = 10

L_vel = BoxSize / Grid_vel
# VECTOR FIELD #

# cell indices for each particle
index_x = np.floor(pos_Neutrino[:, 0] / L_vel).astype(int)
index_y = np.floor(pos_Neutrino[:, 1] / L_vel).astype(int)
index_z = np.floor(pos_Neutrino[:, 2] / L_vel).astype(int)

index_x_cdm = np.floor(pos_CDM[:, 0] / L_vel).astype(int)
index_y_cdm = np.floor(pos_CDM[:, 1] / L_vel).astype(int)
index_z_cdm = np.floor(pos_CDM[:, 2] / L_vel).astype(int)

# set values inside grid range: 0 to grid_vel-1
index_x = np.clip(index_x, 0, Grid_vel - 1)
index_y = np.clip(index_y, 0, Grid_vel - 1)
index_z = np.clip(index_z, 0, Grid_vel - 1)

index_x_cdm = np.clip(index_x_cdm, 0, Grid_vel - 1)
index_y_cdm = np.clip(index_y_cdm, 0, Grid_vel - 1)
index_z_cdm = np.clip(index_z_cdm, 0, Grid_vel - 1)

# Flatten 3D indices
flat_index = np.ravel_multi_index((index_x, index_y, index_z), (Grid_vel, Grid_vel, Grid_vel))
flat_index_cdm = np.ravel_multi_index((index_x_cdm, index_y_cdm, index_z_cdm),
                                      (Grid_vel, Grid_vel, Grid_vel))

# arrays to store data
sum_vx = np.zeros(Grid_vel**3)
sum_vy = np.zeros(Grid_vel**3)
sum_vz = np.zeros(Grid_vel**3)
count  = np.zeros(Grid_vel**3, dtype=int)

sum_vx_cdm = np.zeros(Grid_vel**3)
sum_vy_cdm = np.zeros(Grid_vel**3)
sum_vz_cdm = np.zeros(Grid_vel**3)
count_cdm  = np.zeros(Grid_vel**3, dtype=int)

# Add each vel component separately to storage arrays
np.add.at(sum_vx, flat_index, vel_Neutrino[:, 0])
np.add.at(sum_vy, flat_index, vel_Neutrino[:, 1])
np.add.at(sum_vz, flat_index, vel_Neutrino[:, 2])
np.add.at(count,  flat_index, 1)

np.add.at(sum_vx_cdm, flat_index_cdm, vel_CDM[:, 0])
np.add.at(sum_vy_cdm, flat_index_cdm, vel_CDM[:, 1])
np.add.at(sum_vz_cdm, flat_index_cdm, vel_CDM[:, 2])
np.add.at(count_cdm,  flat_index_cdm, 1)


# Avoid error
nonzero = count > 0
sum_vx[nonzero] /= count[nonzero]
sum_vy[nonzero] /= count[nonzero]
sum_vz[nonzero] /= count[nonzero]

nonzero_cdm = count_cdm > 0
sum_vx_cdm[nonzero_cdm] /= count_cdm[nonzero_cdm]
sum_vy_cdm[nonzero_cdm] /= count_cdm[nonzero_cdm]
sum_vz_cdm[nonzero_cdm] /= count_cdm[nonzero_cdm]

#Reshape
delta_vel_x = sum_vx.reshape(Grid_vel, Grid_vel, Grid_vel)
delta_vel_y = sum_vy.reshape(Grid_vel, Grid_vel, Grid_vel)
delta_vel_z = sum_vz.reshape(Grid_vel, Grid_vel, Grid_vel)

delta_vel_x_cdm = sum_vx_cdm.reshape(Grid_vel, Grid_vel, Grid_vel)
delta_vel_y_cdm = sum_vy_cdm.reshape(Grid_vel, Grid_vel, Grid_vel)
delta_vel_z_cdm = sum_vz_cdm.reshape(Grid_vel, Grid_vel, Grid_vel)

# combine again
delta_vel = np.stack((delta_vel_x, delta_vel_y, delta_vel_z), axis=-1)
delta_vel_cdm = np.stack((delta_vel_x_cdm, delta_vel_y_cdm, delta_vel_z_cdm), axis=-1)

# print some info
print("Non-empty voxels:", np.count_nonzero(nonzero))
print("Mean |v|:", np.mean(np.linalg.norm(delta_vel, axis=-1)))


#################################Draw Density Fields#####################################

# prepare stacked arrays
stacked_n_pos = np.zeros((Grid, Grid))
stacked_cdm_pos = np.zeros((Grid, Grid))
stacked_n_vel = np.zeros((Grid_vel, Grid_vel, 2))
stacked_cdm_vel = np.zeros((Grid_vel, Grid_vel, 2))
stacked_h_pos_x = []
stacked_h_pos_y = []
num = 0
num_vel = 0

# Number of slices
print(f"Voxel size in Neutrinos is {L}Mpc/h")

L = L.astype(int)
L_vel = (BoxSize/Grid_vel).astype(int)

# Loop through each 'slice' of the box, the slice will be one voxel thick.  Each iteration will
# include the slice along each dimension-direction and then within one j-iteration a loop is formed for
# each dimension.  Within this inner loop the halos are then centred and stacked into a single point.

for a in range(Grid_vel):
    
    mean_vel_Neutrino_1 = np.mean(delta_vel[a:a+L_vel,:,:],axis=0)
    mean_vel_Neutrino_2 = np.mean(delta_vel[:,a:a+L_vel,:],axis=1)
    mean_vel_Neutrino_3 = np.mean(delta_vel[:,:,a:a+L_vel],axis=2)
    
    mean_vel_cdm_1 = np.mean(delta_vel_cdm[a:a+L_vel,:,:],axis=0)
    mean_vel_cdm_2 = np.mean(delta_vel_cdm[:,a:a+L_vel,:],axis=1)
    mean_vel_cdm_3 = np.mean(delta_vel_cdm[:,:,a:a+L_vel],axis=2)
    
    # Halo slice
    l_min = a*L_vel
    l_max = (a+1)*L_vel
    
    indexes_halo_1 = np.where((halo_pos[:,0] >= l_min) & (halo_pos[:,0] <= l_max))
    pos_slide_halo_1 = halo_pos[indexes_halo_1]
    vel_slide_halo_1 = halo_vel[indexes_halo_1]
    
    indexes_halo_2 = np.where((halo_pos[:,1] >= l_min) & (halo_pos[:,1] <= l_max))
    pos_slide_halo_2 = halo_pos[indexes_halo_2]
    vel_slide_halo_2 = halo_vel[indexes_halo_2]
    
    indexes_halo_3 = np.where((halo_pos[:,2] >= l_min) & (halo_pos[:,2] <= l_max))
    pos_slide_halo_3 = halo_pos[indexes_halo_3]
    vel_slide_halo_3 = halo_vel[indexes_halo_3]
    
    # Get the angle between the halo velocity to the positive x-axis
    angles_1 = np.degrees(np.arctan2(vel_slide_halo_1[:,1], vel_slide_halo_1[:,2]))
    angles_1 = angles_1 % 360 # To wrap the angle back around the clockwise direction
    
    angles_2 = np.degrees(np.arctan2(vel_slide_halo_2[:,0], vel_slide_halo_2[:,2]))
    angles_2 = angles_2 % 360
    
    angles_3 = np.degrees(np.arctan2(vel_slide_halo_3[:,0], vel_slide_halo_3[:,1]))
    angles_3 = angles_3 % 360
    
    for b, (hx, hy, hz) in enumerate(pos_slide_halo_1):
        num_vel+=1
        cx = int(np.floor(hy/L_vel)%Grid_vel)
        cy = int(np.floor(hz/L_vel)%Grid_vel)
        
        shifted_neutrino_vel = np.roll(np.roll(mean_vel_Neutrino_1, shift=-cx+Grid_vel//2, axis=0),
                                       shift=-cy+Grid//2, axis=1)
        rotated_neutrino_vel = rotate(shifted_neutrino_vel[:, :, [1,2]], angles_1[b], axes=(1,0), reshape=False)
        
        shifted_cdm_vel = np.roll(np.roll(mean_vel_cdm_1, shift=-cx+Grid_vel//2, axis=0),
                                       shift=-cy+Grid//2, axis=1)
        rotated_cdm_vel = rotate(shifted_cdm_vel[:, :, [1,2]], angles_1[b], axes=(1,0), reshape=False)
        
        stacked_n_vel += rotated_neutrino_vel
        stacked_cdm_vel += rotated_cdm_vel
        
        print(f"Stacked vel {num_vel} (b)", end='\r')
        
    print('')
        
    for c, (hx, hy, hz) in enumerate(pos_slide_halo_2):
        num_vel+=1
        cx = int(np.floor(hx/L_vel)%Grid_vel)
        cy = int(np.floor(hz/L_vel)%Grid_vel)
        
        shifted_neutrino_vel = np.roll(np.roll(mean_vel_Neutrino_2, shift=-cx+Grid_vel//2, axis=0),
                                       shift=-cy+Grid//2, axis=1)
        rotated_neutrino_vel = rotate(shifted_neutrino_vel[:, :, [0,2]], angles_2[c], axes=(1,0), reshape=False)
        
        shifted_cdm_vel = np.roll(np.roll(mean_vel_cdm_2, shift=-cx+Grid_vel//2, axis=0),
                                       shift=-cy+Grid//2, axis=1)
        rotated_cdm_vel = rotate(shifted_cdm_vel[:, :, [0,2]], angles_2[c], axes=(1,0), reshape=False)
        
        stacked_n_vel += rotated_neutrino_vel
        stacked_cdm_vel += rotated_cdm_vel
        
        print(f"Stacked vel {num_vel} (c)", end='\r')
    
    print('')
    for d, (hx, hy, hz) in enumerate(pos_slide_halo_3):
        num_vel+=1
        cx = int(np.floor(hx/L_vel)%Grid_vel)
        cy = int(np.floor(hy/L_vel)%Grid_vel)
        
        shifted_neutrino_vel = np.roll(np.roll(mean_vel_Neutrino_3, shift=-cx+Grid_vel//2, axis=0),
                                       shift=-cy+Grid//2, axis=1)
        rotated_neutrino_vel = rotate(shifted_neutrino_vel[:, :, [0,1]], angles_3[d], axes=(1,0), reshape=False)
        
        shifted_cdm_vel = np.roll(np.roll(mean_vel_cdm_3, shift=-cx+Grid_vel//2, axis=0),
                                       shift=-cy+Grid//2, axis=1)
        rotated_cdm_vel = rotate(shifted_cdm_vel[:, :, [0,1]], angles_3[d], axes=(1,0), reshape=False)
        
        stacked_n_vel += rotated_neutrino_vel
        stacked_cdm_vel += rotated_cdm_vel
        
        print(f"Stacked vel {num_vel} (d)", end='\r')
        
    print('')
    
for j in range(Grid):
    #Take the first 5 components along the first axis and compute the mean value
    
    # Along X axis
    mean_density_Neutrino_1 = np.mean(delta_pos_Neutrino[j:j+L,:,:],axis=0)
    #mean_vel_Neutrino_1 = np.mean(delta_vel[j:j+L:,:,:],axis=0)
    
    mean_density_CDM_1 = np.mean(delta_pos_CDM[j:j+L,:,:],axis=0)
    
    # Along Y axis
    mean_density_Neutrino_2 = np.mean(delta_pos_Neutrino[:,j:j+L,:],axis=1)
    #mean_vel_Neutrino_2 = np.mean(delta_vel[:,j:j+L,:],axis=1)
    
    mean_density_CDM_2 = np.mean(delta_pos_CDM[:,j:j+L,:],axis=1)
    
    # Along Z axis
    mean_density_Neutrino_3 = np.mean(delta_pos_Neutrino[:,:,j:j+L],axis=2)
    #mean_vel_Neutrino_3 = np.mean(delta_vel[:,:,j:j+L],axis=2)
    
    mean_density_CDM_3 = np.mean(delta_pos_CDM[:,:,j:j+L],axis=2)


    # Halo slice
    l_min = j*L
    l_max = (j+1)*L
    
    indexes_halo_1 = np.where((halo_pos[:,0] >= l_min) & (halo_pos[:,0] <= l_max))
    pos_slide_halo_1 = halo_pos[indexes_halo_1]
    vel_slide_halo_1 = halo_vel[indexes_halo_1]
    
    indexes_halo_2 = np.where((halo_pos[:,1] >= l_min) & (halo_pos[:,1] <= l_max))
    pos_slide_halo_2 = halo_pos[indexes_halo_2]
    vel_slide_halo_2 = halo_vel[indexes_halo_2]
    
    indexes_halo_3 = np.where((halo_pos[:,2] >= l_min) & (halo_pos[:,2] <= l_max))
    pos_slide_halo_3 = halo_pos[indexes_halo_3]
    vel_slide_halo_3 = halo_vel[indexes_halo_3]
    
    # Get the angle between the halo velocity to the positive x-axis
    angles_1 = np.degrees(np.arctan2(vel_slide_halo_1[:,1], vel_slide_halo_1[:,2]))
    angles_1 = angles_1 % 360 # To wrap the angle back around the clockwise direction
    
    angles_2 = np.degrees(np.arctan2(vel_slide_halo_2[:,0], vel_slide_halo_2[:,2]))
    angles_2 = angles_2 % 360
    
    angles_3 = np.degrees(np.arctan2(vel_slide_halo_3[:,0], vel_slide_halo_3[:,1]))
    angles_3 = angles_3 % 360

    #area_1 = halo_mass[indexes_halo_1]/5e13
    #area_2 = halo_mass[indexes_halo_2]/5e13
    #area_3 = halo_mass[indexes_halo_3]/5e13


    # Loop over every halo in the slide, extracting its position as a reference
    # ALONG X-AXIS
    for i, (hx, hy, hz) in enumerate(pos_slide_halo_1):
        num += 1
        # change to voxel co-ords
        cx = int(np.floor(hy/L)%Grid)
        cy = int(np.floor(hz/L)%Grid)
        
        # density and velocity field shift of neutrinos to centre halo
        shifted_density_field_n = np.roll(np.roll(mean_density_Neutrino_1.T, shift=-cx+Grid//2, axis=0), 
                                        shift=-cy+Grid//2, axis=1)
        #shifted_neutrino_vel = np.roll(np.roll(mean_vel_Neutrino_1, shift=-cx+Grid//2, axis=0),
        #                               shift=-cy+Grid//2, axis=1)
        
        shifted_density_field_cdm = np.roll(np.roll(mean_density_CDM_1.T, shift=-cx+Grid//2, axis=0),
                                            shift=-cy+Grid//2, axis=1)
        
        
        # rotate the neutrino density and velocity fields with rotate function from scipy
        shifted_density_field_n = rotate(shifted_density_field_n, angles_1[i], axes=(1,0), reshape=False)
        shifted_density_field_cdm = rotate(shifted_density_field_cdm, angles_1[i], axes=(1,0), reshape=False)
        
        #rotated_neutrino_vel = rotate(shifted_neutrino_vel[:, :, [1,2]], angles_1[i], axes=(1,0), reshape=False)
        
        
        # Form a rotation matrix
        angle_rad = np.deg2rad(angles_1[i])

        R = np.array([
    [ np.cos(angle_rad), -np.sin(angle_rad)],
    [ np.sin(angle_rad),  np.cos(angle_rad)]
    ]) 
        
        #vel_yz_n = shifted_neutrino_vel[:, :, 1:3]
        
        #rotated_vel_yz_n = vel_yz_n @ R
        
        stacked_n_pos += shifted_density_field_n
        #stacked_n_vel += rotated_neutrino_vel
        
        stacked_cdm_pos += shifted_density_field_cdm
        
        # Extract Y,Z positions relative to the current halo center
        pos_yz_1 = pos_slide_halo_1[:, [1,2]] - np.array([hy, hz])
        
        # Apply rotation in Y–Z plane
        pos_rotated_yz_1 = pos_yz_1 @ R
        
        # Reinsert X coordinate unchanged
        h_pos_rotate_1 = np.copy(pos_slide_halo_1)
        h_pos_rotate_1[:, [1,2]] = pos_rotated_yz_1 + np.array([hy, hz])        
        
        # shift halo position such that it is in the centre and halos that are moved off
        # one edge are located appropriately on the other side
        hy_shift_1 = (h_pos_rotate_1[:,1]-hy+BoxSize/2)%BoxSize
        hz_shift_1 = (h_pos_rotate_1[:,2]-hz+BoxSize/2)%BoxSize
        hy_shift_1 = hy_shift_1//L
        hz_shift_1 = hz_shift_1//L
        
        stacked_h_pos_x.append(hy_shift_1[i])
        stacked_h_pos_y.append(hz_shift_1[i])
    
        print(f"Stacked image {num} (i)", end='\r')
        
    print('')
    
    # ALONG Y-AXIS
    for k, (hx, hy, hz) in enumerate(pos_slide_halo_2):
        num += 1
        # change to voxel co-ords
        cx = int(np.floor(hx/L)%Grid)
        cy = int(np.floor(hz/L)%Grid)
        
        # density and velocity field shift of neutrinos to centre halo
        shifted_density_field_n = np.roll(np.roll(mean_density_Neutrino_2.T, shift=-cx+Grid//2, axis=0), 
                                        shift=-cy+Grid//2, axis=1)
        #shifted_neutrino_vel = np.roll(np.roll(mean_vel_Neutrino_2, shift=-cx+Grid//2, axis=0),
        #                               shift=-cy+Grid//2, axis=1)
        
        shifted_density_field_cdm = np.roll(np.roll(mean_density_CDM_2.T, shift=-cx+Grid//2, axis=0), 
                                        shift=-cy+Grid//2, axis=1)
        
        
        # rotate the neutrino density and velocity fields with rotate function from scipy
        shifted_density_field_n = rotate(shifted_density_field_n, angles_2[k], axes=(1,0), reshape=False)
        shifted_density_field_cdm = rotate(shifted_density_field_cdm, angles_2[k], axes=(1,0), reshape=False)

        #rotated_neutrino_vel = rotate(shifted_neutrino_vel[:, :, [0,2]], angles_2[k], axes=(1,0), reshape=False)
        

        # Form a rotation matrix
        angle_rad = np.deg2rad(angles_2[k])

        R = np.array([
    [ np.cos(angle_rad), -np.sin(angle_rad)],
    [ np.sin(angle_rad),  np.cos(angle_rad)]
    ]) 
        
        #vel_yz_n = shifted_neutrino_vel[:, :, [0,2]]
        
        #rotated_vel_yz_n = vel_yz_n @ R
        
        stacked_n_pos += shifted_density_field_n
        #stacked_n_vel += rotated_neutrino_vel  
        
        stacked_cdm_pos += shifted_density_field_cdm
        
        # Extract Y,Z positions relative to the current halo center
        pos_xz_2 = pos_slide_halo_2[:, [0,2]] - np.array([hx, hz])
        
        # Apply rotation in Y–Z plane
        pos_rotated_xz_2 = pos_xz_2 @ R
        
        # Reinsert X coordinate unchanged
        h_pos_rotate_2 = np.copy(pos_slide_halo_2)
        h_pos_rotate_2[:, [0,2]] = pos_rotated_xz_2 + np.array([hx, hz])        
        
        # shift halo position such that it is in the centre and halos that are moved off
        # one edge are located appropriately on the other side
        hx_shift_2 = (h_pos_rotate_2[:,0]-hx+BoxSize/2)%BoxSize
        hz_shift_2 = (h_pos_rotate_2[:,2]-hz+BoxSize/2)%BoxSize
        hx_shift_2 = hx_shift_2//L
        hz_shift_2 = hz_shift_2//L
        
        stacked_h_pos_x.append(hx_shift_2[k])
        stacked_h_pos_y.append(hz_shift_2[k])
    
        print(f"Stacked image {num} (k)", end='\r')
        
    print('')
    
    # ALONG Z-AXIS
    for q, (hx, hy, hz) in enumerate(pos_slide_halo_3):
        #print("check")
        num += 1
        # change to voxel co-ords
        cx = int(np.floor(hx/L)%Grid)
        cy = int(np.floor(hy/L)%Grid)
        
        # density and velocity field shift of neutrinos to centre halo
        shifted_density_field_n = np.roll(np.roll(mean_density_Neutrino_3.T, shift=-cx+Grid//2, axis=0), 
                                        shift=-cy+Grid//2, axis=1)
        #shifted_neutrino_vel = np.roll(np.roll(mean_vel_Neutrino_3, shift=-cx+Grid//2, axis=0),
        #                               shift=-cy+Grid//2, axis=1)
        
        
        shifted_density_field_cdm = np.roll(np.roll(mean_density_CDM_3.T, shift=-cx+Grid//2, axis=0), 
                                        shift=-cy+Grid//2, axis=1)
        
        # rotate the neutrino density and velocity fields with rotate function from scipy
        shifted_density_field_n = rotate(shifted_density_field_n, angles_3[q], axes=(1,0), reshape=False)
        shifted_density_field_cdm = rotate(shifted_density_field_cdm, angles_3[q], axes=(1,0), reshape=False)

        #rotated_neutrino_vel = rotate(shifted_neutrino_vel[:, :, [0,1]], angles_3[q], axes=(1,0), reshape=False)

        
        # Form a rotation matrix
        angle_rad = np.deg2rad(angles_3[q])

        R = np.array([
    [ np.cos(angle_rad), -np.sin(angle_rad)],
    [ np.sin(angle_rad),  np.cos(angle_rad)]
    ]) 
        
        #vel_yz_n = shifted_neutrino_vel[:, :, [0,1]]
        
        #rotated_vel_yz_n = vel_yz_n @ R
        
        stacked_n_pos += shifted_density_field_n
        #stacked_n_vel += rotated_neutrino_vel 
        
        stacked_cdm_pos += shifted_density_field_cdm
        
        # Extract Y,Z positions relative to the current halo center
        pos_yz_3 = pos_slide_halo_3[:, [0,1]] - np.array([hx, hy])
        
        # Apply rotation in Y–Z plane
        pos_rotated_yz_3 = pos_yz_3 @ R
        
        # Reinsert X coordinate unchanged
        h_pos_rotate_3 = np.copy(pos_slide_halo_3)
        h_pos_rotate_3[:, [0,1]] = pos_rotated_yz_3 + np.array([hx, hy])
        
        # shift halo position such that it is in the centre and halos that are moved off
        # one edge are located appropriately on the other side
        hy_shift_3 = (h_pos_rotate_3[:,0]-hx+BoxSize/2)%BoxSize
        hz_shift_3 = (h_pos_rotate_3[:,1]-hy+BoxSize/2)%BoxSize
        hy_shift_3 = hy_shift_3//L
        hz_shift_3 = hz_shift_3//L
        
        stacked_h_pos_x.append(hy_shift_3[q])
        stacked_h_pos_y.append(hz_shift_3[q])
    
        print(f"Stacked image {num} (q)", end='\r')
    
    print('')
    

# Compute the average so not just stacked
stacked_n_pos /= num
stacked_cdm_pos /= num


print('')    
print("Finished plotting!")
    

# optimised factor according to chi-squared
f_opt = np.sum(stacked_n_pos * stacked_cdm_pos) / np.sum(stacked_n_pos ** 2)

print("Optimal factor =", f_opt)

dipole_field = (stacked_n_pos * f_opt - stacked_cdm_pos)
dipole_vel = stacked_n_vel - stacked_cdm_vel

x = np.linspace(0, Grid, Grid_vel)
y = np.linspace(0, Grid, Grid_vel)

fig, ax, = plt.subplots(figsize=(8,8))
im= ax.imshow(stacked_n_pos, cmap='plasma', origin='lower')
ax_divider = make_axes_locatable(ax)
cax = ax_divider.append_axes("right", size="5%", pad="2%")
cb = fig.colorbar(im, cax=cax, label='\u03C1 / \u03C1\u0304')
#ax.quiver(x, y, dipole_vel[:,:,0].T,
#          dipole_vel[:,:,1].T, color='black')
ax.scatter(stacked_h_pos_x, stacked_h_pos_y, c='g', s=1)
ax.set_xlabel("Mpc/h"); ax.set_ylabel("Mpc/h")
ax.set_title(f"Neutrino Overdensity Field, voxel size: {L} Mpc/h, Mn_p")
#ax.set_title(f"CDM field * mu, voxel size: {L} Mpc/h, Mn_p")
ticks = np.linspace(0, Grid, 10)  # 5 evenly spaced ticks
tick_labels = [f"{i * L:.1f}" for i in ticks]  # Convert to Mpc/h
ax.set_xticks(ticks); ax.set_xticklabels(tick_labels)
ax.set_yticks(ticks); ax.set_yticklabels(tick_labels)
#ax.set(xlim=(0.4*Grid, 0.6*Grid), ylim=(0.4*Grid, 0.6*Grid))
plt.show()
