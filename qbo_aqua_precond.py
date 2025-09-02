import xarray as xr
import numpy as np
from glob import glob
import sys
import pdb
import os

rd = 287.058 # specific gas constant of dry air in J/kg*K
cp = 1.00464e3 # specific heat capacity of dry air at const. pressure in J/kg*K
g = 9.80665 # acceleration of gravity in m/s^2
a = 6.37123e6 # Earth's radius in m
omega = 7.29212e-5 # Earth's angular velocity in 1/s
h = 7000 # atmospheric scale height in m
p_r = 101325 # reference pressure in Pa

case = sys.argv[1]
short_name = sys.argv[2]
year = sys.argv[3] # yyyy
num_levs = int(sys.argv[4])
lat_bnd = int(sys.argv[5])


data_dir = \
        "/glade/derecho/scratch/aaroj/q.e22.QPMT.ne30_ne30_mg17." + case + "/run/"
save_dir = "/glade/derecho/scratch/aaroj/aqua_data/cam7_aqua/" + case + "/"

if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    print(f"Directory '{save_dir}' created.")
else:
    print(f"Directory '{save_dir}' already exists.")

fname = "*h4i." + year + "*"

fpath = data_dir + fname

glob_6hr = glob(fpath)
glob_6hr.sort()

#pdb.set_trace()

# create object to store perturbations by level
lev_dta = dict(zip(np.arange(1, num_levs + 1, 1), [[]]*num_levs))

timestep = 0
for file in glob_6hr:
    timestep += 1
    print("Doing file ", file)
    print("Timestep ", timestep)
    ds_6hr = xr.open_dataset(file)

    # select tropical latitudes and stratospheric levels
    ds_6hr = ds_6hr.sel(lat = slice(-lat_bnd, lat_bnd)).isel(
        lev = slice(None, num_levs))

    # calculate TEM variables

    th_mult = (p_r/(ds_6hr.lev*100))**(rd/cp)

    u = ds_6hr.U.astype('float32')
    v = ds_6hr.V.astype('float32')
    omega = ds_6hr.OMEGA.astype('float32')
    
    th = (ds_6hr.T*th_mult).astype('float32')
    th = th.assign_attrs(
        units = 'K', long_name = 'potential temperature')

    uzm = u.mean("lon").astype('float32')
    uzm = uzm.assign_attrs(
                    units = 'm/s', 
                    long_name = 'zonal mean zonal wind')
    
    duzm_dp = uzm.differentiate("lev").astype('float32')/100
    duzm_dp = duzm_dp.assign_attrs(
                        units = 'm/s*Pa',
                        long_name = 'zonal wind vertical derivative')
   
    thzm = th.mean("lon").astype('float32')
    thzm = thzm.assign_attrs(
                     units = 'K', 
                     long_name = 'zonal mean potential temperature')

    dthzm_dp = thzm.differentiate("lev").astype('float32')/100
    dthzm_dp = dthzm_dp.assign_attrs(
                       units = 'K/Pa',
                       long_name = 'potential temperature vertical derivative')
    
    tem_ds = xr.Dataset({'U':u,
                         'V':v,
                         'OMEGA':omega,
                         'TH':th,
                         'Uzm':uzm,
                         'dUzm_dP':duzm_dp,
                         'THzm':thzm,
                         'dTHzm_dP':dthzm_dp
                         })

    for l in range(num_levs):
        ds_l = tem_ds.isel(lev = [l])
        #ds_l = ds_6hr.isel(lev = [l])
        if timestep == 1:
            lev_dta[l+1] = [ds_l]
        else:
            lev_dta[l+1].append(ds_l)
#pdb.set_trace()

for lev in lev_dta.keys():
    if lev < 10:
        lev_str = "0" + str(lev)
    else:
        lev_str = str(lev)
    ds_lev = xr.concat(lev_dta[lev], dim = "time")
    ds_lev_fname = short_name + "_6hr_eq_level_" + lev_str + \
            "_year_" + year + ".nc"
    ds_lev_fpath = save_dir + ds_lev_fname
    print(f"Saving level {lev} for year {int(year)}.")
    ds_lev.to_netcdf(ds_lev_fpath)

