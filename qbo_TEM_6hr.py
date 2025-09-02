# code to calculate TEM terms using 6-hourly data
# will not calculate streamfunction for tropical data

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
start_year = sys.argv[3] # yy
end_year = sys.argv[4] #yy
num_levs = int(sys.argv[5])

years = ['00' + str(yr) for yr in range(int(start_year), int(end_year) + 1)]
# load in data and calculate TEM level by level

ds_allp = []

for l in range(1, num_levs + 1):
    if l < 10:
        strl = '0' + str(l)
    else:
        strl = str(l)
    print(f"Doing level {strl}.")
    data_dir = "/glade/derecho/scratch/aaroj/aqua_data/filter_data/" + \
    case + "/"
    ds_allyr = []
    for year in years:
        glob_yr_fpth = data_dir + short_name + \
        "_6hr_eq_level_" + strl + "_year_" + year + ".nc"
        ds_yr = xr.open_dataset(glob_yr_fpth)
        ds_allyr.append(ds_yr)
    print("Concatenating yearly data for a single level.")
    ds_allyr = xr.concat(ds_allyr, 'time')

    # TODO fix pressure derivatives in precond script
    # so they doesn't have to be scaled here

    print("Constructing perturbations for a single level.")
    uzm = ds_allyr.Uzm
    vzm = ds_allyr.V.mean('lon')
    wzm = ds_allyr.OMEGA.mean('lon')
    thzm = ds_allyr.THzm
    
    up = ds_allyr.U - uzm
    vp = ds_allyr.V - vzm
    wp = ds_allyr.OMEGA - wzm
    thp = ds_allyr.TH - thzm

    upvp = (up*vp).mean('lon')
    upwp = (up*wp).mean('lon')
    vpthp = (vp*thp).mean('lon')

    #psi = vpthp/(ds_allyr.dTHzm_dP/100) # convert to Pa

    ds_tem = xr.Dataset({'UpVpzm':upvp,
                         'UpWpzm':upwp,
                         'VpTHpzm':vpthp,
                         'Uzm':uzm,
                         'Vzm':vzm,
                         'Wzm':wzm,
                         'THzm':thzm})
    ds_allp.append(ds_tem)

# concatenate TEM data across all levels
print("Concatenating TEM data across levels.")
ds_allp = xr.concat(ds_allp, 'lev')

olat = ds_allp.lat
olev = ds_allp.lev
rlat = np.deg2rad(olat)
rlat = rlat.assign_coords({"lat":rlat.values})
palev = olev*100
palev = palev.assign_coords({"lev":palev.values})

coslat = np.cos(rlat)
sinlat = np.sin(rlat)

ds_allp['lat'] = rlat
ds_allp['lev'] = palev

upvp = ds_allp.UpVpzm
upwp = ds_allp.UpWpzm
vpthp = ds_allp.VpTHpzm
uzm = ds_allp.Uzm
vzm = ds_allp.Vzm
wzm = ds_allp.Wzm
thzm = ds_allp.THzm
#pdb.set_trace()

# calculate TEM streamfunction

psi = vpthp/thzm.differentiate('lev')

# calculate residual velocities
    
vtem = vzm - psi.differentiate('lev')

wtem = wzm + (1/(a*coslat))*(psi*coslat).differentiate('lat')

# advection U tendencies

utendvtem = vtem*(2*omega*sinlat - (1/(a*coslat))*(
    uzm*coslat).differentiate('lat'))

utendwtem = -wtem*uzm.differentiate('lev')

# EP flux 

epfy = a*coslat*(uzm.differentiate('lev')*psi - upvp)
epfz = a*coslat*((2*omega*sinlat - (1/(a*coslat))*(
    uzm*coslat).differentiate('lat'))*psi - upwp)

epfy = epfy.transpose("time","lev","lat")
epfz = epfz.transpose("time","lev","lat")

utendepfd = (1/(a*coslat))*((1/(a*coslat))*(
    epfy*coslat).differentiate('lat') + epfz.differentiate('lev'))

utendepfd = utendepfd.transpose("time","lev","lat")

# transform to log-pressure coordinates
# according to Gerber and Manzini (2016)

#z = -h*np.log(palev/p_r)

wtem *= -(h/palev)
epfy *= (palev/p_r) 
epfz *= -(h/p_r)
#utendepfd *= (palev/p_r) # do not do this transformation; follow Joe's code
                          # body forces should not be affected by coords

# scale variables and assign units & long name

utendepfd*=86400 
utendvtem*=86400
utendwtem*=86400

wtem*=1000 # mm/s

uzm = uzm.assign_attrs(units = 'm/s', 
                       long_name = 'Zonal mean zonal wind')
epfy = epfy.assign_attrs(units = 'm3/s2', 
                         long_name = 'meridional EP flux')
epfz = epfz.assign_attrs(units = 'm3/s2',
                         long_name = 'vertical EP flux')
vtem = vtem.assign_attrs(units = 'm/s',
                         long_name = 'meridional residual velocity')
wtem = wtem.assign_attrs(units = 'mm/s',
                         long_name = 'vertical residual velocity')
utendepfd = \
utendepfd.assign_attrs(units = 'm/s*day',
                    long_name = 'U tendency due to EP flux divergence')
utendvtem = \
utendvtem.assign_attrs(units = 'm/s*day',
                    long_name = 'U tendency due to meridional advection')
utendwtem = \
utendwtem.assign_attrs(units = 'm/s*day',
                    long_name = 'U tendency due to vertical advection')

# assemble into dataset

tem_ds = xr.Dataset({'Uzm':uzm,
                     'epfy':epfy,
                     'epfz':epfz,
                     'vtem':vtem,
                     'wtem':wtem,
                     'utendepfd':utendepfd,
                     'utendvtem':utendvtem,
                     'utendwtem':utendwtem})

tem_ds['lev'] = olev # convert back to hPa
tem_ds['lat'] = olat

# resample and do monthly mean

tem_ds_mm = tem_ds.resample(time = "ME").mean("time")

# save dataset
print("Saving monthly mean TEM dataset.")
save_fpth = data_dir + short_name + "_mm_TEM_years_00" + start_year + "_00" + end_year + ".nc"

if os.path.exists(save_fpth):
    os.remove(save_fpth)
    
tem_ds_mm.to_netcdf(save_fpth)




    

    



    
        
        



