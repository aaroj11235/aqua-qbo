# code to calculate TEM terms using monthly-mean perturbation output

import xarray as xr
import numpy as np
from glob import glob
import sys
import pdb
import os
from geocat.comp import interp_hybrid_to_pressure

rd = 287.058 # specific gas constant of dry air in J/kg*K
cp = 1.00464e3 # specific heat capacity of dry air at const. pressure in J/kg*K
g = 9.80665 # acceleration of gravity in m/s^2
a = 6.37123e6 # Earth's radius in m
omega = 7.29212e-5 # Earth's angular velocity in 1/s
h = 7000 # atmospheric scale height in m
p_r = 101325 # reference pressure in Pa

compset = sys.argv[1]
phys = sys.argv[2]
res = sys.argv[3]
case = sys.argv[4]
short_name = sys.argv[5]
start_year = int(sys.argv[6]) # start year to avoid spinup time
end_year = int(sys.argv[7])

num_years = end_year - start_year + 1
num_months = num_years*12 

data_dir = "/glade/derecho/scratch/aaroj/" + compset[0].lower() + ".e22." + \
compset + "." + res + "." + case + "/run/"

save_dir = "/glade/derecho/scratch/aaroj/aqua_data/" + phys.lower() + \
"_aqua/" + case + "/"

if not os.path.isdir(save_dir):
    os.mkdir(save_dir)  

mm_dta_fpath = save_dir + case.replace(".","_") + "_mm_data.nc"
mm_tm_dta_fpath = save_dir + case.replace(".","_") + "_mm_tm_data.nc"
overwrite = True # whether to overwrite existing monthly mean concatenated file

tem_vars = ['THzm','UVzm','UWzm','Uzm','VTHzm','Vzm','WTHzm','Wzm']

try:
    if overwrite:
        if (os.path.exists(mm_dta_fpath) and os.path.exists(mm_tm_dta_fpath)):
            os.remove(mm_dta_fpath)
            os.remove(mm_tm_dta_fpath)
        raise FileNotFoundError        
    mm_data_zm = xr.open_dataset(mm_dta_fpath)
except FileNotFoundError:

    mm_data = []
    for i in range(num_months):
        #pdb.set_trace()
        print(f"Doing month {i+1} out of {num_months}")
        #pdb.set_trace()
        year = start_year + i//12
        month = i + 1 - 12*(year - start_year)
    
        if year < 10:
            year = '000' + str(year)
        else:
            year = '00' + str(year)

        if month < 10:
            month = '0' + str(month)
        else:
            month = str(month)

        yrmon_str = year + "-" + month

        # open datasets

        case_pref = compset[0].lower() + ".e22." + \
        compset + "." + res + "." + case + ".cam."

        gc_data_fpth = data_dir + case_pref + "h0a." + yrmon_str + ".nc"
        tem_data_fpth = data_dir + case_pref + "h2a." + yrmon_str + ".nc"

        gc_data = xr.open_dataset(gc_data_fpth)
        tem_data = xr.open_dataset(tem_data_fpth)
        #pdb.set_trace()
        # first, interpolate tem data onto finer gc data latitude grid
        # and remove zalon dimension (extrapolate at poles)

        tem_data = tem_data.mean('zalon', keep_attrs = True)
        tem_data = tem_data.rename({'zalat':'lat'})
        tem_data = tem_data.interp(lat = gc_data.lat, 
                                   kwargs={"fill_value": "extrapolate"})
        for v in tem_vars:
            tem_data[v] = getattr(tem_data, v).astype('float32')

        # merge the datasets now
        #pdb.set_trace()
        all_mm_dta = xr.merge([gc_data,tem_data], compat = 'override')

        all_mm_dta = all_mm_dta.drop_vars(['date_written','time_written','time_bounds',
                                           'trop_cld_lev','trop_pref','trop_prefi'])
        all_mm_dta = all_mm_dta.mean('lon', keep_attrs = True)
        
        # now interpolate ------------------------------------------
        
        print("Interpolating merged data...")

        mm_vars = list(all_mm_dta.keys()) # list of all vars in dataset

        # pick out variables with level dimension (for interpolation)
        mm_vars_w_lev = [
            mm_var for mm_var in mm_vars if 'lev' in all_mm_dta[mm_var].dims]
        # remove hyam and hybm
        mm_vars_w_lev.remove('hyam')
        mm_vars_w_lev.remove('hybm')

        # get lev attributes

        olev = all_mm_dta.lev
        palev = olev*100
        palev = palev.assign_coords({"lev":palev.values})
        #lev_name = all_mm_dta['lev'].long_name
        #lev_units = all_mm_dta['lev'].units
        #lev_pos = all_mm_dta['lev'].positive

        # convert level dimension to Pa for interpolating

        all_mm_dta['lev'] = palev
        
        hyam = all_mm_dta['hyam']
        hybm = all_mm_dta['hybm']
        psurf = all_mm_dta['PS']
        #levs = all_mm_dta['lev']
        
        #print(f"Interpolating file {f+1} of {len(mm_dta)}")
        # interpolate variables from sigma coords to pressure levels
        for mmv in mm_vars_w_lev:
            #print(f"Doing variable {mmv}")
            mmv_temp = all_mm_dta[mmv]
            mmv_temp_i = interp_hybrid_to_pressure(data = mmv_temp,
                                                   ps = psurf,
                                                   hyam = hyam,
                                                   hybm = hybm,
                                                   p0 = 100000.,
                                                   new_levels = palev.values,
                                                   lev_dim = 'lev',
                                                   extrapolate = True,
                                                   variable = 'other')
            
            # rename lev dimension
            mmv_temp_i = mmv_temp_i.rename({'plev':'lev'})
            mmv_temp_i_comp = mmv_temp_i.compute() # load interpolated array
            all_mm_dta[mmv] = mmv_temp_i_comp
        
        # calculate Brunt-Vaisala frequency and add to dataset
        N2 = -((all_mm_dta.lev*g**2)/(
            rd*all_mm_dta.T*all_mm_dta.THzm))*all_mm_dta.THzm.differentiate('lev')
        N2 = N2.astype(np.float32).transpose('time','lev','lat')
        all_mm_dta['N2'] = N2.assign_attrs(units = '1/s^2', 
                                           long_name = 'Buoyancy Frequency Squared')


        # Do TEM calculations --------------------------
        print("Doing TEM calculations...")
        #pdb.set_trace()
        # convert latitude to radians for TEM derivatives

        olat = all_mm_dta.lat
        rlat = np.deg2rad(olat)
        rlat = rlat.assign_coords({"lat":rlat.values})
    
        coslat = np.cos(rlat)
        sinlat = np.sin(rlat)

        all_mm_dta['lat'] = rlat

        upvp = all_mm_dta.UVzm
        upwp = all_mm_dta.UWzm
        vpthp = all_mm_dta.VTHzm
        
        uzm = all_mm_dta.Uzm
        vzm = all_mm_dta.Vzm
        wzm = all_mm_dta.Wzm
        thzm = all_mm_dta.THzm

        # (approximately) scale terms with vertical velocity 
        # to get vertical pressure velocity

        upwp *= -(all_mm_dta.lev/(rd*all_mm_dta.T))*g
        wzm *= -(all_mm_dta.lev/(rd*all_mm_dta.T))*g
        
        #pdb.set_trace()

        # calculate TEM streamfunction
        #pdb.set_trace()
        psi = vpthp/thzm.differentiate('lev')
        # replace infs in psi with nans
        psi = psi.where(~xr.apply_ufunc(np.isinf,psi))
        kwargs = {'fill_value':'extrapolate'}
        psi = psi.interpolate_na(dim = 'lev', **kwargs)
        
        # calculate residual velocities
    
        vtem = vzm - psi.differentiate('lev')

        wtem = wzm + (1/(a*coslat))*(psi*coslat).differentiate('lat')

        # calculate residual streamfunction

        psitem = xr.zeros_like(vtem)
        # streamfunction as in "BDC in CMIP6" Abalos et al. (2021)
        for k in range(1,len(palev)):
            psitem[:,k,:] = (coslat/g)*vtem.isel(
            lev = slice(None,k)).integrate(coord = 'lev')       

        # advection U tendencies
        #pdb.set_trace()
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

        # add TEM data to merged dataset
        all_mm_dta['epfy'] = epfy
        all_mm_dta['epfz'] = epfz
        all_mm_dta['vtem'] = vtem
        all_mm_dta['wtem'] = wtem
        all_mm_dta['psitem'] = psitem
        all_mm_dta['utendvtem'] = utendvtem
        all_mm_dta['utendwtem'] = utendwtem
        all_mm_dta['utendepfd'] = utendepfd
        # residual tendency (dycore diffusion?)
        all_mm_dta['utendres'] = all_mm_dta.UTEND_TOT - \
                                 all_mm_dta.utendepfd - \
                                 all_mm_dta.utendvtem - \
                                 all_mm_dta.utendwtem - \
                                 all_mm_dta.UTEND_GWDTOT
        # total u tendency due to advection
        all_mm_dta['utendadv'] = utendvtem + utendwtem

        # remove TEM ingredients from dataset
        all_mm_dta = all_mm_dta.drop_vars(tem_vars)
        
        # scale variables and assign units & long name

        scale_dict = {'CLDTOT':[100,'Vertically-integrated total cloud','%'],
                      'PRECC':[86400000,'Convective precipitation rate','mm/day'],
                      'PRECL':[86400000,'Large-scale precipitation rate','mm/day'],
                      'PRECT':[86400000,'Total precipitation rate','mm/day'],
                      'PS':[1/100,'Surface pressure','hPa'],
                      'epfy':[1,'meridional EP flux','m3/s2'],
                      'epfz':[1,'vertical EP flux','m3/s2'],
                      'vtem':[1,'meridional residual velocity','m/s'],
                      'wtem':[1000,'vertical residual velocity','mm/s'],
                      'utendepfd':[86400,'U tendency due to EP flux divergence','m/s*day'],
                      'utendvtem':[86400,'U tendency due to meridional advection','m/s*day'],
                      'utendwtem':[86400,'U tendency due to vertical advection','m/s*day'],
                      'utendres':[86400,'U tendency due to dycore diffusion','m/s*day'],
                      'utendadv':[86400,'U tendency due to advection','m/s*day'],
                      'CLDICE':[1000,'Cloud ice amount','g/kg'],
                      'CLDLIQ':[1000,'Cloud liquid amount','g/kg'],
                      'CLOUD':[100,'Cloud percentage','%'],
                      'Q':[1000,'Specific humidity','g/kg'],
                      'QRS':[86400,'Solar heating rate','K/day'],
                      'QRL':[86400,'Longwave heating rate','K/day'],
                      'TROP_P':[1/100,'Tropopause pressure','hPa'],
                      'UTEND_CORE':[86400,'U tendency due to dynamical core','m/s*day'],
                      'UTEND_GWDTOT':[86400,'U tendency due to gravity wave drag','m/s*day'],
                      'UTEND_PHYSTOT':[86400,'U tendency due to physics','m/s*day'],
                      'UTEND_TOT':[86400,'Total U tendency','m/s*day'],
                      'psitem':[1,'Residual streamfunction','kg/m*s'],
                      'NETDT':[86400,'Net heating rate','K/day'],
                      'PTTEND':[86400,'T total physics tendency','K/day'],
                      'TTEND_TOT':[86400,'Total temperature tendency','K/day'],
                      'DTCOND':[86400,'T tendency - moist processes','K/day']}


        for s in list(scale_dict.keys()):
            temp_s = getattr(all_mm_dta,s)*scale_dict[s][0] # scale variable
            all_mm_dta[s] = temp_s.assign_attrs(units = scale_dict[s][2],
                                                 long_name = scale_dict[s][1])
        
        # convert latitude back to degrees
        all_mm_dta['lat'] = olat
        
        # convert level units back to hPa

        #new_lev = all_mm_dta.lev/100
        all_mm_dta['lev'] = olev
        #new_lev.assign_attrs(units = lev_units, 
        #                                         long_name = lev_name,
        #                                         positive = lev_pos)

        # append single mm file to list of all mm files
        mm_data.append(all_mm_dta)
        #pdb.set_trace()
    # concatenate all mm data

    mm_data_concat = xr.concat(mm_data, 'time')
    # save data
    mm_data_concat.to_netcdf(mm_dta_fpath)

    # create and save time-mean dataset

    weights = [31,28,31,30,31,30,31,31,30,31,30,31]*num_years

    wgts = xr.DataArray(weights, 
                        dims = mm_data_concat.time.dims, 
                        coords = mm_data_concat.time.coords)

    mm_data_concat_tm = mm_data_concat.weighted(wgts).mean('time',keep_attrs = True)

    tm_fpth = save_dir + case.replace(".","_") + "_mm_tm_data.nc"
    mm_data_concat_tm.to_netcdf(tm_fpth)


    


    

    
