import xarray as xr
import numpy as np
import pdb
from kf_filter import split_hann_taper
#from CASutils import qbo_utils as qbo


def calc_qbo_tt(data, lev, do_smoothing = True, do_avg = True):
    '''

    Calculates QBO transition times from equatorial zonal wind
    data

    data : xarray DataArray, must include monthly-mean, zonal-mean
    variable U or Uzm and have dimensions time, lat, lev only
    lev : numeric type, pressure level at which to calculate 
    transition times, in hPa
    '''
    
    #pdb.set_trace()
    if lev != None:
        data = data.sel(lev = lev, method = 'nearest')
        data = data.drop_vars("lev")

    # average over 5S to 5N

    if "lat" in data.dims:
#        data = data.sel(lat = slice(-5,5))
#        gws = np.cos(np.deg2rad(data.lat))
#        data = data.weighted(gws).mean("lat")
        data = data.sel(lat = 0, method = 'nearest')
    # perform 5-month rolling average
    if do_smoothing:
        data = data.rolling(time = 5, center = True).mean().dropna("time")
    else:
        data = data[2:-2] # to change length like rolling method
    data_max = np.max(data) # get range of QBO timeseries for later
    data_min = np.min(data)

    # create shifted array to analyze U one timestep in the future
    data_s = data.shift({"time":-1}, fill_value = 0)

    # transition times are when product of non-shifted and
    # shifted arrays is less than zero (u one timestep ahead
    # has opposite sign ==> transition)
    data_p = data*data_s

    # get indices of negative values

    idx_neg = list(np.argwhere(data_p.values < 0).flatten())
    idx_to_remove = []
    # filter out quick peaks
    w2eloc = []
    e2wloc = []
    for j in range(len(idx_neg[:-1])):
        # get data between indices (single peak)
        rdata = data.isel(time = slice(idx_neg[j] + 1,
                                       idx_neg[j+1] + 1)).values
        # get peak value
        pval = rdata[np.argmax(np.abs(rdata))]
        # aquaplanet QBO is very asymmetric, so I'm assuming we can't use
        # mean and std as other studies have done to filter spurious peaks
        # instead use max or min of timeseries divided by 3 as a cutoff
        if (pval >= 0):
            if (pval <= data_max/3):
                idx_to_remove.append(idx_neg[j])
            else:
                e2wloc.append(idx_neg[j]+3) # +2 for rolling, +1 for method
        elif (pval < 0):
            if (pval >= data_min/3):
                idx_to_remove.append(idx_neg[j])
            else:
                w2eloc.append(idx_neg[j]+3) # these indices are on the right

        #if (idx_neg[j+1] - idx_neg[j])<= min_sep:#transit. times are too close
            # remove both transition times
        #    idx_to_remove.append(idx_neg[j+1])
            #idx_to_remove.append(idx_neg[j])

    #print(e2wloc)
    #print(w2eloc)
    # remove TTs
    for ix in list(set(idx_to_remove)): # remove duplicate indices
        idx_neg.remove(ix)

    # assuming transition times are correct, 
    #print(idx_neg)    
    num_per = (len(idx_neg)-1)//2 # num full periods

    periods = []
    amplitudes = []

    for i in range(num_per):
        i*=2
        start_p = idx_neg[i]
        end_p = idx_neg[i+2]
        pd = end_p - start_p
        pd_vals = data.isel(time = slice(start_p + 1, end_p + 1)).values
        pd_max = np.max(pd_vals)
        pd_min = np.min(pd_vals)
        amp = (pd_max - pd_min)/2
        periods.append(pd)
        amplitudes.append(amp)

    if do_avg:
        return np.mean(np.array(periods)), np.mean(np.array(amplitudes)), e2wloc, w2eloc
    else:
        return periods, amplitudes, e2wloc, w2eloc

def we_split(ds, l, do_smoothing = True):

    #ds_trop = ds.sel(lat = slice(-5,5))
    #ds_ts = ds_trop.sel(lev=10, method = 'nearest')
    #lats = ds_ts.lat
    #gw = np.cos(np.deg2rad(lats))
    #u_ds_ts = ds_ts.U
    #u_mean = u_ds_ts.weighted(gw).mean('lat').squeeze()
    if "U" in list(ds.variables):
        _, _, e2wloc, w2eloc = calc_qbo_tt(ds.U, lev = l,
                                           do_smoothing = do_smoothing)
    elif "Uzm" in list(ds.variables):
        _, _, e2wloc, w2eloc = calc_qbo_tt(ds.Uzm, lev = l,
                                           do_smoothing = do_smoothing)
    #pdb.set_trace()
    
    #e2w_i = [int(x) for x in e2wloc]
    #w2e_i = [int(x) for x in w2eloc]
    # include neighboring indices for averaging purposes

    # by construction, index x is the first index of the 
    # new wind regime. For averaging purposes, include the 
    # last index from the old regime.
    # e.g. for a e2w transition, x - 1 would be easterly
    # and x would be westerly. Thus we at least bracket
    # the shear line, since U will never be exactly zero
    e2w_i = [[x - 1, x] for x in e2wloc] 
    w2e_i = [[x - 1, x] for x in w2eloc] 

    e2w_i = np.array(e2w_i)
    w2e_i = np.array(w2e_i)
    e2w_i = e2w_i.astype('int')
    w2e_i = w2e_i.astype('int')
    e2w_i = e2w_i.flatten()
    w2e_i = w2e_i.flatten()
    e2w_i = list(e2w_i)
    w2e_i = list(w2e_i)

    print(e2w_i)
    print(w2e_i)

    ds_e2w = ds.isel(time = e2w_i)
    ds_w2e = ds.isel(time = w2e_i)

    ds_e2w_mean = ds_e2w.mean('time', keep_attrs = True)
    ds_w2e_mean = ds_w2e.mean('time', keep_attrs = True)

    return ds_e2w_mean, ds_w2e_mean

def calc_ddamp(data,
               deseasonalize = False
              ):
    """Calculate the Dunkerton and Delisi QBO amplitude. 
    From NCAR ADF diagnostics."""
    
    if deseasonalize:
        datseas = data.groupby('time.month').mean('time')
        datdeseas = data.groupby('time.month')-datseas
        ddamp = np.sqrt(2)*datdeseas.std(dim='time')
        ddamp = ddamp.assign_attrs(units = data.units, long_name = data.long_name)
        
    else:
        ddamp = np.sqrt(2)*data.std(dim = 'time')
        ddamp = ddamp.assign_attrs(units = data.units, long_name = data.long_name)
    return ddamp

def fft_QBO(data,
            plev,
            spd,
            lat_range
            ):
    """Produce a frequency spectrum of the QBO at a single pressure level.

    data: xarray.DataArray of zonal wind
    
    """
    if plev != None: # select pressure level
        data = data.sel(lev = plev, method = 'nearest')
        data = data.drop_vars("lev")

    if "lon" in data.dims:
        data_zm = data.mean('lon', keep_attrs = True)
    
    if "lat" in data.dims:
        data_trop = data_zm.sel(lat = slice(lat_range[0],lat_range[1]))
        # do weighted average across latitude
        data_trop = data_trop.weighted(
        np.cos(np.deg2rad(data_trop.lat))).mean('lat', keep_attrs = True)
    
    print(f"Final shape of data is: {data_trop.sizes}")
    # should be 1D timeseries at this point

    time_size = data_trop.sizes['time']
    time_dim = data_trop.dims.index('time')
    # taper timeseries

    taper = split_hann_taper(time_size, 0.1)
    taper = xr.DataArray(taper,
                         coords = [data_trop.time],
                         dims = ['time'],
                         name = 'taper')
    data_tp = taper*data_trop
    
    # do fft
    data_fft = np.fft.rfft(data_trop.values) # do real fft

    # get coordinates
    data_fft_dims = ["frequency"]
    data_fft_coord_vals = np.fft.rfftfreq(time_size, 1/spd)
    data_fft_coords = dict(zip(data_fft_dims, [data_fft_coord_vals]))

    fft_qbo = xr.DataArray(data_fft, dims = data_fft_dims, 
                           coords = data_fft_coords)
    
    return fft_qbo

def calc_TEM(data):

    data['lat'] = np.deg2rad(data.lat)
    data['lev'] = data.lev*100
    uzm = data.Uzm
    thzm = data.THzm
    #if "VpTHpzm" in data.variables:
    vpthp_zm = data.VpTHpzm
    upvp_zm = data.UpVpzm
    upwp_zm = data.UpWpzm
    #else:
    #    vpthp_zm = data.VTHzm
    #    upvp_zm = data.UVzm
    #    upwp_zm = data.UWzm
    lat = data.lat
    coslat = np.cos(lat)
    uzmcos = uzm*coslat
    sinlat = np.sin(lat)

    # calculate EP Flux, f_phi is the latitudinal component,
    # f_z is the vertical component (in pressure coords.)

    f = 2*omega*sinlat

    f_phi = a*coslat*(
            (uzm.differentiate('lev')*vpthp_zm)/thzm.differentiate('lev') - upvp_zm)

    f_p = a*coslat*(
            ((f - (a*coslat)**-1*uzmcos.differentiate(
                'lat'))*vpthp_zm)/thzm.differentiate('lev') - upwp_zm)

    # calculate momentum and heat components of ep flux divergence

    epfmy = (a*coslat**2)**-1*(-1*coslat**2*upvp_zm).differentiate('lat')
    epfhy = (a*coslat**2)**-1*(coslat**2*(uzm.differentiate(
        'lev')*vpthp_zm/thzm.differentiate('lev'))).differentiate('lat')

    epfmz = (-1*upwp_zm).differentiate('lev')
    epfhz = ((f - (a*coslat)**-1*uzmcos.differentiate(
        'lat'))*vpthp_zm/thzm.differentiate('lev')).differentiate('lev')

    # calculate total momentum and total heat EP flux

    epfm = epfmy + epfmz
    epfh = epfhy + epfhz

    # calculate divergence of EP Flux (sum of momentum and heat terms)

    delf = epfm + epfh

    delf *= 86400 # change to (m/s)/day

    # transpose arrays and assign attributes

    uzm = uzm.transpose("time","lev","lat") # ensure dimension ordering is correct
                                # time is first (to match input data)

    delf = delf.transpose("time","lev","lat")
    delf = delf.assign_attrs(
            units = 'm/s*day',
            long_name = 'U tendency due to EP flux divergence')

    epfmy*=86400
    epfmy = epfmy.transpose("time","lev","lat")
    epfmy = epfmy.assign_attrs(
            units = 'm/s*day',
            long_name = 'U tendency due to EP meridional momentum flux')

    epfmz*=86400
    epfmz = epfmz.transpose("time","lev","lat")
    epfmz = epfmz.assign_attrs(
            units = 'm/s*day',
            long_name = 'U tendency due to EP vertical momentum flux')

    epfhy*=86400
    epfhy = epfhy.transpose("time","lev","lat")
    epfhy = epfhy.assign_attrs(
            units = 'm/s*day',
            long_name = 'U tendency due to EP meridional heat flux')

    epfhz*=86400
    epfhz = epfhz.transpose("time","lev","lat")
    epfhz = epfhz.assign_attrs(
            units = 'm/s*day',
            long_name = 'U tendency due to EP vertical heat flux')

    epfm*=86400
    epfm = epfm.transpose("time","lev","lat")
    epfm = epfm.assign_attrs(
            units = 'm/s*day',
            long_name = 'U tendency due to EP momentum flux')

    epfh*=86400
    epfh = epfh.transpose("time","lev","lat")
    epfh = epfh.assign_attrs(
            units = 'm/s*day',
            long_name = 'U tendency due to EP heat flux')

    f_p/=100
    f_p = f_p.transpose("time","lev","lat")
    f_p = f_p.assign_attrs(
            units = 'm^2*hPa/s^2',
            long_name = 'Vertical component of EP flux')

    f_phi = f_phi.transpose("time","lev","lat")
    f_phi = f_phi.assign_attrs(
            units = 'm^3/s^2',
            long_name = 'Meridional component of EP flux')

    # combine variables in a dataset
    comb_ds = xr.Dataset({"U":uzm,
                          "utendepfd":delf,
                          "utendepfmy":epfmy,
                          "utendepfmz":epfmz,
                          "utendepfhy":epfhy,
                          "utendepfhz":epfhz,
                          "utendepfm":epfm,
                          "utendepfh":epfh,
                          "epfy":f_phi,
                          "epfz":f_p
                          })

    comb_ds['lat'] = np.rad2deg(lat)
    comb_ds['lev'] = comb_ds.lev/100

    return comb_ds

