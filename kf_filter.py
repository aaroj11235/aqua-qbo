import numpy as np
import xarray as xr
import pdb
from scipy.signal import detrend

# constants

rd = 287.0529 # specific gas constant of dry air in J/kg*K
cp = 1.0035e3 # specific heat capacity of dry air at constant pressure in J/kg*K 
p_r = 1000 # reference pressure in hPa
rho_r = 1.5 # reference density in kg/m^3
g = 9.81 # acceleration of gravity in m/s^2
h = 6800 # scale height in m for stratosphere
omega = 7.292e-5 # Earth's rotation rate
a = 6.37122e6 # Radius of Earth
bta = 2.28e-11 # meridional derivative of coriolis parameter evaluated at equator (from ncl script)

def split_hann_taper(series_length, taper_frac):
    npts = int(np.rint(taper_frac*series_length)) # total size of taper
    taper = np.hanning(npts)
    series_taper = np.ones(series_length)
    series_taper[0:npts//2+1] = taper[0:npts//2+1]
    series_taper[-npts//2+1:] = taper[npts//2+1:]

    return series_taper

def kf_filter(data_in, 
              spd,
              kbnd,
              fbnd,
              hbnd,
              sym,
              wavetype,
              bkgd_u = 0 # background zonal velocity at source in m/s
              ):

    """Perform space-time spectral decomposition and filter in wavenumber-frequency space
    
    data_ds: an xarray Dataset to be analyzed; needs to have (time, lat, lon) dimensions.
    
    kbnd: a tuple of wavenumbers for filtering

    fbnd: a tuple of frequencies for filtering

    wavetype: 0,1,2 are (asymmetric) MRG, n=2 WIG, n=0 EIG; 3,4,5 are (symmetric) Kelvin, n=1 ER, n=1 WIG 
                6 is general IG, only used for kc2015 method

    spd: sampling rate, in "samples per day" (e.g. daily=1, 6-houry=4)

    method: "kc2015" or "wk1998" or None; which method is used to filter the perturbation variables. "kc2015"
            is from Kim and Chun, 2015, and "wk1998" is from Wheeler and Kiladis, 1998.
    
    latitude_bounds: a tuple of (southern_extent, northern_extent) to reduce data size.
    
    
    Method
    ------
        1. Subsample in latitude if latitude_bounds is specified.
        2. Detrend the data 
        3. High-pass filter if rmvLowFrq is True
        4. Construct symmetric/antisymmetric array if dosymmetries is True.
        5. Apply taper in time dimension.
        6. Fourier transform
        7. Apply Hayashi method to get propagation direction 
        8. Filter according to wavenumber/frequency ranges
        9. Filter according to equatorial wave dispersion relationships
        10. Do inverse Fourier transform
        11. Return filtered variable
       
    """

    print('data shape at beginning ', data_in.sizes)
    
    lat_dim = data_in.dims.index('lat')

    if sym == "asym":
        data_asym = 0.5*(data_in.values - np.flip(data_in.values, axis = lat_dim))
        data = data_in.copy(deep = True, data = data_asym)
        del data_asym
    elif sym == "sym":
        data_sym = 0.5*(data_in.values + np.flip(data_in.values, axis = lat_dim))
        data = data_in.copy(deep = True, data = data_sym)
        del data_sym
    elif sym == "both":
        data_asym = 0.5*(data_in.values - np.flip(data_in.values, axis = lat_dim))
        data_a = data_in.copy(deep = True, data = data_asym)
        data_sym = 0.5*(data_in.values + np.flip(data_in.values, axis = lat_dim))
        data_s = data_in.copy(deep = True, data = data_sym)
        data = xr.concat([data_s,data_a],"component")
        data = data.assign_coords({"component":["symmetric", "antisymmetric"]})
        del data_asym, data_a, data_sym, data_s
    else:
        data = data_in.copy(deep = True)

    londim = data.dims.index('lon')
    timedim = data.dims.index('time')
    lat_size = data.sizes['lat']
    lon_size = data.sizes['lon']
    time_size = data.sizes['time']

    # taper both ends of the timeseries to mean using a Hann taper

    taper = split_hann_taper(time_size, 0.1)
    taper = xr.DataArray(taper, coords = [data.time], dims = ['time'], name = 'taper')
    data = taper*data  # taper to zero

    # transpose data so lon and time dimensions are last
    
    data = data.transpose(..., "lon","time").astype('float32')

    z_fft = np.fft.rfft2(data.values, axes = (-2, -1)) # was fft, no axes

    # flip aross wavenumber

    z_fft[...,1:,:] = np.flip(z_fft[...,1:,:],axis=-2)

    print(np.shape(z_fft))

    #pdb.set_trace()

    # generate dimensions and coordinates for the new array

    z_fft_dims = list(data.dims)
    z_fft_dims[-2] = "wavenumber"
    z_fft_dims[-1] = "frequency"
 
    z_fft_coord_vals = [0]*len(z_fft_dims)

    for i in range(len(z_fft_dims)):
        if z_fft_dims[i] == "wavenumber":
            z_fft_coord_vals[i] = np.fft.fftshift(np.fft.fftfreq(lon_size, 1/lon_size)) # adj
            #z_fft_coord_vals[i] = np.fft.fftfreq(lon_size, 1/lon_size)
        elif z_fft_dims[i] == "frequency":
            z_fft_coord_vals[i] = np.fft.rfftfreq(time_size, 1/spd) #adj was fftfreq
            #z_fft_coord_vals[i] = np.fft.fftfreq(time_size, 1/spd)
        else:
            z_fft_coord_vals[i] = data[z_fft_dims[i]]

    z_fft_coords = dict(zip(z_fft_dims, z_fft_coord_vals))

    z_reorder = xr.DataArray(np.fft.fftshift(z_fft, axes = -2), #adj # was z_fft 
                         dims = tuple(z_fft_dims),
                         coords = z_fft_coords)

    del z_fft

        
    if sym == 'both':
        # first nested list is bounds for symmetric part
        k_arr_s = z_reorder.wavenumber.sel(wavenumber = slice(kbnd[0][0],kbnd[1][0])).values
        kmin_s = np.min(k_arr_s)
        kmax_s = np.max(k_arr_s)
        f_arr_s = z_reorder.frequency.sel(frequency = slice(fbnd[0][0],fbnd[1][0])).values
        fmin_s = np.min(f_arr_s)
        fmax_s = np.max(f_arr_s)

        # second nested list is bounds for antisymmetric part
        k_arr_a = z_reorder.wavenumber.sel(wavenumber = slice(kbnd[0][1],kbnd[1][1])).values
        kmin_a = np.min(k_arr_a)
        kmax_a = np.max(k_arr_a)
        f_arr_a = z_reorder.frequency.sel(frequency = slice(fbnd[0][1],fbnd[1][1])).values
        fmin_a = np.min(f_arr_a)
        fmax_a = np.max(f_arr_a)

        # only use this case for box filter, so don't get h
    else:
        # get filtering parameters
        k_arr = z_reorder.wavenumber.sel(wavenumber = slice(kbnd[0],kbnd[1])).values
        kmin = np.min(k_arr)
        kmax = np.max(k_arr)
        f_arr = z_reorder.frequency.sel(frequency = slice(fbnd[0],fbnd[1])).values
        fmin = np.min(f_arr)
        fmax = np.max(f_arr)
        hmin = hbnd[0]
        hmax = hbnd[1]
        c_low = np.sqrt(g*hmin)
        c_high = np.sqrt(g*hmax)
        

    # change units of wavenumber for dispersion relation calculation (from unitless to 1/m)

    wn = z_reorder.wavenumber/a

    if ((wavetype == "MRG")&(sym != 'both')): # MRG
        
        # first, do box filter
        z_fft_filter = z_reorder.where(
            ((z_reorder.wavenumber >= kmin)&(z_reorder.wavenumber <= kmax)&\
                    (z_reorder.frequency >= fmin)&(z_reorder.frequency <= fmax)), 0)

        low_freq = wn*c_low*(0.5 - 0.5*(1+(4*bta)/(wn**2*c_low))**0.5)
        high_freq = wn*c_high*(0.5 - 0.5*(1+(4*bta)/(wn**2*c_high))**0.5)
        
        low_freq = (low_freq*86400)/(2*np.pi) # convert to cpd
        high_freq = (high_freq*86400)/(2*np.pi)

        # only apply to k < 0 since MRGs propagate westward
        z_fft_filter = z_fft_filter.where(((z_fft_filter.wavenumber < 0)&(
            z_fft_filter.frequency>low_freq)&(
            z_fft_filter.frequency<high_freq)),0)

    elif ((wavetype == "IG2")&(sym != 'both')): # n=2 IG
        
        # box filter
        z_fft_filter = z_reorder.where(
            ((z_reorder.wavenumber >= kmin)&(z_reorder.wavenumber <= kmax)&\
                    (z_reorder.frequency >= fmin)&(z_reorder.frequency <= fmax)), 0)

        low_freq = (5*bta*c_low + wn**2*c_low**2)**0.5
        high_freq = (5*bta*c_high + wn**2*c_high**2)**0.5

        # apply correction to dispersion relation
        for i in range(1,6): #Not sure why 6 is used, but this applies iterative correction to
                             #dispersion rel'n
            low_freq = (5*bta*c_low + wn**2*c_low**2 + bta*c_low**2*wn/low_freq)**0.5
            high_freq = (5*bta*c_high + wn**2*c_high**2 + bta*c_high**2*wn/high_freq)**0.5
        
        # convert to cpd
        low_freq = (low_freq*86400)/(2*np.pi)
        high_freq = (high_freq*86400)/(2*np.pi)

        z_fft_filter = z_fft_filter.where(((z_fft_filter.frequency>low_freq)&(
            z_fft_filter.frequency<high_freq)),0)

    elif ((wavetype == "IG0")&(sym != 'both')): #n=0 EIG
        
        # box filter
        z_fft_filter = z_reorder.where(
            ((z_reorder.wavenumber >= kmin)&(z_reorder.wavenumber <= kmax)&\
                    (z_reorder.frequency >= fmin)&(z_reorder.frequency <= fmax)), 0)
        
        # dispersion relationship
        low_freq = wn*c_low*(0.5 + 0.5*(1+(4*bta)/(wn**2*c_low))**0.5)
        high_freq = wn*c_high*(0.5 + 0.5*(1+(4*bta)/(wn**2*c_high))**0.5)

        # convert to cpd
        low_freq = (low_freq*86400)/(2*np.pi)
        high_freq = (high_freq*86400)/(2*np.pi)

        # only apply to positive wavenumbers since EIGWs propagate eastward
        z_fft_filter = z_fft_filter.where(((z_fft_filter.wavenumber > 0)&(
            z_fft_filter.frequency>low_freq)&(
            z_fft_filter.frequency<high_freq)),0)

    elif ((wavetype == "KELVIN")&(sym != 'both')): # Kelvin
        
        # box filter
        z_fft_filter = z_reorder.where(
            ((z_reorder.wavenumber >= kmin)&(z_reorder.wavenumber <= kmax)&\
                    (z_reorder.frequency >= fmin)&(z_reorder.frequency <= fmax)), 0)
        
        # dispersion relationship
        low_freq = wn*c_low
        high_freq = wn*c_high
        
        # convert to cpd
        low_freq = (low_freq*86400)/(2*np.pi)
        high_freq = (high_freq*86400)/(2*np.pi)

        # only apply to k>0 since KWs propagate eastward
        z_fft_filter = z_fft_filter.where(((z_fft_filter.wavenumber > 0)&(
            z_fft_filter.frequency>low_freq)&(
            z_fft_filter.frequency<high_freq)),0)

    elif ((wavetype == "ER")&(sym != 'both')): # n=1 ER
        
        # box filter
        z_fft_filter = z_reorder.where(
            ((z_reorder.wavenumber >= kmin)&(z_reorder.wavenumber <= kmax)&\
                    (z_reorder.frequency >= fmin)&(z_reorder.frequency <= fmax)), 0)
        
        # dispersion relationship
        low_freq = (-bta*wn)/(wn**2 + (3*bta)/c_low)
        high_freq = (-bta*wn)/(wn**2 + (3*bta)/c_high)

        low_freq = (low_freq*86400)/(2*np.pi)
        high_freq = (high_freq*86400)/(2*np.pi)
        
        # only apply to k < 0 since ERWs propagate westward
        z_fft_filter = z_fft_filter.where(((z_fft_filter.wavenumber < 0)&(
            z_fft_filter.frequency>low_freq)&(
            z_fft_filter.frequency<high_freq)),0)

    elif ((wavetype == "IG1")&(sym != 'both')): #n=1 IG

        # box filter
        z_fft_filter = z_reorder.where(
            ((z_reorder.wavenumber >= kmin)&(z_reorder.wavenumber <= kmax)&\
                    (z_reorder.frequency >= fmin)&(z_reorder.frequency <= fmax)), 0)
        
        # dispersion relationship
        low_freq = (3*bta*c_low + wn**2*c_low**2)**0.5
        high_freq = (3*bta*c_high + wn**2*c_high**2)**0.5

        # apply correction to dispersion relation
        for i in range(1,6):
            low_freq = (3*bta*c_low + wn**2*c_low**2 + bta*c_low**2*wn/low_freq)**0.5
            high_freq = (3*bta*c_high + wn**2*c_high**2 + bta*c_high**2*wn/high_freq)**0.5
        
        # convert to cpd
        low_freq = (low_freq*86400)/(2*np.pi)
        high_freq = (high_freq*86400)/(2*np.pi)

        z_fft_filter = z_fft_filter.where(((z_fft_filter.frequency>low_freq)&(
            z_fft_filter.frequency<high_freq)),0)


    elif ((wavetype == "SSG")&(sym != 'both')): # small-scale GWs

        # box filter, adjusted to select outside kbnd
        z_fft_filter = z_reorder.where(
            ((z_reorder.wavenumber <= kmin)&(z_reorder.wavenumber >= kmax)&\
                    (z_reorder.frequency >= fmin)&(z_reorder.frequency <= fmax)), 0)
    else:
        # box filter
        if sym == 'both':
            z_reorder_s = z_reorder.sel(component = "symmetric")
            z_fft_filter_s = z_reorder_s.where(
            ((z_reorder_s.wavenumber >= kmin_s)&(z_reorder_s.wavenumber <= kmax_s)&\
                    (z_reorder_s.frequency >= fmin_s)&(z_reorder_s.frequency <= fmax_s)), 0)
            
            z_reorder_a = z_reorder.sel(component = "antisymmetric")
            z_fft_filter_a = z_reorder_a.where(
            ((z_reorder_a.wavenumber >= kmin_a)&(z_reorder_a.wavenumber <= kmax_a)&\
                    (z_reorder_a.frequency >= fmin_a)&(z_reorder_a.frequency <= fmax_a)), 0)

            z_fft_filter = z_fft_filter_s + z_fft_filter_a 
        
        else:
            z_fft_filter = z_reorder.where(
                ((z_reorder.wavenumber >= kmin)&(z_reorder.wavenumber <= kmax)&\
                (z_reorder.frequency >= fmin)&(z_reorder.frequency <= fmax)), 0)

    del z_reorder
    
    z_undo_reorder = np.fft.ifftshift(z_fft_filter, axes = -2)

    del z_fft_filter
    
    # undo wavenumber flip

    z_undo_reorder[...,1:,:] = np.flip(z_undo_reorder[...,1:,:],axis=-2)


    z_filter = np.fft.irfft2(z_undo_reorder, axes = (-2,-1))
    #pdb.set_trace()

    del z_undo_reorder
    
    print('The shape of the filtered space-time array is:', np.shape(z_filter))

    # turn z_filter into DataArray
    #pdb.set_trace()
    if sym == 'both': # drop component dimension
        new_coords = data.drop_vars("component").coords
        new_dims = [dd for dd in list(data.dims) if dd != 'component']
    else:
        new_dims = data.dims
        new_coords = data.coords

    z_filter = xr.DataArray(
            z_filter, 
            dims = new_dims,
            coords = new_coords)
            #dims = new_dims,
            #coords = data_tp.drop_vars("component").coords) # coords = data_tp.drop_vars("component").coords, dims = new_dims
    
    z_filter = z_filter.transpose("time",...).astype('float32')
    
    
    return z_filter

