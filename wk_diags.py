import argparse
from pathlib import Path
import xarray as xr
import numpy as np
import qbo_diags as qd
import qbo_aux as qa
import pdb
from scipy.signal import convolve2d, detrend
from glob import glob

# constants needed for spectral flux calculation

a = 6371e3 # radius of Earth in m
omega = 7.292e-5 # rotation rate of Earth in 1/s
h = 7000 # atmospheric scale height in m
p_r = 101325 # reference pressure in Pa

###############################################################################

# Helper functions for WK analysis 

###############################################################################

def convolvePosNeg(arr, 
                   k, 
                   dim, 
                   boundary_index):
    """Apply convolution of (arr, k) excluding data at boundary_index in 
    dimension dim.
    Credit: Brian Medeiros
    
    arr: numpy ndarray of data
    k: numpy ndarray, same dimension as arr, this should be the kernel
    dim: integer indicating the axis of arr to split
    boundary_index: integer indicating the position to split dim
    
    Split array along dim at boundary_index;
    perform convolution on each sub-array;
    reconstruct output array from the two subarrays;
    the values of output at boundary_index of dim will be same as input.
    
    `convolve2d` is `scipy.signal.convolve2d()`
    """
    # arr: numpy ndarray
    oarr = arr.copy()  # maybe not good to make a fresh copy every time?
    # first pass is [0 : boundary_index)
    slc1 = [slice(None)] * arr.ndim
    slc1[dim] = slice(None, boundary_index)
    arr1 = arr[tuple(slc1)]
    ans1 = convolve2d(arr1, k, boundary='symm', mode='same')
    # second pass is [boundary_index+1, end]
    slc2 = [slice(None)] * arr.ndim
    slc2[dim] = slice(boundary_index+1,None)
    arr2 = arr[tuple(slc2)]
    ans2 = convolve2d(arr2, k, boundary='symm', mode='same')
    # fill in the output array
    oarr[tuple(slc1)] = ans1
    oarr[tuple(slc2)] = ans2
    return oarr


def simple_smooth_kernel():
    """Provide a very simple smoothing kernel.
       Credit: Brian Medeiros
    """
    kern = np.array([[0, 1, 0],[1, 4, 1],[0, 1, 0]])
    return kern / kern.sum()


def smooth_wavefreq(data, 
                    kern=None, 
                    nsmooth=None, 
                    freq_ax=None, 
                    freq_name=None):
    """Apply a convolution of (data,kern) nsmooth times.
       The convolution is applied separately to the positive and negative 
       frequencies. Either the name (freq_name: str) or axis index 
       (freq_ax: int) of frequency is required, with the name preferred.
       Credit: Brian Medeiros
    """
    assert isinstance(data, xr.DataArray)
    if kern is None:
        kern = simple_smooth_kernel()
    if nsmooth is None:
        nsmooth = 20
    if freq_name is not None:
        axnum = list(data.dims).index(freq_name)
        nzero =  data.sizes[freq_name] // 2 # <-- THIS IS SUPPOSED TO BE 
                                            #THE INDEX AT FREQ==0.0
    elif freq_ax is not None:
        axnum = freq_ax
        nzero = data.shape[freq_ax] // 2
    else:
        raise ValueError(
            "smooth_wavefreq needs to know how to find frequency dimension.")
    smth1pass = convolvePosNeg(data, kern, axnum, nzero) # this is a custom 
    #function to skip 0-frequency (mean)
    # note: the convolution is strictly 2D and the boundary condition 
    #is symmetric --> if kernel is normalized, preserves the sum.
    smth1pass = xr.DataArray(
        smth1pass, dims=data.dims, coords=data.coords) # ~copy_metadata
    # repeat smoothing many times:
    smthNpass = smth1pass.values.copy()
    for i in range(nsmooth):
        smthNpass = convolvePosNeg(smthNpass, kern, axnum, nzero)
    return xr.DataArray(smthNpass, dims=data.dims, coords=data.coords)

def wk_calc(data, 
            segsize=96, 
            noverlap=60, 
            spd=1, 
            latitude_bounds=None,
            do_tavg = True,
            do_symmetries = True,
            calc_power = True):
    """Perform space-time spectral decomposition and return power spectrum 
    following Wheeler-Kiladis approach. Modified version of 
    Brian Medeiros' code.

    data: an xarray DataArray to be analyzed; needs to have 
    (time, lat, lon) dimensions.
    
    segsize: integer denoting the size of time samples that will be 
    decomposed (typically about 96)
    
    noverlap: integer denoting the number of days of overlap from one 
    segment to the next (typically about segsize-60 => 2-month overlap)
    
    spd: sampling rate, in "samples per day" (e.g. daily=1, 6-houry=4)

    latitude_bounds: a tuple of (southern_extent, northern_extent) to reduce 
    data size.

    Method
    ------
        1. Subsample in latitude if latitude_bounds is specified.
        2. Detrend the data (but keeps the mean value, as in NCL)
        3. High-pass filter if rmvLowFrq is True
        4. Construct symmetric/antisymmetric array if dosymmetries is True.
        5. Construct overlapping window view of data.
        6. Detrend the segments (strange enough, removing mean).
        7. Apply taper in time dimension of windows (aka segments).
        8. Fourier transform
        9. Apply Hayashi reordering to get propagation direction & 
        convert to power.
       10. return DataArray with power.

    Notes
    -----
        Upon returning power, this should be comparable to "raw" spectra.
        Next step would be be to smooth with `smooth_wavefreq`,
        and divide raw spectra by smooth background to obtain "significant" 
        spectral power.

    """

    print('data shape at beginning ',np.shape(data))

    segsize = spd*segsize
    noverlap = spd*noverlap #AMJ: because otherwise rollstride 
                            #would make no sense

    if latitude_bounds is not None:
        assert isinstance(latitude_bounds, tuple)
        data = data.sel(lat=slice(*latitude_bounds))  # CAUTION: is this a 
        #mutable argument?
        print(f"Data reduced by latitude bounds. Size is {data.sizes}")
        slat = latitude_bounds[0]
        nlat = latitude_bounds[1]
    else:
        slat = data['lat'].min().item()
        nlat = data['lat'].max().item()

    # "Remove dominant signals"

    # "detrend" the data, including removing the mean 
    #(uses scipy.signal.detrend):
    #  --> ncl version keeps the mean:
    timedim = data.dims.index('time')
    xmean = data.mean(dim='time')
    xdetr = detrend(data.values, axis=timedim, type='linear')
    xdetr = xr.DataArray(xdetr, dims=data.dims, coords=data.coords)
    xdetr = xdetr + xmean

    # filter low-frequencies
    data = qa.rmvAnnualCycle(xdetr, spd, 1/segsize)
        
    #data = qa.dec2symasym(data) # this function adds a dimension 
        #"component"
        # that contains symmetric and antisymmetric parts (does it?)
    if do_symmetries:
        data = qa.decompose2SymAsym(data)

    dimsizes = data.sizes  # dict
    lon_size = dimsizes['lon']
    lat_size = dimsizes['lat']

    print('data shape after dosymmetries',np.shape(data))

    # 2. Windowing with the xarray "rolling" operation, and then limit 
    #overlap wiTH `CONSTruct` to produce a new dataArray.
    x_roll = data.rolling(time=segsize, min_periods=segsize)  # WK99 use 
    #96-day window
    rollstride = segsize - noverlap
    assert rollstride > 0, f"Error, inconsistent specification of segsize" + \
            " and noverlap results in stride of {rollstride}, but must be > 0."
    x_win = x_roll.construct("segments") #.dropna("time")  # WK99 say "2-month" 
    #overlap
    # dimension "segments" has length segsize, dimension "time" now has 
    # length equal to the number of overlapping segments
    # apply stride
    x_win = x_win.isel(time=slice(segsize-1,None,segsize-noverlap))

    print(f"[wk_calc] x_win shape is {x_win.shape}")
    seg_dim = x_win.dims.index('segments')
    # Additional detrend for each segment:
    if  np.logical_not(np.any(np.isnan(x_win))):
        print("No missing, so use simplest segment detrend.")
        x_win_detr = detrend(x_win.values, axis=seg_dim, type='linear') 
        #<-- missing 
        #data makes this not work
        x_win = xr.DataArray(x_win_detr, dims=x_win.dims, coords=x_win.coords)
    else:
        print(
                "EXTREME WARNING -- This method to detrend with missing" + \
                        " values present does not quite work, probably" + \
                        " need to do interpolation instead.")
        print("There are missing data in x_win, so have to try to" +\
                " detrend around them.")
        x_win_cp = x_win.values.copy()
        print("[spacetime_power] x_win_cp windowed data has shape" + \
                f" {x_win_cp.shape} \n \t It is a numpy array, copied from" +\
                f" x_win which has dims: {x_win.sizes} \n \t ** about to" + \
                " detrend this in the rightmost dimension.")
        x_win_cp[np.logical_not(np.isnan(x_win_cp))] = detrend(
                x_win_cp[np.logical_not(np.isnan(x_win_cp))])
        x_win = xr.DataArray(x_win_cp, dims=x_win.dims, coords=x_win.coords)

    # 3. Taper in time to make the signal periodic, as required for FFT.
    # taper = np.hanning(segsize)  # WK seem to use some kind of stretched 
    # out hanning window; unclear if it matters
    taper = qa.split_hann_taper(segsize, 0.1)  # try to replicate NCL's
    x_wintap = x_win*taper # would do XTAPER = 
                            #(X - X.mean())*series_taper + X.mean()
                           # But since we have removed the mean, 
                           #taper going to 0 is equivalent to taper going to 
                           #the mean.


    # transpose data so lon and segments dimensions are last

    x_wintap = x_wintap.transpose(..., "lon", "segments")

    print('data input to fft',np.shape(x_wintap))
    # do 2D fourier transform (normalize since we won't be doing inverse fft)

    z = np.fft.fft2(x_wintap, axes = (-2, -1))/(lon_size * segsize)

    # flip across wavenumber (this takes the place of Hayashi
    # reordering). Starts at index 1 b/c mean is first element of freq's.

    z[...,1:,:] = np.flip(z[...,1:,:],axis = -2)

    print('shape of fft data is ', np.shape(z))

    # calculate power
    if calc_power:
        z_power = (np.abs(z))**2
    else:
        z_power = np.copy(z)

    # generate coordinates for new array

    waven_val = np.fft.fftshift(np.fft.fftfreq(lon_size, 1/lon_size))
    freq_val = np.fft.fftshift(np.fft.fftfreq(segsize, 1/spd))
    
    z_arr = xr.DataArray(np.fft.fftshift(z_power, axes = (-2,-1)), dims=(
        "time","lat","wavenumber","frequency"),
                     coords={"time":x_wintap["time"],
                             "lat":x_wintap["lat"],
                             "wavenumber":waven_val,
                             "frequency":freq_val})

    print('shape of z_arr ',np.shape(z_arr))

    if not calc_power:
        # should not use do_symmetries when calc_power = False
        return z_arr
    else:
        # drop negative frequencies
        #z_arr = z_arr.where(z_arr.frequency >= 0, drop = True) 

        if do_symmetries:
            if do_tavg:
                # multiply by 2 b/c we only used one hemisphere
                z_symmetric = 2.0 * z_arr.isel(lat=z_arr.lat<0).mean(
                    dim='time').sum(dim='lat').squeeze()
                z_symmetric.name = "power"
                z_antisymmetric = 2.0 * z_arr.isel(lat=z_arr.lat>0).mean(
                    dim='time').sum(dim='lat').squeeze()
                z_antisymmetric.name = "power"
                z_final = xr.concat([z_symmetric, z_antisymmetric], 
                                    "component")
                z_final = z_final.assign_coords(
                    {"component":["symmetric","antisymmetric"]})
            else:
                # multiply by 2 b/c we only used one hemisphere
                z_symmetric = 2.0 * z_arr.isel(lat=z_arr.lat<0).sum(
                    dim='lat').squeeze()
                z_symmetric.name = "power"
                z_antisymmetric = 2.0 * z_arr.isel(lat=z_arr.lat>0).sum(
                    dim='lat').squeeze()
                z_antisymmetric.name = "power"
                z_final = xr.concat([z_symmetric, z_antisymmetric], 
                                    "component")
                z_final = z_final.assign_coords(
                    {"component":["symmetric","antisymmetric"]})
        else:
            if do_tavg:
                z_final = z_arr.mean(dim = 'time').sum(
                    dim = 'lat').squeeze()
            else:
                z_final = z_arr.sum(dim = 'lat').squeeze()
        print('z_final shape is ', np.shape(z_final))
        print('z_final coords are ', z_final.coords)

        return z_final

def genDispersionCurves(nWaveType=6, 
                        nPlanetaryWave=50, 
                        rlat=0, 
                        Ahe=[50, 25, 12]):
    """
    Function to derive the shallow water dispersion curves. 
    Closely follows NCL version. 
    Credit: Brian Medeiros

    input:
        nWaveType : integer, number of wave types to do
        nPlanetaryWave: integer
        rlat: latitude in radians (just one latitude, usually 0.0)
        Ahe: [50.,25.,12.] equivalent depths
              ==> defines parameter: nEquivDepth ; integer, number of 
              equivalent depths to do == len(Ahe)

    returns: tuple of size 2
        Afreq: Frequency, shape is (nWaveType, nEquivDepth, nPlanetaryWave)
        Apzwn: Zonal savenumber, shape is (nWaveType, nEquivDepth, 
        nPlanetaryWave)
        
    notes:
        The outputs contain both symmetric and antisymmetric waves. 
        In the case of 
        nWaveType == 6:
        0,1,2 are (ASYMMETRIC) "MRG", "IG", "EIG" (mixed rossby gravity, 
        inertial gravity, equatorial inertial gravity)
        3,4,5 are (SYMMETRIC) "Kelvin", "ER", "IG" (Kelvin, equatorial rossby,
        inertial gravity)
    """
    nEquivDepth = len(Ahe) # this was an input originally,but I don't know why.
    pi    = np.pi
    radius = 6.37122e06    # [m]   average radius of earth
    g     = 9.80665        # [m/s] gravity at 45 deg lat used by the WMO
    omega = 7.292e-05      # [1/s] earth's angular vel
    # U     = 0.0   # NOT USED, so Commented
    # Un    = 0.0   # since Un = U*T/L  # NOT USED, so Commented
    ll    = 2.*pi*radius*np.cos(np.abs(rlat))
    Beta  = 2.*omega*np.cos(np.abs(rlat))/radius
    fillval = 1e20
    
    # NOTE: original code used a variable called del,
    #       I just replace that with `dell` because `del` is a python keyword.

    # Initialize the output arrays
    Afreq = np.empty((nWaveType, nEquivDepth, nPlanetaryWave))
    Apzwn = np.empty((nWaveType, nEquivDepth, nPlanetaryWave))

    for ww in range(1, nWaveType+1):
        for ed, he in enumerate(Ahe):
            # this loops through the specified equivalent depths
            # ed provides index to fill in output array, while
            # he is the current equivalent depth
            # T = 1./np.sqrt(Beta)*(g*he)**(0.25) This is close to 
            #pre-factor of the dispersion relation, but is not used.
            c = np.sqrt(g * he)  # phase speed   
            L = np.sqrt(c/Beta)  # was: (g*he)**(0.25)/np.sqrt(Beta), 
                                 #this is Rossby radius of deformation        

            for wn in range(1, nPlanetaryWave+1):
                s  = -20.*(wn-1)*2./(nPlanetaryWave-1) + 20.
                k  = 2.0 * pi * s / ll
                kn = k * L 

                # Anti-symmetric curves  
                if (ww == 1):       # MRG wave
                    if (k < 0):
                        dell  = np.sqrt(1.0 + (4.0 * Beta)/(k**2 * c))
                        deif = k * c * (0.5 - 0.5 * dell)
                    
                    if (k == 0):
                        deif = np.sqrt(c * Beta)
                    
                    if (k > 0):
                        deif = fillval
                    
                
                if (ww == 2):       # n=0 IG wave
                    if (k < 0):
                        deif = fillval
                    
                    if (k == 0):
                        deif = np.sqrt( c * Beta)
                    
                    if (k > 0):
                        dell  = np.sqrt(1.+(4.0*Beta)/(k**2 * c))
                        deif = k * c * (0.5 + 0.5 * dell)
                    
                
                if (ww == 3):       # n=2 IG wave
                    n=2.
                    dell  = (Beta*c)
                    deif = np.sqrt((2.*n+1.)*dell + (g*he) * k**2)
                    # do some corrections to the above calculated frequency....
                    for i in range(1,5+1):
                        deif = np.sqrt((2.*n+1.)*dell + (g*he) * k**2 + \
                                       g*he*Beta*k/deif)
                    
    
                # symmetric curves
                if (ww == 4):       # n=1 ER wave
                    n=1.
                    if (k < 0.0):
                        dell  = (Beta/c)*(2.*n+1.)
                        deif = -Beta*k/(k**2 + dell)
                    else:
                        deif = fillval
                    
                if (ww == 5):       # Kelvin wave
                    deif = k*c

                if (ww == 6):       # n=1 IG wave
                    n=1.
                    dell  = (Beta*c)
                    deif = np.sqrt((2. * n+1.) * dell + (g*he)*k**2)
                    # do some corrections to the above calculated frequency
                    for i in range(1,5+1):
                        deif = np.sqrt((2.*n+1.)*dell + (g*he)*k**2 + \
                                       g*he*Beta*k/deif)
                
                eif  = deif  # + k*U since  U=0.0
                P    = 2.*pi/(eif*24.*60.*60.)  #  => PERIOD
                # dps  = deif/k  # Does not seem to be used.
                # R    = L #<-- this seemed unnecessary, I just changed R to 
                #L in Rdeg
                # Rdeg = (180.*L)/(pi*6.37e6) # And it doesn't get used.
            
                Apzwn[ww-1,ed-1,wn-1] = s
                if (deif != fillval):
                    # P = 2.*pi/(eif*24.*60.*60.) # not sure why we would 
                    #re-calculate now
                    Afreq[ww-1,ed-1,wn-1] = 1./P
                else:
                    Afreq[ww-1,ed-1,wn-1] = fillval
    return  Afreq, Apzwn

###############################################################################

# Wheeler-Kiladis Analysis Functions

###############################################################################

def wf_analysis(x, 
                segsize,
                noverlap,
                spd,
                latitude_bounds):
    """Return normalized spectra of x using standard processing parameters."""
    # Get the "raw" spectral power
    
    z2 = wk_calc(x,
                 segsize,
                 noverlap,
                 spd,
                 latitude_bounds,
                 do_tavg = True,
                 do_symmetries = True,
                 calc_power = True)
    
    z2avg = z2.mean(dim='component')
    z2.loc[{'frequency':0}] = np.nan # get rid of spurious power at \nu = 0
    # the background is supposed to be derived from both symmetric & 
    #antisymmetric
    background = smooth_wavefreq(z2avg, 
                                 kern=simple_smooth_kernel(), 
                                 nsmooth=50, 
                                 freq_name='frequency')
    # separate components
    z2_sym = z2[0,...].drop_vars("component")
    z2_asy = z2[1,...].drop_vars("component")
    # normalize
    nspec_sym = z2_sym / background 
    nspec_asy = z2_asy / background
    #if do_save:
    #    print(f"Save is triggered...")
    #    background.name = "background"
    #    z2_sym.name = "symmetric"
    #    z2_asy.name = "antisymmetric"
    dsout = xr.Dataset({"symmetric": nspec_sym, "antisymmetric":nspec_asy, 
                        "background":background})
    #    dsout.to_netcdf(ofil)
    #    print(f"Save is complete: {ofil}")        
    return dsout

def wf_analysis_qbo(x, 
                    U, 
                    l,
                    segsize,
                    noverlap,
                    spd,
                    latitude_bounds):
    
    """
       Wheeler-Kiladis analysis but for Westerly and Easterly phases of the
       QBO at level l.
    """
    # Get the "raw" spectral power
    
    z2 = wk_calc(x, 
                 segsize,
                 noverlap,
                 spd,
                 latitude_bounds,
                 do_tavg = False,
                 do_symmetries = True,
                 calc_power = True)
    #z2.to_netcdf("/glade/derecho/scratch/aaroj/wk_analysis_bpm.nc")
    # should be an easier way- just mask over positive
    # and negative U
    # figure out time average
    _, _, e2w, w2e = qd.calc_qbo_tt(U, l)
    # check whether e2w or w2e first
    if e2w[0] < w2e[0]: #e2w first
        r_w = [[e2w[tw],w2e[tw]] for tw in range(len(w2e))]
        r_e = [[w2e[te - 1],e2w[te]] for te in range(1, len(e2w))]
    else:
        r_w = [[e2w[tw - 1],w2e[tw]] for tw in range(1, len(w2e))]
        r_e = [[w2e[te],e2w[te]] for te in range(len(e2w))]

    # initialize list of datasets which will contain:
    w_ds = [] # westerly QBO spectra
    e_ds = [] # easterly QBO spectra

    for rw in r_w:
        # select range of dates
        ds_rw = z2.sel(
            time = slice(U.time[rw[0]].values,U.time[rw[1]].values))
        w_ds.append(ds_rw)
    for re in r_e:
        ds_re = z2.sel(
            time = slice(U.time[re[0]].values,U.time[re[1]].values))
        e_ds.append(ds_re)

    # concatenate datasets and average over time dimension

    e_ds = xr.concat(e_ds, dim = 'time')
    w_ds = xr.concat(w_ds, dim = 'time')

    z2_WQBO = w_ds.mean(dim = 'time')
    z2_EQBO = e_ds.mean(dim = 'time')

    z2avg_WQBO = z2_WQBO.mean(dim='component')
    z2avg_EQBO = z2_EQBO.mean(dim = 'component')

    z2_WQBO.loc[{'frequency':0}] = np.nan # get rid of spurious power at nu = 0
    z2_EQBO.loc[{'frequency':0}] = np.nan
    # the background is supposed to be derived from both 
    #symmetric & antisymmetric
    background_WQBO = \
            smooth_wavefreq(z2avg_WQBO, 
                            kern=simple_smooth_kernel(), 
                            nsmooth=50, 
                            freq_name='frequency')
    
    background_EQBO = \
            smooth_wavefreq(z2avg_EQBO, 
                            kern=simple_smooth_kernel(), 
                            nsmooth=50, 
                            freq_name='frequency')
    # separate components
    z2_sym_WQBO = z2_WQBO[0,...].drop_vars("component")
    z2_asy_WQBO = z2_WQBO[1,...].drop_vars("component")
    
    z2_sym_EQBO = z2_EQBO[0,...].drop_vars("component")
    z2_asy_EQBO = z2_EQBO[1,...].drop_vars("component")

    # normalize
    nspec_sym_WQBO = z2_sym_WQBO / background_WQBO
    nspec_asy_WQBO = z2_asy_WQBO / background_WQBO

    nspec_sym_EQBO = z2_sym_EQBO / background_EQBO
    nspec_asy_EQBO = z2_asy_EQBO / background_EQBO

    #if do_save:
    #    print(f"Save is triggered...")
    #    background.name = "background"
    #    z2_sym.name = "symmetric"
    #    z2_asy.name = "antisymmetric"
    dsout = xr.Dataset({"symmetric_WQBO":nspec_sym_WQBO, 
                        "antisymmetric_WQBO":nspec_asy_WQBO, 
                        "background_WQBO":background_WQBO,
                        "symmetric_EQBO":nspec_sym_EQBO,
                        "antisymmetric_EQBO":nspec_asy_EQBO,
                        "background_EQBO":background_EQBO})
    #    dsout.to_netcdf(ofil)
    #    print(f"Save is complete: {ofil}")
    return dsout

def ep_wf_analysis(years, 
                   levels,
                   physn,
                   casen, 
                   shortn,
                   segsize,
                   noverlap,
                   spd,
                   latitude_bounds):
    """
       Creates wavenumber-frequency spectrum of EP flux averaged
       over vertical pressure range[levels].
    """
    # Get the "raw" spectral power
    
    fps = []
    #pdb.set_trace()
    for l in levels:
        if l < 10:
            strl = '0' + str(l)
        else:
            strl = str(l)
        data_dir = \
        "/glade/derecho/scratch/aaroj/aqua_data/" + physn.lower() + \
        "_aqua/" + casen + "/"
        # load in data
        # list of year strings
        yrs = ['00' + str(y) for y in range(years[0],years[-1] + 1)] 
        ds_allyr = []
        for yr in yrs:
            fname = data_dir + shortn + "_6hr_eq_level_" + strl + \
            "_year_" + yr + ".nc" 
            ds_oneyr = xr.open_dataset(fname)
            ds_oneyr = ds_oneyr.sel(lat = slice(-15,15)) # further filter data
            ds_allyr.append(ds_oneyr)
        ds_all = xr.concat(ds_allyr, dim = 'time')

        # use squeeze to remove lev dimension from all arrays since
        # wk_calc can't handle it
        up = ds_all.U - ds_all.Uzm
        u_fft = wk_calc(up.squeeze(), 
                        segsize,
                        noverlap,
                        spd,
                        latitude_bounds,
                        do_tavg = True,
                        do_symmetries = False,
                        calc_power = False)
        vp = ds_all.V - ds_all.V.mean('lon')
        v_fft = wk_calc(vp.squeeze(),
                        segsize,
                        noverlap,
                        spd,
                        latitude_bounds,
                        do_tavg = True,
                        do_symmetries = False,
                        calc_power = False)
        wp = ds_all.OMEGA - ds_all.OMEGA.mean('lon')
        w_fft = wk_calc(wp.squeeze(),
                        segsize,
                        noverlap,
                        spd,
                        latitude_bounds,
                        do_tavg = True,
                        do_symmetries = False,
                        calc_power = False)
        thp = ds_all.TH - ds_all.THzm
        th_fft = wk_calc(thp.squeeze(),
                         segsize,
                         noverlap,
                         spd,
                         latitude_bounds,
                         do_tavg = True,
                         do_symmetries = False,
                         calc_power = False)

        dthdp  = ds_all.dTHzm_dP.squeeze()
        dudp   = ds_all.dUzm_dP.squeeze()
        u      = ds_all.Uzm
        coslat = np.cos(np.deg2rad(ds_all.lat))
        sinlat = np.sin(np.deg2rad(ds_all.lat))

        # calculate vertical component of EP flux
        fp = a*coslat*(((v_fft*th_fft.conjugate()).real/dthdp)*(
            2*omega*sinlat - ((u*coslat).differentiate('lat')/(
                a*coslat))) - (u_fft*w_fft.conjugate()).real)

        fp *= (-h/p_r) # change to log-p scaling

        # remove negative frequencies

        fp = fp.where(fp.frequency >= 0, drop = True)

        # average over time (segments) and latitude

        fp = fp.mean(dim = ['time','lat'])

        fps.append(fp)
    fps_final = xr.concat(fps, dim = 'lev').mean('lev')
    fps_final = \
    fps_final.assign_attrs(long_name = 'Vertical Component of EP Flux', 
                           units = 'm3/(s2*wn*cpd)')
    fps_final = xr.Dataset({'Fz':fps_final})

    #out_fpth = "/glade/derecho/scratch/aaroj/aqua_data/filter_data/" + \
    #casen + "/" + shortn + "_spectral_epf.nc"

    #if os.path.exists(out_fpth):
    #    os.remove(out_fpth)
        
    #fps_final.to_netcdf(out_fpth)
    return fps_final
