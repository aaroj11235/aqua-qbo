import pdb
import xrft
import copy
import numpy as np
import xarray as xr
import dask.array as dsar

#### Initialize Constants #####
# should put these constants in a shared file

rearth = 6.37123e6 # radius of earth in m
omega = 7.29212e-5 # Earth's rotation rate in 1/s
p0 = 101325 # reference surface pressure in Pa
H = 7000 # scale height in m
g0 = 9.80665 # global average of gravity in m/s2
Rd = 287.058 # gas constant for dry air in J/kg*K
rho_surf = p0/(H*g0) # reference surface density in kg/m3
t_surf = g0*H/Rd # reference surface temperature in K
Cp = 1004.64 # specific heat at constant pressure for dry air, J/kg*K
bta = 2.28e-11 # meridional derivative of coriolis parameter evaluated
# at equator

def calc_dispersion(wavetype, 
                    wn, 
                    hs, 
                    u_shift = None, 
                    rescale = True,
                    fill_val = 0):

    '''
    Calculate array of frequencies following specified wave
    dispersion relationship.

    wavetype: string, must be one of approved wavetype names
    wn: 1D array of wavenumbers, units 1/m
    hs: 1D array of equivalent depths, units m
    u_shift: Background wind speed, units m/s
    rescale: Boolean, whether to rescale output to 1/day
    fill_val: frequency value to use when dispersion relationship
    is undefined. For example, MRG for positive wavenumbers
    '''

    # check that user-specified wavetype is implemented
    allowed_wavetypes = ['KELVIN','MRG','IG0','IG1','IG2','ER']
    assert wavetype in allowed_wavetypes
        
    cs = np.sqrt(g0*hs)
    
    freqs = [] # list of dispersion curves for different h

    for c in cs:
        if wavetype == 'KELVIN':
            freq = wn*c
            
        elif wavetype == 'MRG':
            # only defined for negative wavenumbers
            freq = np.zeros_like(wn)
            wn_filt = wn[wn<0]
            freq[wn < 0] = wn_filt*c*(0.5 - 0.5*(1+(4*bta)/(wn_filt**2*c))**0.5)
            freq[wn >= 0] = fill_val
        
        elif wavetype == 'IG0':
            # only defined for positive wavenumbers
            freq = np.zeros_like(wn)
            wn_filt = wn[wn>0]
            freq[wn > 0] = wn_filt*c*(0.5 + 0.5*(1+(4*bta)/(wn_filt**2*c))**0.5)
            freq[wn <= 0] = fill_val
        
        elif wavetype == 'IG1':
            freq = (3*bta*c + wn**2*c**2)**0.5
            # apply iterative correction
            for i in range(5):
                freq = (3*bta*c + wn**2*c**2 + \
                        bta*c**2*wn/freq)**0.5
        
        elif wavetype == 'IG2':
            freq = (5*bta*c + wn**2*c**2)**0.5
            # apply iterative correction
            for i in range(5):
                freq = (5*bta*c + wn**2*c**2 + \
                        bta*c**2*wn/freq)**0.5

        elif wavetype == 'ER':
            freq = (-bta*wn)/(wn**2 + (3*bta)/c)
        
        # apply doppler shift
        if u_shift != None:
            freq += wn*u_shift

        # rescale to cpd
        if rescale:
            freq *= ((86400)/(2*np.pi))

        # eliminate negative frequencies
        freq[freq<0] = fill_val
        
        freqs.append(freq)

    freq_dict = dict(zip(hs,freqs))

    return freq_dict

def detrend_dim(da, dim, deg=1):
    '''xarray function for detrending along a single
    dimension. Also compatible w/ dask chunked arrays'''
    p = da.polyfit(dim=dim, deg=deg)
    fit = xr.polyval(da[dim], p.polyfit_coefficients)
    return da - fit
    
def do_symmetries(da,
                  do_only = None):
    '''
    Function that will create symmetric and antisymmetric components
    of a DataArray and add a new dimension called "component". Can
    also just output symmetric or antisymmetric component. Doesn't use
    np.flip so should work with dask

    da: xarray.DataArray with dimension 'lat'
    do_only: if do_only = 'sym' or do_only = 'asym', do not add 
    "component" dimension but just output symmetric or asym. part
    '''

    # create flipped (across equator) version of data
    # TODO: This only works if latitude coordinates are symmetric!
    
    da_reversed = da.isel(lat = slice(None,None,-1))
    da_reversed = da_reversed.assign_coords(lat = da.lat)
    
    if do_only == None:
        # need to calculate sym and asym components
        da_sym = 0.5*(da + da_reversed)
        da_asym = 0.5*(da - da_reversed)
        da_component = xr.concat([da_sym, da_asym], "component")
        da_component = da_component.assign_coords({"component":["sym","asym"]})

    else:
        if do_only == 'sym':
            da_component = 0.5*(da + da_reversed)
        elif do_only == 'asym':
            da_component = 0.5*(da - da_reversed)
        else:
            return da

    return da_component

def hpshift(da,
            dim,
            forward):
    '''
    "half-plane shift"
    This function takes some code from the xrft python package
    to permit a np.fft.fftshift analog on chunked DataArrays.
    This capability is needed because xrft does not call fftshift
    when the input data are real, potentially b/c only positive 
    frequencies are output for real dimension (but why not shift
    all remaining dimensions?)
    
    da: xarray.DataArray with at least dimensions listed in dims
    dim: list of dimension names for which to shift coordinates
    forward: bool. whether doing fftshift (True) or ifftshift operation (False)
    '''
    
    # make sure dimensions exist
    for d in dim:
        assert d in da.dims

    # get dimension indices
    dim_idxs = [da.get_axis_num(d) for d in da.dims]
        
    # as in xrft, choose fft module based on whether array is chunked
    if da.chunks:
        fft_mod = dsar.fft
    else:
        fft_mod = np.fft
        
    # unlike in xrft, 'da' here already has frequency coordinate information
    # that we can directly input into (i)fftshift

    # shift frequency information
    coords_shift = []
    for d in dim:
        freq_unshift = da[d]
        # use numpy b/c coords are loaded even with chunks
        if forward:
            freq_shift = np.fft.fftshift(freq_unshift)
        else:
            freq_shift = np.fft.ifftshift(freq_unshift)
        coords_shift.append(freq_shift)
    coords_shift = dict(zip(dim,coords_shift))

    # shift array
    if forward:
        da_shift = fft_mod.fftshift(da, axes = dim_idxs)
    else:
        da_shift = fft_mod.ifftshift(da, axes = dim_idxs)

    # since shift turns DataArray into Dask (or numpy) array, we need
    # to add back coordinate information
    daft = xr.DataArray(da_shift,
                        dims = da.dims,
                        coords = dict(
                            [c for c in da.coords.items() if c[0] not in dim]))

    # now, assign the shifted coordinates
    daft = daft.assign_coords(coords_shift)

    return daft

def kf_filter_box(da,
                  #dim, # remove this b/c only 2D filter
                  filt_dict,
                  wn_dim = 'freq_lon',
                  freq_dim = 'freq_time'):
    '''
    Function for filtering FFT'd data according
    to a box in wavenumber-frequency space defined
    by the limits in bounds. Fourier coefficients
    outside the box will be set to zero.
    
    da: xarray.DataArray - FFT'd data
    wn_dim: string - name of wavenumber dimension in FFT array
    freq_dim: string - name of frequency dimension in FFT array
    filt_dict: dictionary mapping dimensions to filtering range
    {wavetype:{component:{freq_dim:[]}}}
    '''
    # assemble dimensions of mask
    # needs to have all dimensions in dims, in addition to
    # "wavetype" and "component"

    assert('wavetype') in da.dims # require a wavetype dimension
    assert('component') in da.dims # require a component dimension

    # get list of all components for filtering purposes
    comp_all = list(da['component'].values)
         
    # get list of all wavetypes for filtering purposes
    wt_all = list(da['wavetype'].values)
    # do not filter 'no_filt' wavetype
    if 'no_filt' in wt_all:
        wt_all.remove('no_filt')
        
    # iteratively apply filtering
    for wt in wt_all: # select a wavetype
        for c in comp_all: # select a component
            if c in list(filt_dict[wt].keys()):
                # if component is present, do filtering
                assert wn_dim in list(filt_dict[wt][c].keys())
                assert freq_dim in list(filt_dict[wt][c].keys())
                
                kbnd = filt_dict[wt][c][wn_dim]
                fbnd = filt_dict[wt][c][freq_dim]
                
                # a bound of None signifies no wavenumber 
                # bound at that end 
                if kbnd[0] == None:
                    kbnd[0] = -np.inf
                if kbnd[1] == None:
                    kbnd[1] = np.inf

                # minimum frequency bound is 0 (no negative freq)
                if fbnd[0] == None:
                    fbnd[0] = 0 
                if fbnd[1] == None:
                    fbnd[1] = np.inf

                # minimum freq. bound is 0, and 
                # frequency bounds must be in increasing order
                assert fbnd[0] >= 0
                assert fbnd[1] > fbnd[0]
                
                # initialize condition for da.where
                # do not zero out other wavetypes or components
                condition = (da.wavetype != wt)|(da.component != c)

                # check monotonicity along wavenumber bounds
                # if decreasing, then set coeffs to 
                # zero INSIDE region
                if kbnd[1] < kbnd[0]: # set to zero inside
                    #### box filter ####
                    box_cond = \
                    ((da.wavetype == wt)&(da.component == c)&(
                    # wavenumber bounds
                     (da[wn_dim] <= min(kbnd))|(da[wn_dim] >= max(kbnd)))&(
                    # frequency bounds
                     (da[freq_dim] >= min(fbnd))&(da[freq_dim] <= max(fbnd))))
                    
                    #### conjugate box filter ####
                    conj_box_cond = \
                    ((da.wavetype == wt)&(da.component == c)&(
                    # wavenumber bounds
                     (da[wn_dim] <= -max(kbnd))|(da[wn_dim] >= -min(kbnd)))&(
                    # frequency bounds
                     (da[freq_dim] <= -min(fbnd))&(da[freq_dim] >= -max(fbnd))))

                    ### combine conditions ####
                    da = \
                    da.where((condition|box_cond|conj_box_cond),0)
                    
                else: # set to zero outside
                    #### box filter ####
                    box_cond = ((da.wavetype == wt)&(da.component == c)&(
                    # wavenumber bounds
                     (da[wn_dim] >= min(kbnd))&(da[wn_dim] <= max(kbnd)))&(
                    # frequency bounds
                     (da[freq_dim] >= min(fbnd))&(da[freq_dim] <= max(fbnd))))
                    
                    #### conjugate box filter ####
                    conj_box_cond = ((da.wavetype == wt)&(da.component == c)&(
                    # wavenumber bounds
                     (da[wn_dim] <= -min(kbnd))&(da[wn_dim] >= -max(kbnd)))&(
                    # frequency bounds
                     (da[freq_dim] <= -min(fbnd))&(da[freq_dim] >= -max(fbnd))))

                    ### combine conditions ####
                    da = \
                    da.where((condition|box_cond|conj_box_cond),0)
                    
            else:
                # if component is not present in dictionary, zero out that comp
                da = da.where(((da.wavetype != wt)|(da.component != c)),0)
                    
    return da
    
def kf_filter_disp(da,
                   filt_dict,
                   wn_dim = 'freq_lon',
                   freq_dim = 'freq_time',
                   u_shift = 0,
                   kflip = False):

    '''
    Function to filter FFT'd array in wavenumber-
    frequency space according to dispersion relationships
    for "wavetype"

    da: xarray.DataArray
    filt_dict: dictionary mapping dimensions to filtering range
    {wavetype:{component:{freq_dim:[]}}}
    wn_dim: name of wavenumber dimension
    freq_dim: name of frequency dimension (this function assumes 2D 
    wavenumber-freq spectrum)
    u_shift: average background wind speed
    kflip: bool- whether to flip the sign of the wavenumbers before
    calculating dispersion rel'n. Needed to properly identify
    direction of waves after FFT
    '''

    # array needs to have "wavetype" and "component" dimensions

    assert('wavetype') in da.dims # require a wavetype dimension
    assert('component') in da.dims # require a component dimension
    
    # get list of all components for filtering purposes
    comp_all = list(da['component'].values)
         
    # get list of all wavetypes for filtering purposes
    wt_all = list(da['wavetype'].values)
    # do not filter 'no_filt' wavetype
    if 'no_filt' in wt_all:
        wt_all.remove('no_filt')
        
    # get and rescale wavenumber coordinate
    wn_rescale = da[wn_dim]/rearth
    
    for wt in wt_all:
        for c in comp_all:
            if c in list(filt_dict[wt].keys()):
                # if component c is present in filter dict,
                # get equivalent depth bounds 
                # if not, do nothing (assume that kf_filter_box has
                # already zeroed out this component
                hbnd = filt_dict[wt][c]['hbnd']
                # check if filtering is needed
                # this handles the case of box filter only
                if not (None in hbnd):
                    # calculate dispersion relation curve at each hbnd
                    # get wavetype name for calc_dispersion
                    # from prefix of wt string, that is, everything
                    # before _
                    wt_loc = wt.split('_')[0]
                    if kflip:
                        wn_loc = -wn_rescale
                        wn_conj = wn_rescale
                    else:
                        wn_loc = wn_rescale
                        wn_conj = -wn_rescale
                    bound_freqs = calc_dispersion(wt_loc, 
                                                  wn_loc, 
                                                  np.array(hbnd),
                                                  u_shift = u_shift,
                                                  fill_val = 0)
                    bound_freqs_conj = calc_dispersion(wt_loc,
                                                       wn_conj,
                                                       np.array(hbnd),
                                                       u_shift = u_shift,
                                                       fill_val = 0)
                                                         
                    low_freq = bound_freqs[min(hbnd)]
                    high_freq = bound_freqs[max(hbnd)]
                    high_freq_conj = -bound_freqs_conj[min(hbnd)]
                    low_freq_conj = -bound_freqs_conj[max(hbnd)]
                
                    # The dispersion relationships should be set to fill_val when
                    # they are undefined (as for MRG at positive wavenumbers) or
                    # produce negative frequencies (as for ER at positive 
                    # wavenumbers). This means that when da.where is applied at 
                    # these wavenumbers, low_freq == high_freq == fill_val and so
                    # the Fourier coefficients are zeroed out for all frequencies.
                    # I think this means that we do not have to filter wavenumber
                    # at all in this routine

                    # initialize dispersion filter condition for da.where
                    # do not zero out coefficients for other wavetypes or components
                    condition = (da.wavetype != wt)|(da.component != c)
                    
                    #### dispersion relation filter ####
                    disp_cond = ((da.wavetype == wt)&(da.component == c)&(
                    # filter frequency
                    (da[freq_dim] > low_freq)&(da[freq_dim] < high_freq)))

                    #### conjugate dispersion relation filter ####
                    disp_cond_conj = ((da.wavetype == wt)&(da.component == c)&(
                    # filter frequency
                    (da[freq_dim] > low_freq_conj)&(da[freq_dim] < high_freq_conj)))
                    
                    da = da.where((condition|disp_cond|disp_cond_conj),0)

    return da
   
def scale_freq(da,
               dim,
               scales,
               forward):
    '''
    xrft package produces coordinates for wavenumbers and frequencies,
    but their units are not cpd (freq) and unitless (wavenumber), so 
    we need to scale them to work with filtering routine. 

    da: xarray DataArray with at least dim "dim" which should be
    'freq_lon' or 'freq_time' (or 'freq_*')
    dim: list of dimension names to rescale
    scale: dictionary of scales
    forward: whether to do (True) or undo (False) scaling
    '''

    for d in dim:
        assert d in da.dims
        if d in list(scales.keys()):
            scale = scales[d]
        else:
            scale = 1
        # apply scaling and modify da in place
        if forward:
            scaled_freq = da[d] * scale
        else:
            scaled_freq = da[d] / scale
        da = da.assign_coords({d:scaled_freq})
    
    return da

def split_hann_taper(da, dim, taper_frac):
    '''Function that applies a <<split>>
    hanning cosine window to a given dimension of an
    xarray DataArray. This is needed because it is
    not an option in xrft.
    da: xarray.DataArray- data to taper
    dim: str- name of dimension in da to be tapered
    taper_frac: float (between 0 and 1)- fraction of dimension
    to taper. e.g. if taper_frac = 0.1, the first and last 5%
    of the dimension will be tapered to zero.'''
    
    series_length = len(da[dim])
    npts = int(np.rint(taper_frac*series_length)) # total size of taper
    # hanning cosine bell window
    taper = np.hanning(npts)
    series_taper = np.ones(series_length)
    series_taper[0:npts//2+1] = taper[0:npts//2+1]
    series_taper[-npts//2+1:] = taper[npts//2+1:]
    series_taper = xr.DataArray(series_taper, 
                                dims = (dim),
                                coords = {dim:da[dim]})
    da_taper = da*series_taper

    return da_taper

#### Core function ############################################################
               
def kf_filter(data_in, # DataArray 
              spd,
              filter_dict_in,
              time_dim = 'time',
              lon_dim = 'lon',
              ubar = 0, # background ZM U in m/s
              swap_filt = False):

    """Perform space-time spectral decomposition and filter in 
    wavenumber-frequency space
    
    data_in: an xarray DataArray to be analyzed; needs to have 
    (time, lat, lon, wavetype, component) dimensions.

    ubar: scalar zonal-mean background wind; used for
    Doppler-shifting dispersion relations
    
    spd: integer; samples per day
    
    filter_dict: dictionary mapping dimensions to filtering range
    {wavetype:{component:{freq_dim:[]}}}

    swap_filt: whether to swap the 'sym' and 'asym' component keys 
    in filter_dict. This should be set to True for V (meridional wind) only, 
    because the symmetric filters must be used when filtering the 
    antisymmetric component of V and vice versa.
    
    
    Method
    ------
        1. Construct symmetric/antisymmetric array.
        2. Apply taper and detrend in time dimension.
        3. Fourier transform
        4. Flip wavenumbers to get propagation direction 
        5. Filter according to wavenumber/frequency ranges
        6. Filter according to equatorial wave dispersion relationships
        7. Do inverse Fourier transform
        8. Return filtered variable
       
    """
    print('data shape at beginning ', data_in.sizes)

    #pdb.set_trace()
    # make local copy of dictionary 
    filter_dict = copy.deepcopy(filter_dict_in)
    filter_keys = list(filter_dict.keys())
    # the following code does not apply to 'no_filt'
    if 'no_filt' in filter_keys:
        filter_keys.remove('no_filt')
    # swap component keys if necessary
    if swap_filt:
        # iterate over wavetypes
        for ky in filter_keys:
            comp_keys = list(filter_dict[ky].keys())
            if len(comp_keys) == 2:
                # get old keys
                old_sym = filter_dict[ky].pop('sym')
                old_asym = filter_dict[ky].pop('asym')
                # swap
                filter_dict[ky]['sym'] = old_asym
                filter_dict[ky]['asym'] = old_sym
            elif len(comp_keys) == 1:
                if comp_keys[0] == 'sym':
                    filter_dict[ky]['asym'] = filter_dict[ky].pop('sym')
                else:
                    filter_dict[ky]['sym'] = filter_dict[ky].pop('asym')

    print(filter_dict)
    #filt_keys = list(filter_dict[wavetype].keys())
    wavetypes = data_in.wavetype.values

    # ensure wavetype dimension has coordinates
    # (wavetype names) which correspond to wavetypes provided
    # in filter_dict
    for wn in wavetypes:
        assert wn in list(filter_dict.keys())

    # taper data to zero at ends of time dimension
    data = split_hann_taper(data_in, time_dim, 0.1)
    
    # z_fft = xrft.fft(data, 
    #                  dim = ['time','lon'],
    #                  real_dim = 'time')
    odims = data.dims
    ocoords = data.coords
    time_ax = odims.index(time_dim)
    time_len = len(data[time_dim].values)
    lon_ax = odims.index(lon_dim)
    lon_len = len(data[lon_dim].values)
    #pdb.set_trace()
    z_fft = dsar.fft.fftn(data,
                          axes = (lon_ax,time_ax))

    new_dims = list(copy.deepcopy(odims))
    new_coords = dict(copy.deepcopy(ocoords))
    for d in odims:
        if d == time_dim:
            new_dims[time_ax] = 'freq_' + time_dim
            new_coords.pop(d)
            new_coords[new_dims[time_ax]] = \
            np.fft.fftfreq(time_len, 1/spd)
        elif d == lon_dim:
            new_dims[lon_ax] = 'freq_' + lon_dim
            new_coords.pop(d)
            new_coords[new_dims[lon_ax]] = \
            np.fft.fftfreq(lon_len, 1/lon_len)

    #pdb.set_trace()
    z_fft = xr.DataArray(z_fft, dims = tuple(new_dims), coords = new_coords)
    #print(z_fft)
            
    # Now, need to scale wavenumbers and frequencies. xrft produces its
    # own coordinates, but they don't match what we need for filtering
    # z_fft_flip = scale_freq(z_fft, 
    #                         dim = ['freq_lon','freq_time'],
    #                         scales = {'freq_time':86400,'freq_lon':360},
    #                         forward = True)

    # first, do box filter
    z_fft_filter = kf_filter_box(z_fft, 
                                 #dim = ['freq_lon','freq_time'],
                                 filter_dict)

    #pdb.set_trace()
    # do disp rel'n filter
    z_fft_filter = kf_filter_disp(z_fft_filter,
                                  filter_dict,
                                  u_shift = ubar,
                                  kflip = True)
    #pdb.set_trace()
    # undo rescaling
    # z_fft_filter = scale_freq(z_fft_flip, 
    #                           dim = ['freq_lon','freq_time'],
    #                           scales = {'freq_time':86400,'freq_lon':360},
    #                           forward = False)

    # add back lag/spacing attributes for xrft
    # z_fft_filter['freq_lon'] = z_fft_filter.freq_lon.assign_attrs(
    #     z_fft.freq_lon.attrs)
    # z_fft_filter['freq_time'] = z_fft_filter.freq_time.assign_attrs(
    #     z_fft.freq_time.attrs)

    # do inverse fft
    # z_filter = xrft.ifft(z_fft_filter, 
    #                      dim = ['freq_time','freq_lon'],
    #                      real_dim = 'freq_time')    

    z_filter = dsar.fft.ifftn(z_fft_filter,
                              axes = (lon_ax,time_ax)).real.astype('float32')

    z_filter = xr.DataArray(z_filter, dims = odims, 
                            coords = ocoords)

    # collapse component dimension since we are done filtering
    z_filter = z_filter.sum(dim = 'component', keep_attrs = True)

    # for whatever reason, even with the lag and spacing attributes,
    # xrft cannot correctly recover the time coordinates. So, add those
    # manually here
    #z_filter = z_filter.assign_coords({'time':data_in.time})
    
    print('The shape of the filtered space-time array is:', np.shape(z_filter))
    
    return z_filter

