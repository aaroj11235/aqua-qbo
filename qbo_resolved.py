import xarray as xr
import numpy as np
import kf_filter as kff
from glob import glob
import pdb
import sys
import qbo_diags as qb
#from filter_TEM_E3SM import wave_filter

# constants

a = 6.37123e6 # radius of Earth
rd = 287.058 # specific gas constant of dry air in J/kg*K
cp = 1.00464e3 # specific heat capacity of dry air at constant pressure in J/kg*K 
p_r = 101325 # reference pressure in Pa
g = 9.80665 # acceleration of gravity in m/s^2
h = 7000 # scale height in m for stratosphere
omega = 7.29212e-5 # Earth's rotation rate

# dictionary that describes which component of the perturbation field 
# ['Up','Vp','Wp','THp'] to filter
# e.g. for the MRG wave, the algorithm filters the symmetric component of Vp
sym_dict = {"MRG":["asym","sym","asym","asym"],
            "ER":["sym","asym","sym","sym"],
            "KELVIN":["sym","asym","sym","sym"],
            "IG":[None, None, None, None],
            "SSG":[None, None, None, None],
            "WESTERLY":[None, None, None, None],
            "EASTERLY":[None, None, None, None],
            "MRGb":["asym","sym","asym","asym"],
            "KELVINb":["sym","asym","sym","sym"],
            "IGb":["both","both","both","both"],
            "ERb":["both","both","both","both"]}

# user inputs

case = sys.argv[1]
short_name = sys.argv[2]
start_year = sys.argv[3] #yyyy
end_year = sys.argv[4]# yyyy
wavetype = sys.argv[5]
num_levs = int(sys.argv[6])
spd = int(sys.argv[7])
#kmin = int(sys.argv[8]) 
#pdb.set_trace()
if ',' in sys.argv[8]:
    kmins = sys.argv[8].split(',')
    kmin_s = int(kmins[0])
    kmin_a = int(kmins[1])
    kmin = np.array([kmin_s,kmin_a])
else:
    kmin = int(sys.argv[8])
if ',' in sys.argv[9]:
    kmaxs = sys.argv[9].split(',')
    kmax_s = int(kmaxs[0])
    kmax_a = int(kmaxs[1])
    kmax = np.array([kmax_s,kmax_a])
else:
    kmax = int(sys.argv[9])
#kmax = int(sys.argv[9])

if ',' in sys.argv[10]:
    fmins = sys.argv[10].split(',')
    fmin_s = float(fmins[0])
    fmin_a = float(fmins[1])
    fmin = np.array([fmin_s,fmin_a])
else:
    fmin = float(sys.argv[10])

#fmin = float(sys.argv[10])
#fmax = float(sys.argv[11])
if ',' in sys.argv[11]:
    fmaxs = sys.argv[11].split(',')
    fmax_s = float(fmaxs[0])
    fmax_a = float(fmaxs[1])
    fmax = np.array([fmax_s,fmax_a])
else:
    fmax = float(sys.argv[11])


hmin = int(sys.argv[12])
hmax = int(sys.argv[13])# hbnd cannot be (None, None)
dname = sys.argv[14] # aqua or jra or era

kbnd = [kmin, kmax]
fbnd = [fmin, fmax]
hbnd = [hmin, hmax]

# define data directory (assume save and data directories are same)

data_type = {"aqua":["aqua_data/cam7_aqua/" + case + "/", short_name], "jra":["rnl_data/jra_data/", "jra"],
             "era":["rnl_data/era_data/", "era"]}
overwrite = True # whether to overwrite previous filtered datasets

data_dir = \
    "/glade/derecho/scratch/aaroj/"

data_dir += data_type[dname][0] # adjust for data source

# print inputs

print("data type is: " + dname)
print("start year is: " + start_year)
print("end year is: " + end_year)
print("wavetype is: " + wavetype)
print("number of levels is: ", num_levs)
print("samples per day are: ", spd)

# print filtering parameters

#kbnd = wavetype_dict[wavetype][1]
print("kbnd is: ", kbnd)
#fbnd = wavetype_dict[wavetype][2]
print("fbnd is: ", fbnd)
#hbnd = wavetype_dict[wavetype][3]
print("hbnd is: ", hbnd)

years = list(np.arange(int(start_year), int(end_year) + 1, 1))

#pdb.set_trace()
for i in range(num_levs):
    l = i+1
    if l < 10:
        lev_str = "0" + str(l)
    else:
        lev_str = str(l)
    print("Doing level ", l)
    tem_pt_fpath = data_dir + "filter_data/" + data_type[dname][1] + "_6hr_TEM_level_" + \
            lev_str + "_years_" + start_year + "_" + end_year + \
            "_wt_" + wavetype + ".nc"
    
    try:
        if overwrite:
            raise FileNotFoundError
        ds_6hr_tem = xr.open_dataset(tem_pt_fpath)
    except FileNotFoundError:

        # concatenate perturbation data

        glob_fpth = data_dir + data_type[dname][1] + "_6hr_eq_level_" + \
            lev_str + "_year_*"
        glob_lev = glob(glob_fpth)
        # filter glob according to start and end years
        glob_tot = [gl for gl in glob_lev if int(gl[-7:-3]) in years]
        glob_tot.sort()
        lev_data = []
        for fp in glob_tot:
            ds_lev = xr.open_dataset(fp)
            lev_data.append(ds_lev)
        ds_input = xr.concat(lev_data, dim = "time")
    
        print("Shape of dataset is: ", ds_input.sizes)
        print("Filtering variables")
        #pdb.set_trace()
        if wavetype != "no_filt":

            #pdb.set_trace()            
            if wavetype in list(sym_dict.keys()):
                syms = sym_dict[wavetype]
            else:
                syms = [None, None, None, None]

            print("Filtering Up")
            up = ds_input.U - ds_input.Uzm
            up_filter = kff.kf_filter(up,
                                      spd,
                                      kbnd,
                                      fbnd,
                                      hbnd,
                                      sym = syms[0],
                                      wavetype = wavetype)
            
            print("Filtering Vp")
            vp = ds_input.V - ds_input.V.mean('lon')
            if syms[1] == 'both': # filtering range switch between
                                  # sym and asym for v 
                                  # kbnd should be defined as for
                                  # u, w, theta
                kbnd_v = [np.flip(kb) for kb in kbnd]
                fbnd_v = [np.flip(fb) for fb in fbnd]
            else: # no change for waves which are only sym or only asym
                kbnd_v = kbnd.copy()
                fbnd_v = fbnd.copy()
            #pdb.set_trace()  
            vp_filter = kff.kf_filter(vp,
                                      spd,
                                      kbnd = kbnd_v,
                                      fbnd = fbnd_v,
                                      hbnd = hbnd,
                                      sym = syms[1],
                                      wavetype = wavetype)
            
            print("Filtering Wp")
            wp = ds_input.OMEGA - ds_input.OMEGA.mean('lon')
            wp_filter = kff.kf_filter(wp,
                                      spd,
                                      kbnd,
                                      fbnd,
                                      hbnd,
                                      sym = syms[2],
                                      wavetype = wavetype)
            
            print("Filtering THp")
            thp = ds_input.TH - ds_input.THzm
            thp_filter = kff.kf_filter(thp,
                                       spd,
                                       kbnd,
                                       fbnd,
                                       hbnd,
                                       sym = syms[3],
                                       wavetype = wavetype)


        else:
            print("No filtering")
            up_filter = ds_input.U - ds_input.Uzm
            vp_filter = ds_input.V - ds_input.V.mean('lon')
            wp_filter = ds_input.OMEGA - ds_input.OMEGA.mean('lon')
            thp_filter = ds_input.TH - ds_input.THzm

        #pdb.set_trace()
        #vpthp = ds_filter.Vp*ds_filter.THp
        vpthp = vp_filter*thp_filter
        vpthp = vpthp.assign_attrs(
                units = 'm*K/s',
                long_name = 'zonal mean meridional eddy heat flux')
        #upvp = ds_filter.Up*ds_filter.Vp
        upvp = up_filter*vp_filter
        upvp = upvp.assign_attrs(
                units = 'm^2/s^2',
                long_name = 'zonal mean meridional eddy momentum flux') 
        #upwp = ds_filter.Up*ds_filter.Wp
        upwp = up_filter*wp_filter
        upwp = upwp.assign_attrs(
                units = 'kg/m*s^2',
                long_name = 'zonal mean vertical eddy momentum flux')

        # take zonal means of filtered quantities
        ds_filter = xr.Dataset({'VpTHpzm':vpthp.mean("lon"),
                                'UpVpzm':upvp.mean("lon"),
                                'UpWpzm':upwp.mean("lon"),
                                'Uzm':ds_input.Uzm,
                                'THzm':ds_input.THzm})
        #ds_filter['VpTHpzm'] = vpthp.mean("lon")
        #ds_filter['UpVpzm'] = upvp.mean("lon")
        #ds_filter['UpWpzm'] = upwp.mean("lon")

        # build final Dataset

        #ds_6hr_tem = ds_filter.drop_vars(variables)

        ds_filter.to_netcdf(tem_pt_fpath)


# now, open datasets and concatenate along level dimension
#pdb.set_trace()
tem_file_string = data_dir + "filter_data/" + data_type[dname][1] + \
        "_6hr_TEM_level_*_years_" + start_year + "_" + end_year + \
        "_wt_" + wavetype + ".nc"

tem_files = glob(tem_file_string)
tem_files.sort()
#pdb.set_trace()
ds_tem_concat = xr.open_mfdataset(tem_files)
#pdb.set_trace()
# load into memory
#pdb.set_trace()
ds_tem_concat = ds_tem_concat.compute()
#ds_tem_concat.to_netcdf("/glade/derecho/scratch/aaroj/rnl_data/jra_data/ptest.nc")

# dims are time, lat, lon (check)

# extract variables for EP Flux calculation

#comb_ds = qb.calc_TEM(ds_tem_concat)

ds_tem_concat['lat'] = np.deg2rad(ds_tem_concat.lat)
ds_tem_concat['lev'] = ds_tem_concat.lev*100
uzm = ds_tem_concat.Uzm
thzm = ds_tem_concat.THzm
vpthp_zm = ds_tem_concat.VpTHpzm
upvp_zm = ds_tem_concat.UpVpzm
upwp_zm = ds_tem_concat.UpWpzm
lat = ds_tem_concat.lat
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

# resample to monthly mean data 

comb_ds_mm = comb_ds.resample(time = "ME").mean("time")

# save monthly-mean data to disk

tem_mm_fstring = data_dir + "filter_data/" + data_type[dname][1] + \
        "_mm_TEM_yrs_" + start_year + "_" + end_year + \
        "_wt_" + wavetype + ".nc"

comb_ds_mm.to_netcdf(tem_mm_fstring)


