#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 17 13:39:37 2024

@author: aaroj
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import LogFormatter
import pdb
from ps_defaults import *
import string

lstr = string.ascii_lowercase

cmapd = mpl.colormaps['Dark2']  # type: matplotlib.colors.ListedColormap
colorsd = cmapd.colors  # type: list

h = 7000 # scale height in m
p_surf = 101325 # reference pressure in Pa
a = 6.37123e6 # radius of Earth in m
g = 9.80665 # acceleration of gravity in m/s^2



def scale_epf(epfy, epfz, v_coord, fig, ax, h_scale, dslat = 4, dsp = None):
    # get height and width of axis in inches
    
    fig_width, fig_height    = fig.get_size_inches()
    _, _, ax_wsize, ax_hsize = ax.get_position().bounds
    X, Y = fig_width * ax_wsize, fig_height * ax_hsize

    # get vertical scale type
    pscale = ax.get_yscale()

    # downsample if requested
    if(dslat is not None):
        epfy = epfy.isel(lat=slice(dslat, None, dslat))
        epfz = epfz.isel(lat=slice(dslat, None, dslat))
    if(dsp):
        if(pscale == 'linear'):
            epfy = epfy.isel(lev=slice(dsp, None, dsp))
            epfz = epfz.isel(lev=slice(dsp, None, dsp))
        elif(pscale == 'log'):
            idx = np.logspace(0, np.log10(len(epfy.lev)-dsp), num=int(len(epfy.lev)/dsp), base=10)
            idx = np.unique(np.round(idx).astype(int))
            epfy = epfy.isel(lev=idx)

    # get dimensions
    lev       = epfy.lev
    lat        = epfy.lat
    coslat = np.cos(np.deg2rad(lat))
    acoslat    = a*coslat
    p0, p1     = lev.max(), lev.min()
    lat0, lat1 = lev.min(), lev.max()

    if v_coord == 'z':
        # remove density scaling factor
        #rho = (p_surf/(g*h_scale))*np.exp(np.log(100*lev/p_surf))
        #epfy /= rho
        #epfz /= rho

        # transform from log-pressure to pressure coordinate
        Fphi = epfy * (p_surf/100)/lev
        Fp  = epfz * -(p_surf/100)/h_scale
    else:
        Fphi = epfy
        Fp = epfz
    # get EP flux components, i.e. Jucker Eq. 3
    fphi = Fphi / acoslat
    fp   = Fp   / acoslat

    # apply Edmon (1980) scaling, i.e. Jucker Eq. 4
    hFphi = 2*np.pi/g * acoslat**2 * (fphi)
    hFp   = 2*np.pi/g * acoslat**2 * (a * fp)

    # set the type of pressure scaling
    if(pscale == 'linear'): pdiff = p1-p0
    elif(pscale == 'log'):  pdiff = lev * np.log(p1/p0)

    # do scaling
    Fx = hFphi * (X/Y) / ((lat1-lat0) * np.pi/180)
    Fy = hFp * 1/pdiff

    return Fx.transpose("lev","lat"), Fy.transpose("lev","lat"), lev, lat, X
        


def plot_zm_contour(ds_dict,
                    lev_dict,
                    save_dir,
                    plot_dim,
                    variables, 
                    lat_range=[None, None],
                    lev_range=[None, None],
                    overlay_u=False,
                    overlay_contour = False,
                    contour_file = None,
                    overlay_epf = False,
                    overlay_tpp = False,
                    log_p = True,
                    lev_type = 'pressure',
                    share_cbar = False,
                    show_grid = True):
    '''
    

    Parameters
    ----------
    files : dictionary
        DESCRIPTION.
    save_dir : string
        DESCRIPTION.
    variables : list of strings
        DESCRIPTION.
    interpolate : boolean
        DESCRIPTION.
    levs : list
        DESCRIPTION.
    control : boolean
        DESCRIPTION.
    control_num : integer
        DESCRIPTION.
    lat_range : list
        DESCRIPTION.
    lev_range : list
        DESCRIPTION.
    

    Returns
    -------
    None.

    '''
    
    if lev_dict == None:
        lev_dict = lev_dict_default_latp
    num_subplots = plot_dim[0]*plot_dim[1]
    data_keys = list(ds_dict.keys())
    num_files = len(data_keys)
    num_vars = len(variables)
        
        
    if num_files == 1:
        if num_subplots != num_vars:
            raise ValueError(
                f'Number of subplots: {num_subplots}' +\
                    f' and variables: {num_vars} do not match.')
        
        # extract variables
        
        data_dict = {}
        ds = ds_dict[data_keys[0]].sel(
                lat = slice(lat_range[0],lat_range[1]),
                lev = slice(lev_range[0],lev_range[1]))
        lat = ds.lat.values
        #if lev_type == 'z':
        #    lev = -h*np.log(ds.lev.values/(p0/100))
        #else:
        lev = ds.lev.values
        if overlay_u:
            if "U" in ds.variables:
                u = getattr(ds,'U').squeeze()
            elif "Uzm" in ds.variables:
                u = getattr(ds,'Uzm').squeeze()
            else:
                u = np.zeros(np.shape(var))
        if overlay_tpp:
            if "TROP_P" in ds.variables:
                tpp = getattr(ds,'TROP_P').squeeze()
            else:
                tpp = np.zeros(np.shape(lat))
        if overlay_epf:
            epfy = getattr(ds, 'epfy').squeeze()
            epfz = getattr(ds, 'epfz').squeeze()
        #pdb.set_trace()
        for v in variables:
            var = getattr(ds,v).squeeze()
            
            #if v in list(lev_dict.keys()):
            #    var *= lev_dict[v][3]
            
            [lv,lt] = np.meshgrid(lev,lat,indexing='ij')
            data_dict[v] = {"data":var.values,
                            "lev":lv,
                            "lat":lt,
                            "units":var.units,
                            "name":var.long_name}
            if overlay_u:
                data_dict[v]["u"] = u
            if overlay_tpp:
                data_dict[v]["tpp"] = tpp
                data_dict[v]["lat1D"] = lat
            if overlay_epf:
                data_dict[v]["epfy"] = epfy
                data_dict[v]["epfz"] = epfz

        if len(variables) == 1:
            fig, ax = plt.subplots(1,1,layout='constrained',
                                  figsize = (7,6))
            ky = list(data_dict.keys())[0]
            if ky in list(lev_dict.keys()):
                im = ax.contourf(data_dict[ky]["lat"],
                                 data_dict[ky]["lev"],
                                 data_dict[ky]["data"],
                                 lev_dict[ky][0],
                                 cmap = lev_dict[ky][1],
                                 norm = lev_dict[ky][2],
                                 extend = 'both')
                
                if overlay_contour:
                    csc = ax.contour(data_dict[ky]["lat"],
                                     data_dict[ky]["lev"],
                                     data_dict[ky]["data"],
                                     lev_dict[ky][0],
                                     colors = 'black',
                                     linewidths = 1)
                    ax.clabel(csc, 
                              inline = True, 
                              inline_spacing = 0, 
                              fontsize = 12)

                cbar = fig.colorbar(im, ax = ax)
                cbar_ticks = lev_dict[ky][0]
                cbar_ticks_str = []
                for c in cbar_ticks:
                    cnew = str(c).replace('-','\N{MINUS SIGN}')
                    cbar_ticks_str.append(cnew)
                cbar.set_ticks(cbar_ticks,labels = cbar_ticks_str)
                ax.set_title("Zonal Mean " + str(data_dict[ky]["name"]) + ", (" +\
                                       str(data_dict[ky]["units"]) + ")")
            else:
                im = ax.contourf(data_dict[ky]["lat"],
                                 data_dict[ky]["lev"],
                                 data_dict[ky]["data"])
                cbar = fig.colorbar(im, ax = ax)
                ax.set_title("Zonal Mean " + str(data_dict[ky]["name"]) + ", (" +\
                        str(data_dict[ky]["units"]) + ")")
            if overlay_u:
                cs = ax.contour(data_dict[ky]["lat"],
                                data_dict[ky]["lev"],
                                data_dict[ky]["u"],
                                [-20,-15,-10,-5,5,10,15,20],
                                colors='black',
                                linewidths = 1)
                ax.clabel(cs, inline=True,inline_spacing=0, fontsize=12)            
            if overlay_tpp:
                ax.plot(data_dict[ky]["lat1D"],
                        data_dict[ky]["tpp"],
                        color = 'gray')
                        
            ax.invert_yaxis()
            #ax.tick_params(axis = 'both', which = 'major', labelsize = 16)
            #ax.tick_params(axis = 'both', which = 'minor', labelsize = 14)
            ax.set_box_aspect(aspect=1)
            if log_p:
                ax.set_yscale('log')
                ax.yaxis.set_minor_formatter(LogFormatter(minor_thresholds=(2, 0.5)))
            #ax.set_xlim([lat_range[0],lat_range[1]])
            #ax.set_ylim([lev_range[0],lev_range[1]])
            if overlay_epf:
                epfy_mod, epfz_mod, dsp, dslat, X = \
                        scale_epf(data_dict[ky]["epfy"],
                                  data_dict[ky]["epfz"],
                                  lev_type,
                                  fig,
                                  ax,
                                  h)
                # do streamlines plot

                ax.quiver(dslat,
                          dsp,
                          epfy_mod,
                          epfz_mod,
                          color = 'k',
                          width = 0.0075*X)
            if show_grid:
                ax.grid(visible = True, color = 'k', linestyle = 'dotted')
            fig.suptitle(data_keys[0])
            fig.supxlabel("Latitude (Degrees North)")
            fig.supylabel("Pressure (hPa)")
                
            figname = "ZM"
            figname += "_" + variables[0] 
            figname += "_" + data_keys[0].replace(' ','_') + ".png"
            fig.savefig(save_dir + figname)      
        
        else:
            fig, axs = plt.subplots(plot_dim[0],plot_dim[1],
                                    layout='constrained', figsize = (10,8))
            
            axs = axs.ravel()
            for i in range(len(list(axs))):
                ky = list(data_dict.keys())[i]
                if ky in list(lev_dict.keys()):
                    im = axs[i].contourf(data_dict[ky]["lat"],
                                         data_dict[ky]["lev"],
                                         data_dict[ky]["data"],
                                         lev_dict[ky][0],
                                         cmap = lev_dict[ky][1],
                                         norm = lev_dict[ky][2],
                                         extend = 'both')
                    if overlay_contour:
                        csc = axs[i].contour(data_dict[ky]["lat"],
                                             data_dict[ky]["lev"],
                                             data_dict[ky]["data"],
                                             lev_dict[ky][0],
                                             colors = 'black',
                                             linewidths = 1)
                        axs[i].clabel(csc, 
                                      inline = True, 
                                      inline_spacing = 0, 
                                      fontsize = 12)
                    
                    if share_cbar == False:
                        cbar = fig.colorbar(im, ax = axs[i])
                        cbar_ticks = lev_dict[ky][0]
                        cbar_ticks_str = []
                        for c in cbar_ticks:
                            cnew = str(c).replace('-','\N{MINUS SIGN}')
                            cbar_ticks_str.append(cnew)
                        cbar.set_ticks(cbar_ticks,labels = cbar_ticks_str)
                    axs[i].set_title("ZM " + str(data_dict[ky]["name"]) + ", (" +\
                                           str(data_dict[ky]["units"]) + ")")
                else:
                    im = axs[i].contourf(data_dict[ky]["lat"],
                                         data_dict[ky]["lev"],
                                         data_dict[ky]["data"])
                    if share_cbar == False:
                        cbar = fig.colorbar(im, ax = axs[i])
                    axs[i].set_title("ZM " + str(data_dict[ky]["name"]) + ", (" +\
                                           str(data_dict[ky]["units"]) + ")")
                if overlay_u:
                    cs = axs[i].contour(data_dict[ky]["lat"],
                                        data_dict[ky]["lev"],
                                        data_dict[ky]["u"],
                                        [-20,-15,-10,-5,5,10,15,20],
                                        colors='black',
                                        linewidths = 1)
                    axs[i].clabel(cs, inline=True,inline_spacing=0, fontsize=12)   
                if overlay_tpp:
                    axs[i].plot(data_dict[ky]["lat1D"],
                                data_dict[ky]["tpp"],
                                color = 'gray')
                    
                axs[i].invert_yaxis()
                #axs[i].tick_params(axis = 'both', which = 'major', labelsize = 16)
                #axs[i].tick_params(axis = 'both', which = 'minor', labelsize = 14)
                axs[i].set_box_aspect(aspect=1)
                if log_p:   
                    axs[i].set_yscale('log')
                    axs[i].yaxis.set_minor_formatter(LogFormatter(minor_thresholds=(2, 0.5)))
                #axs[i].set_xlim([lat_range[0],lat_range[1]])
                #axs[i].set_ylim([lev_range[0],lev_range[1]])
                if overlay_epf:
                    epfy_mod, epfz_mod, dsp, dslat, X = \
                            scale_epf(data_dict[ky]["epfy"],
                                      data_dict[ky]["epfz"],
                                      lev_type,
                                      fig,
                                      axs[i],
                                      h)
                # do streamlines plot
                    axs[i].quiver(dslat,
                                  dsp,
                                  epfy_mod,
                                  epfz_mod,
                                  color = 'k',
                                  width = 0.0075*X)
                if show_grid:
                    axs[i].grid(visible = True, color = 'k', linestyle = 'dotted')
            fig.suptitle(data_keys[0])
            fig.supxlabel("Latitude (Degrees North)")
            fig.supylabel("Pressure (hPa)")
            
            if share_cbar == True:
                cbar = fig.colorbar(im, ax = axs.tolist())
                # just use levels from first entry in dict
                cbar_ticks = lev_dict[list(data_dict.keys())[0]][0] 
                cbar_ticks_str = []
                for c in cbar_ticks:
                    cnew = str(c).replace('-','\N{MINUS SIGN}')
                    cbar_ticks_str.append(cnew)
                cbar.set_ticks(cbar_ticks,labels = cbar_ticks_str)

            figname = "ZM"
            for vb in range(len(variables)):
                figname += "_" + variables[vb]
            figname += "_" + data_keys[0].replace(' ','_') + ".png"
            fig.savefig(save_dir + figname) 
        
                        
                
    else:
        if num_subplots != num_files:
            raise ValueError(
                f'Number of subplots: {num_subplots} and files: {num_files}'\
                    ' do not match.')
        if num_vars != 1:
            raise ValueError('Number of variables must be 1 (currently '\
                             f'{num_vars}) for multiple files')
    
        # extract variables
        
        data_dict = {}
        
        v = variables[0]
        
        #if not control:        
        for k in data_keys:
            ds = ds_dict[k].sel(lat = slice(lat_range[0],lat_range[1]),
                                lev = slice(lev_range[0],lev_range[1]))
            lat = ds.lat.values
            lev = ds.lev.values
            var = getattr(ds,v).squeeze()
            
            #if v in list(lev_dict.keys()):
            #    var *= lev_dict[v][3]
            
            [lv,lt] = np.meshgrid(lev,lat,indexing='ij')
            data_dict[k] = {"data":var.values,
                            "lev":lv,
                            "lat":lt,
                            "units":var.units,
                            "name":var.long_name}

            # extract u for overlay if desired
            if overlay_u:
                if "U" in ds.variables:
                    u = getattr(ds,'U').squeeze()
                elif "Uzm" in ds.variables:
                    u = getattr(ds,'Uzm').squeeze()
                else:
                    u = np.zeros(np.shape(var))
                data_dict[k]["u"] = u

            if overlay_tpp:
                if "TROP_P" in ds.variables:
                    tpp = getattr(ds,'TROP_P').squeeze()
                else:
                    tpp = np.zeros(np.shape(lat))
                data_dict[k]["tpp"] = tpp
                data_dict[k]["lat1D"] = lat

            if overlay_contour:
                if contour_file == None:
                    data_dict[k]["cdata"] = var.values
                elif type(contour_file) == int: 
                    ds_c = ds_dict[data_keys[contour_file]].sel(
                        lat = slice(lat_range[0],lat_range[1]),
                        lev = slice(lev_range[0],lev_range[1]))
                    # assume that each case has the same grid
                    data_dict[k]["cdata"] = getattr(ds_c,v).squeeze().values
                else:
                    data_dict[k]["cdata"] = np.zeros_like(var.values)
                    
            if overlay_epf:
                epfy = getattr(ds, 'epfy').squeeze()
                epfz = getattr(ds, 'epfz').squeeze()
                data_dict[k]["epfy"] = epfy
                data_dict[k]["epfz"] = epfz
                
                        
        fig, axss = plt.subplots(plot_dim[0],plot_dim[1],figsize = (16,8),
                                 layout = 'compressed')
        # TODO some sort of automatic scaling of figure size according
        # to number of subplots
        
        axs = axss.ravel()
        for i in range(len(list(axs))):
            ky = list(data_dict.keys())[i]
            if v in list(lev_dict.keys()):

                if type(lev_dict[v]) == dict: #add functionality for diff colorbars
                    lev_d = lev_dict[v][i]
                    clevs = lev_dict[v][int(contour_file)]
                else:
                    lev_d = lev_dict[v]
                    clevs = lev_dict[v]
                im = axs[i].contourf(data_dict[ky]["lat"],
                                     data_dict[ky]["lev"],
                                     data_dict[ky]["data"],
                                     levels = lev_d[0],
                                     cmap = lev_d[1],
                                     norm = lev_d[2],
                                     extend = 'both')

                # don't add contour lines to the plot they're taken from
                if ((overlay_contour) and (i != contour_file)): 
                    csc = axs[i].contour(data_dict[ky]["lat"],
                                         data_dict[ky]["lev"],
                                         data_dict[ky]["cdata"],
                                         levels = clevs[0],
                                         colors = 'black',
                                         linewidths = 1)
                    axs[i].clabel(csc, 
                                  inline = True, 
                                  inline_spacing = 0, 
                                  fontsize = 12)

                if type(lev_dict[v]) == dict:
                    cbar = fig.colorbar(im, ax = axs[i])
                    cbar_ticks_str = []
                    cbar_ticks = lev_dict[v][i][0]
                    for c in cbar_ticks:
                        cnew = str(c).replace('-','\N{MINUS SIGN}')
                        cbar_ticks_str.append(cnew)
                    cbar.set_ticks(cbar_ticks,labels = cbar_ticks_str)
                    cbar.ax.tick_params(labelsize=16)
            else:
                im = axs[i].contourf(data_dict[ky]["lat"],
                                     data_dict[ky]["lev"],
                                     data_dict[ky]["data"])
                    
                cbar = fig.colorbar(im, ax = axs[i],extend = "both")
                cbar.ax.tick_params(labelsize=16)
            if overlay_u:
                cs = axs[i].contour(data_dict[ky]["lat"],
                                    data_dict[ky]["lev"],
                                    data_dict[ky]["u"],
                                    [-20,-15,-10,-5,5,10,15,20],
                                    colors='black',
                                    linewidths = 1)
                axs[i].clabel(cs, inline=True,inline_spacing=0, fontsize=12)

            if overlay_tpp:
                axs[i].plot(data_dict[ky]["lat1D"],
                            data_dict[ky]["tpp"],
                            color = 'gray')
                
            axs[i].invert_yaxis()
            #axs[i].yaxis.set_minor_formatter(LogFormatter(minor_thresholds=(2, 0.5)))
            #axs[i].set_box_aspect(aspect=1)
            if log_p:
                axs[i].set_yscale('log')
                axs[i].yaxis.set_minor_formatter(LogFormatter(minor_thresholds=(2, 0.5)))
            axs[i].set_title(ky, {"fontsize":16})
            axs[i].tick_params(axis='both', which = 'both', labelsize=16)
            #axs[i].set_xlim([lat_range[0],lat_range[1]])
            #axs[i].set_ylim([lev_range[0],lev_range[1]])
            if overlay_epf:
                epfy_mod, epfz_mod, dsp, dslat, X = \
                        scale_epf(data_dict[ky]["epfy"],
                                  data_dict[ky]["epfz"],
                                  lev_type,
                                  fig,
                                  axs[i],
                                  h)
                # do streamlines plot
                axs[i].quiver(dslat,
                              dsp,
                              epfy_mod,
                              epfz_mod,
                              color = 'k',
                              width = 0.006*X)
            if show_grid:
                axs[i].grid(visible = True, color = 'k', linestyle = 'dotted')
                
        if ((v in list(lev_dict.keys())) and (type(lev_dict[v]) != dict)):
            cbar = fig.colorbar(im, ax = axss)
            cbar_ticks_str = []
            cbar_ticks = lev_dict[v][0]
            for c in cbar_ticks:
                cnew = str(c).replace('-','\N{MINUS SIGN}')
                cbar_ticks_str.append(cnew)
            cbar.set_ticks(cbar_ticks,labels = cbar_ticks_str)
            cbar.ax.tick_params(labelsize=16)

        fig.suptitle("Zonal Mean " + str(
            data_dict[list(data_dict.keys())[0]]["name"]) + ", (" + \
                         str(data_dict[list(data_dict.keys())[0]]["units"]) + ")",
                     fontsize = 20)
        #else:
        #    fig.suptitle("Zonal Mean " + v)
        fig.supxlabel("Latitude (Degrees)", fontsize = 20)
        fig.supylabel("Pressure (hPa)", fontsize = 20)
        
        
        figname = "ZM_" + v + "_"
        
        for kk in data_keys:
            kk_new = kk.replace(" ", "_")
            if kk != data_keys[-1]:
                figname += kk_new + "_"
            else:
                figname += kk_new
        
        #if control == True:
        #    figname += "control.png"
        #else:
        figname += ".png"
            
        fig.savefig(save_dir + figname)
    plt.close('all')
    return None



def plot_zm_line(ds_dict,
                    save_dir,
                    plot_dim,
                    variables, 
                    share,
                    lat_range=[-90,90]
                    ):
    '''
    

    Parameters
    ----------
    files : dictionary
        DESCRIPTION.
    save_dir : string
        DESCRIPTION.
    variables : list of strings
        DESCRIPTION.
    interpolate : boolean
        DESCRIPTION.
    levs : list
        DESCRIPTION.
    control : boolean
        DESCRIPTION.
    control_num : integer
        DESCRIPTION.
    lat_range : list
        DESCRIPTION.
    

    Returns
    -------
    None.

    '''
    
    
    
    num_subplots = plot_dim[0]*plot_dim[1]
    data_keys = list(ds_dict.keys())
    num_files = len(data_keys)
    num_vars = len(variables)
        
        
    if num_files == 1:
        if num_subplots != num_vars:
            raise ValueError(
                f'Number of subplots: {num_subplots}' + \
                    ' and variables: {num_vars} do not match.')
        if share == True:
            raise ValueError('Plots of different variables cannot share axes.')
        # extract variables
        
        data_dict = {}
        ds = ds_dict[data_keys[0]]
        ds = ds.sel(lat = slice(lat_range[0],lat_range[1])) #subset dataset first
        lat = ds.lat.to_numpy()
            
            
        for v in variables:
            var = getattr(ds,v).squeeze()
            #if v in list(var_dict_zmline.keys()): # should do scaling beforehand
            #    var *= var_dict_zmline[v][2]
            data_dict[v] = {"data":var.to_numpy(),
                            "lat":lat,
                            "units":var.units,
                            "name":var.long_name}
                
                
                        
        fig, axs = plt.subplots(plot_dim[0],plot_dim[1],
                                layout='constrained',
                                figsize = (8,6))

        for a in axs:
            a.set_prop_cycle(color=colorsd)
        if num_subplots == 1:
            axs = np.array([axs])
        else:
            axs = axs.ravel()
        for i in range(len(list(axs))):
            ky = list(data_dict.keys())[i]
            dta = data_dict[ky]["data"]
            lt = data_dict[ky]["lat"]
            axs[i].plot(lt,dta)
            
            axs[i].set_title("Zonal Mean " + data_dict[ky]["name"] + " (" +\
                                       data_dict[ky]["units"] + ")")
            axs[i].set_ylabel(ky + " (" + data_dict[ky]["units"] + ")")
            if ky in list(var_dict_zmline.keys()):
                axs[i].set_yticks(var_dict_zmline[ky])
            
            #axs[i].set_xlim([lat_range[0],lat_range[1]]) # already handled via subsetting
        fig.suptitle(data_keys[0])
        fig.supxlabel("Latitude (Degrees)")
                
        figname = "ZM"
        for vb in range(len(variables)):
            figname += "_" + variables[vb]
        figname += "_" + data_keys[0].replace(' ','_') + ".png"
        fig.savefig(save_dir + figname)    
                        
                
    else:
        if ((num_subplots != num_files) and (share==False)):
            raise ValueError(
                f'Number of subplots: {num_subplots} and files: {num_files}' + \
                    ' do not match.')
        if num_vars > 1:
            raise ValueError('Number of variables must be 1 (currently ' + \
                             f'{num_vars}) for multiple files')
    
        # extract variables
        
        data_dict = {}
        
        v = variables[0]
        
        if not share:        
            for k in data_keys:
                ds = ds_dict[k]
                ds = ds.sel(lat = slice(lat_range[0],lat_range[1]))
                lat = ds.lat.to_numpy()
                var = getattr(ds,v).squeeze()
                #if v in list(var_dict_zmline.keys()):
                #    var *= var_dict_zmline[v][2]
                data_dict[k] = {"data":var.to_numpy(),
                                "lat":lat,
                                "units":var.units,
                                "name":var.long_name}
            
            fig, axs = plt.subplots(plot_dim[0],plot_dim[1],
                                    layout='constrained',
                                    figsize = (10,6))
            for a in axs:
                a.set_prop_cycle(color=colorsd)
            axs = axs.ravel()
            for i in range(len(list(axs))):
                ky = list(data_dict.keys())[i]
                dta = data_dict[ky]["data"]
                lt = data_dict[ky]["lat"]
                axs[i].plot(lt,dta)
                axs[i].set_title(ky)
                #axs[i].set_xlim([lat_range[0],lat_range[1]])
                if v in list(var_dict_zmline.keys()):
                    axs[i].set_yticks(var_dict_zmline[v])
                    
            fig.suptitle("Zonal Mean " + data_dict[data_keys[0]]["name"] + " (" + \
                             data_dict[data_keys[0]]["units"] + ")")
            #fig.supylabel(v + " (" + data_dict[data_keys[0]]["units"] + ")")
            
            fig.supxlabel("Latitude (Degrees North)")
            
            figname = "ZM_" + v 
            
            for kk in data_keys:
                #kk_new = kk.replace(" ", "_")
                figname += "_" + kk.replace(" ", "_")
            figname += ".png"
                
            fig.savefig(save_dir + figname)  
                
        else:   
            for k in data_keys:
                ds = ds_dict[k]
                ds = ds.sel(lat = slice(lat_range[0],lat_range[1]))
                lat = ds.lat.to_numpy()
                var = getattr(ds,v).squeeze()
                #if v in list(var_dict_zmline.keys()):
                #    var *= var_dict_zmline[v][2]
                data_dict[k] = {"data":var.to_numpy(),
                                "lat":lat,
                                "units":var.units,
                                "name":var.long_name}
            
            fig, ax = plt.subplots(1,1,
                                    layout='constrained',
                                    figsize = (10,8))
            
            ax.set_prop_cycle(color=colorsd)
            for ky in data_keys:
                dta = data_dict[ky]["data"]
                lt = data_dict[ky]["lat"]
                ax.plot(lt,dta,label = ky, linewidth = 2)
            #ax.set_xlim([lat_range[0],lat_range[1]])
            ax.legend(loc='upper right', fontsize = 14)
            
            fig.suptitle("Zonal Mean " + data_dict[data_keys[0]]["name"] + " (" + \
                             data_dict[data_keys[0]]["units"] + ")", fontsize = 16)
            #fig.supylabel(v + " (" + data_dict[data_keys[0]]["name"] + ")", fontsize = 16)
            if v in list(var_dict_zmline.keys()):
                ax.set_yticks(var_dict_zmline[v])
            
            ax.tick_params(axis = 'both', labelsize = 14)
            fig.supxlabel("Latitude (Degrees)", fontsize = 14)
            
            figname = "ZM_" + v 
            
            for kk in data_keys:
                #kk_new = kk.replace(" ", "_")
                figname += "_" + kk.replace(" ", "_")
            
            figname += "_share.png"
                
            fig.savefig(save_dir + figname)  
                        
    plt.close('all')
    return None
                        
                
        
def plot_lev_time(ds_dict,
                  lev_dict,
                  save_dir,
                  plot_dim,
                  variables, 
                  in_res,
                  lev_range=[0.1,1000],
                  display_range=True,
                  overlay_u = True,
                  overlay_tpp = False,
                  u_levs = [-30,-20,-10,0,10,20,30],
                  log_p = True):
    '''
    

    Parameters
    ----------
    files : dictionary
        DESCRIPTION.
    save_dir : string
        DESCRIPTION.
    variables : list of strings
        DESCRIPTION.
    interpolate : boolean
        DESCRIPTION.
    levs : list
        DESCRIPTION.
    control : boolean
        DESCRIPTION.
    control_num : integer
        DESCRIPTION.
    lat_range : list
        DESCRIPTION.
    lev_range : list
        DESCRIPTION.
    

    Returns
    -------
    None.

    '''
    
    if lev_dict == None:
        lev_dict = lev_dict_default_timep
    num_subplots = plot_dim[0]*plot_dim[1]
    data_keys = list(ds_dict.keys())
    num_files = len(data_keys)
    num_vars = len(variables)
        
        
    if num_files == 1:
        if num_subplots != num_vars:
            raise ValueError(
                f'Number of subplots: {num_subplots}'\
                    ' and variables: {num_vars} do not match.')
        
        # extract variables
        
        data_dict = {}
        ds = ds_dict[data_keys[0]].sel(lev = slice(lev_range[0],lev_range[1]))
        ds = ds.transpose("lev","time")
        lev = ds.lev.values
        tme = ds.time.values
        time = np.arange(1,len(tme)+1,1)
        if overlay_u:
            if 'U' in ds.variables:
                u = getattr(ds,'U').squeeze()
            elif 'Uzm' in ds.variables:
                u = getattr(ds,'Uzm').squeeze()
            else:
                u = np.zeros(np.shape(var))

        if overlay_tpp:
            if "TROP_P" in ds.variables:
                tpp = getattr(ds,'TROP_P').squeeze()
            else:
                tpp = np.zeros(np.shape(tme)) 
                
        for v in variables:
            var = getattr(ds,v).squeeze()
            #if v in list(lev_dict.keys()):
            #    var *= lev_dict[v][3]
            [lv,tm] = np.meshgrid(lev,time,indexing='ij')
            data_dict[v] = {"data":var.values,
                            "lev":lv,
                            "time":tm,
                            "units":var.units,
                            "name":var.long_name}

            if overlay_u:
                data_dict[v]["u"] = u

            if overlay_tpp:
                data_dict[v]["tpp"] = tpp
                data_dict[v]["time1D"] = time
                
                
        if len(variables)==1:
            fig, ax = plt.subplots(1,1,
                                    layout='constrained',
                                    figsize = (10,6),
                                    )
            
            
            ky = list(data_dict.keys())[0]
            #dta = np.transpose(data_dict[ky][0])
            #udta = np.transpose(data_dict[ky][-1])
            #lv = data_dict[ky][1]
            #tm = data_dict[ky][2]
            if ky in list(lev_dict.keys()):
                #if ky in ['AOA','T','Q','CLOUD','RELHUM','delf','v_adv_term',
                #              'w_adv_term','UTEND_GWDTOT','UTEND_PHYSTOT',
                #              'UTEND_TOT','UTEND_RES','UTEND_CORE',
                #              'UTEND_PHYS_RES']:
                im = ax.contourf(data_dict[ky]["time"],
                                 data_dict[ky]["lev"],
                                 data_dict[ky]["data"],
                                 lev_dict[ky][0],
                                 cmap = lev_dict[ky][1],
                                 norm = lev_dict[ky][2],
                                 extend='both')
                #else:
                #    im = ax.contourf(tm,lv,dta,lev_dict[ky][0],
                #                              cmap = lev_dict[ky][2],
                #                              norm = lev_dict[ky][4])
                #    ax.contour(tm,lv,udta,[-20,0,20],colors='black',
                #                   linewidths = 1)
                cbar = fig.colorbar(im, ax = ax,
                                    orientation='horizontal')
                cbar_ticks = lev_dict[ky][0]
                cbar_ticks_str = []
                for c in cbar_ticks:
                    cnew = str(c).replace('-','\N{MINUS SIGN}')
                    cbar_ticks_str.append(cnew)
                cbar.set_ticks(cbar_ticks,labels = cbar_ticks_str, 
                               rotation = 45)
                cbar.ax.tick_params(labelsize = 16)
                ax.set_title("Zonal Mean " + data_dict[ky]["name"] + ", (" +\
                                           data_dict[ky]["units"] + ")",
                             fontsize = 20)
            else:
                im = ax.contourf(data_dict[ky]["time"],
                                 data_dict[ky]["lev"],
                                 data_dict[ky]["data"])
                cbar = fig.colorbar(im, ax = ax,orientation='horizontal')
                cbar.ax.tick_params(labelsize = 16)
                ax.set_title("Zonal Mean " + data_dict[ky]["name"] + ", (" +\
                        data_dict[ky]["units"] + ")", fontsize = 16)
            if overlay_u:
                cs = ax.contour(data_dict[ky]["time"],
                                data_dict[ky]["lev"],
                                data_dict[ky]["u"],
                                u_levs,
                                colors='black',
                                linewidths = 1)
                ax.clabel(cs, inline=True,inline_spacing=0, fontsize=12)
            if overlay_tpp:
                ax.plot(data_dict[ky]["time1D"],
                        data_dict[ky]["tpp"],
                        color = 'gray', linewidth = 2)
            ax.invert_yaxis()
            if log_p:
                ax.set_yscale('log')
                # show minor ticks for log y axis
                ax.yaxis.set_minor_formatter(LogFormatter(minor_thresholds=(2, 0.5)))
            #ax.set_ylim([lev_range[0],lev_range[1]])
            ax.set_box_aspect(aspect=0.3)
            #fig.suptitle(data_keys[0], fontsize = 20)
            ax.set_xlabel("Time (" + in_res + ")", fontsize = 20)
            ax.set_ylabel("Pressure (hPa)", fontsize = 20)           
            ax.tick_params(axis = 'both', which = 'major', labelsize = 16)
            ax.tick_params(axis = 'both', which = 'minor', labelsize = 14)
            figname = "ZM_MM_"
            figname += variables[0] + "_"
            figname += data_keys[0].replace(" ", "_") + ".png"
            fig.savefig(save_dir + figname)    
            
        else:
            fig, axs = plt.subplots(plot_dim[0],plot_dim[1],
                                    layout='constrained',
                                    figsize = (10,6),
                                    )
            
            axs = axs.ravel()
            for i in range(len(list(axs))):
                ky = list(data_dict.keys())[i]
                #dta = np.transpose(data_dict[ky][0])
                #udta = np.transpose(data_dict[ky][-1])
                #lv = data_dict[ky][1]
                #tm = data_dict[ky][2]
                if ky in list(lev_dict.keys()):
                    #if ky in ['AOA','T','Q','CLOUD','RELHUM','delf',
                    #          'v_adv_term',
                    #          'w_adv_term','UTEND_GWDTOT','UTEND_PHYSTOT',
                    #          'UTEND_TOT','UTEND_RES','UTEND_CORE',
                    #          'UTEND_PHYS_RES']:
                    im = axs[i].contourf(data_dict[ky]["time"],
                                         data_dict[ky]["lev"],
                                         data_dict[ky]["data"],
                                         lev_dict[ky][0],
                                         cmap = lev_dict[ky][1],
                                         norm = lev_dict[ky][2],
                                         extend='both')
                    #axs[i].contour(tm,lv,udta,[-20,0,20],colors='black',
                    #               linewidths = 1)
                    #else:
                    #    im = axs[i].contourf(tm,lv,dta,lev_dict[ky][0],
                    #                          cmap = lev_dict[ky][2],
                    #                          norm = lev_dict[ky][4])
                    #    axs[i].contour(tm,lv,udta,[-20,0,20],colors='black',
                    #               linewidths = 1)
                    cbar = fig.colorbar(im, ax = axs[i],
                                        orientation='horizontal',
                                        shrink=0.75)
                    cbar_ticks = lev_dict[ky][0]
                    cbar_ticks_str = []
                    for c in cbar_ticks:
                        cnew = str(c).replace('-','\N{MINUS SIGN}')
                        cbar_ticks_str.append(cnew)
                    cbar.set_ticks(cbar_ticks,labels = cbar_ticks_str)
                    axs[i].set_title("Zonal Mean " + data_dict[ky]["name"] + ", (" +\
                                           data_dict[ky]["units"] + ")")
                else:
                    im = axs[i].contourf(data_dict[ky]["time"],
                                         data_dict[ky]["lev"],
                                         data_dict[ky]["data"])
                    #axs[i].contour(tm,lv,udta,[-20,0,20],colors='black',
                    #               linewidths = 1)
                    cbar = fig.colorbar(im, ax = axs[i],
                                        orientation='horizontal',
                                        shrink=0.75)
                    axs[i].set_title("Tropical (4S-4N) Zonal Mean " + ky)
                if overlay_u:
                    cs = axs[i].contour(data_dict[ky]["time"],
                                data_dict[ky]["lev"],
                                data_dict[ky]["u"],
                                u_levs,
                                colors='black',
                                linewidths = 1)
                    axs[i].clabel(cs, inline=True,inline_spacing=0, fontsize=12)
                if overlay_tpp:
                    axs[i].plot(data_dict[ky]["time1D"],
                                data_dict[ky]["tpp"],
                                color = 'gray', linewidth = 2)
                axs[i].invert_yaxis()
                if log_p:
                    axs[i].set_yscale('log')
                    axs[i].yaxis.set_minor_formatter(LogFormatter(minor_thresholds=(2, 0.5)))
                #axs[i].set_ylim([lev_range[0],lev_range[1]])
                axs[i].set_box_aspect(aspect=0.3)
                axs[i].set_xlabel("Time (" + in_res + ")")
                axs[i].set_ylabel("Pressure (hPa)")
            fig.suptitle(data_keys[0])
                    
            figname = "ZM_MM_"
            for vb in range(len(variables)):
                if vb == (len(variables)-1):
                    figname += "_" + variables[vb]
                else:
                    figname += "_" + variables[vb] + "_"
            figname += data_keys[0].replace(" ", "_") + ".png"
            fig.savefig(save_dir + figname)            
        
                        
                
    else:
        if num_subplots != num_files:
            raise ValueError(
                f'Number of subplots: {num_subplots} and files: {num_files}'\
                    ' do not match.')
        if num_vars > 1:
            raise ValueError('Number of variables must be 1 (currently '\
                             '{num_vars}) for multiple files')
    
        # extract variables
        
        data_dict = {}
        
        v = variables[0]
        
        #if not control:        
        for k in data_keys:
            ds = ds_dict[k].sel(lev = slice(lev_range[0],lev_range[1]))
            ds = ds.transpose("lev","time")
            lev = ds.lev.values
            tme = ds.time.values
            time = np.arange(1,len(tme)+1,1)
            var = getattr(ds,v).squeeze()
            #if v in list(lev_dict.keys()):
            #    var *= lev_dict[v][3]
            [lv,tm] = np.meshgrid(lev,time,indexing='ij')
            data_dict[k] = {"data":var.values,
                            "lev":lv,
                            "time":tm,
                            "units":var.units,
                            "name":var.long_name}
            # also get u for plotting contours
            if overlay_u:
                if 'U' in ds.variables:
                    u = getattr(ds,'U').squeeze()
                elif 'Uzm' in ds.variables:
                    u = getattr(ds,'Uzm').squeeze()
                else:
                    u = np.zeros(np.shape(var))
                data_dict[k]["u"] = u
                
            if overlay_tpp:
                if "TROP_P" in ds.variables:
                    tpp = getattr(ds,'TROP_P').squeeze()
                else:
                    tpp = np.zeros(np.shape(time))
                data_dict[k]["tpp"] = tpp
                data_dict[k]["time1D"] = time      
                
                        
        fig, axss = plt.subplots(plot_dim[0],plot_dim[1],
                                layout='constrained',
                                figsize = (11,10),
                                )
        axs = axss.ravel()
        for i in range(len(list(axs))):
            ky = list(data_dict.keys())[i]
            #dta = np.transpose(data_dict[ky][0])
            #lv = data_dict[ky][2]
            #tm = data_dict[ky][3]
            #udta = np.transpose(data_dict[ky][1])
            if v in list(lev_dict.keys()):
                #if v in ['AOA','T','Q','CLOUD','RELHUM','delf','v_adv_term',
                #         'w_adv_term','UTEND_GWDTOT','UTEND_PHYSTOT',
                #         'UTEND_TOT','UTEND_RES','UTEND_CORE',
                #         'UTEND_PHYS_RES']:
                im = axs[i].contourf(data_dict[ky]["time"],
                                     data_dict[ky]["lev"],
                                     data_dict[ky]["data"],
                                     lev_dict[v][0],
                                     cmap = lev_dict[v][1],
                                     norm = lev_dict[v][2],
                                     extend = 'both')
                #axs[i].contour(tm,lv,udta,[-20,0,20],colors='black',
                #                   linewidths = 1)
                #else:
                #    im = axs[i].contourf(tm,lv,dta,lev_dict[v][0],
                #                          cmap = lev_dict[v][2],
                #                          norm = lev_dict[v][4])
                #    axs[i].contour(tm,lv,udta,[-20,0,20],colors='black',
                #                   linewidths = 1)
            else:
                im = axs[i].contourf(data_dict[ky]["time"],
                                     data_dict[ky]["lev"],
                                     data_dict[ky]["data"])
                #axs[i].contour(tm,lv,udta,[-20,0,20],colors='black',
                #               linewidths=1)
            if overlay_u:
                cs = axs[i].contour(data_dict[ky]["time"],
                                    data_dict[ky]["lev"],
                                    data_dict[ky]["u"],
                                    u_levs,
                                    colors='black',
                                    linewidths = 1)
            if overlay_tpp:
                axs[i].plot(data_dict[ky]["time1D"],
                            data_dict[ky]["tpp"],
                            color = 'gray', linewidth = 2)
                #axs[i].clabel(cs, inline=True,inline_spacing=0, fontsize=12)
            if display_range:
                lev1 = lv[lv >= lev_range[1]]
                lev_fin = lev1[lev1 <= lev_range[0]]
                lev_low = np.where(lv == lev_fin[-1])[0][0]
                lev_high = np.where(lv == lev_fin[0])[0][0]
                dta_sub = dta[lev_high:lev_low,:]
                ax_max = np.nanmax(dta_sub)
                ax_min = np.nanmin(dta_sub)
                axs[i].set_title(
                    ky + " Max:{max_v:.4f}, Min:{min_v:.4f}".format(
                    max_v = ax_max, min_v = ax_min), fontsize = 16)
            else:
                axs[i].set_title(ky, fontsize = 16)

            axs[i].invert_yaxis()
            if log_p:
                axs[i].set_yscale('log')
                #axs[i].yaxis.set_minor_formatter(LogFormatter(minor_thresholds=(2, 0.5)))

                axs[i].set_yticks([2,5,10,20,50,100])
                axs[i].set_yticklabels(['2', '5', '10', '20', '50', '100'])
            #axs[i].set_ylim([lev_range[0],lev_range[1]])
            #axs[i].set_box_aspect(aspect=0.2)
            axs[i].tick_params(axis = 'both', which = 'major', labelsize = 14)
            axs[i].tick_params(axis = 'both', which = 'minor', labelsize = 14)

            axs[i].annotate(lstr[i] + ")",
                            xy=(0, 1), 
                            xycoords='axes fraction',
                            xytext=(0.5, -0.5), 
                            textcoords='offset fontsize',
                            fontsize='xx-large', 
                            verticalalignment='top', 
                            fontfamily='serif',
                            bbox=dict(facecolor='1', 
                                      edgecolor='none', 
                                      pad=3.0, 
                                      alpha = 1))
            #axs[i].set_xlabel("Time (" + in_res + ")")
            #axs[i].set_ylabel("Pressure (hPa)")
        cbar = fig.colorbar(im, ax =axss)
        if v in list(lev_dict.keys()):
            cbar_ticks = lev_dict[v][0]
            cbar_ticks_str = []
            for c in cbar_ticks:
                cnew = str(c).replace('-','\N{MINUS SIGN}')
                cbar_ticks_str.append(cnew)
            cbar.set_ticks(cbar_ticks,labels = cbar_ticks_str)
        cbar.ax.tick_params(labelsize = 16)
        fig.suptitle("Tropical (4S-4N) Zonal Mean Monthly Mean " + \
                data_dict[list(data_dict.keys())[0]]["name"] + \
                         ", (" + data_dict[list(data_dict.keys())[0]]["units"] + ")",
                     fontsize = 18)
        fig.supxlabel("Time (" + in_res + ")", fontsize = 18)
        fig.supylabel("Pressure (hPa)", fontsize = 18)
        
        
        figname = "ZM_MM_" + v 
        
        for kk in data_keys:
            kk_new = kk.replace(" ", "_")
            figname += "_" + kk_new 
        
        #if control == True:
        #    figname += "control.png"
        #else:
        figname += ".png"
            
        fig.savefig(save_dir + figname)  
    plt.close('all')
    return None
        
            
def plot_time_line(ds_dict,
                   save_dir,
                   plot_dim,
                   variables, 
                   share
                  ):
    return None

def plot_height_line(ds_dict,
                     save_dir,
                     plot_dim,
                     variables, 
                     share
                    ):
    return None
            
            
        
    
