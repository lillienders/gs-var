import netCDF4 as nc
import numpy as np
import xarray as xr
import warnings
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd 
import seaborn as sns
import cmocean as cmo
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from eofs.xarray import Eof
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
sns.set()
from scipy.signal import butter, lfilter

def rmv_clm(dataset):
    """
    Remove climatological mean (i.e., long-term average) from each datavariable in dataset
    Inputs:
    - dataset: xarray dataset, read in from read_file and formated in (time,lat,lon) dimensions
    Outputs:
    - dataset: xarray dataset, formated in (time,lat,lon) dimensions with long-term mean removed
    """
    ds = dataset
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for var in list(ds.data_vars):
            if 'time' in ds[var].dims:
                ds[var] = ds[var] - ds[var].mean(dim = 'time')
        #ds['sla'] = ds['sla'] - np.nanmean(ds['sla'], axis=0)
    dataset = ds
    return (dataset)

def seasonal_detrend(dataset):
    """
    Remove seasonal cycle (here defined as monthly mean) from time series of each variable
    Inputs:
    - dataset: xarray dataset, read in from read_file and formated in (time,lat,lon) dimensions
    Outputs:
    - dataset: xarray dataset, formated in (time,lat,lon) dimensions with seasonal cycle removed
    """
    ds = dataset
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for var in list(ds.data_vars):
            if 'time' in ds[var].dims:
                mn_av = np.zeros((12, len(ds.latitude), len(ds.longitude)))
                mn_av[:] = np.nan
                for mn in range(12):
                    mn_av[mn, :, :] = np.nanmean(ds[var][mn::12, :, :], axis=0)
                mn_av = np.tile(mn_av, (int(len(ds[var]) / 12), 1, 1))
                detrend_temp = ds[var] - mn_av
                ds[var] = detrend_temp
    dataset = ds
    return (dataset)

def seasonal_detrend_ts(dataset):
    """
    Remove seasonal cycle (here defined as monthly mean) from time series of each variable
    Inputs:
    - dataset: xarray dataset, read in from read_file and formated in (time,lat,lon) dimensions
    Outputs:
    - dataset: xarray dataset, formated in (time,lat,lon) dimensions with seasonal cycle removed
    """
    ds = dataset
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mn_av = np.zeros((12))
        mn_av[:] = np.nan
        for mn in range(12):
            mn_av[mn] = np.nanmean(ds[mn::12], axis=0)
        mn_av = np.tile(mn_av, (int(len(ds) / 12)+1))
        mn_av = mn_av[0:len(ds)]
        detrend_temp = ds - mn_av
    dataset = detrend_temp
    return (dataset)
    
def linear_detrend(dataset):
    """
    Remove a linear trend from each grid point
    Inputs:
    - dataset: xarray dataset, read in from read_file and formated in (time,lat,lon) dimensions
    Outputs:
     - dataset: xarray dataset, formated in (time,lat,lon) dimensions with linear trend removed
    """
    ds_temp = dataset
    time_idx = np.linspace(0, len(ds_temp.time) - 1, len(ds_temp.time))
    ds_temp['time'] = time_idx
    ds_poly = ds_temp.polyfit(dim='time', deg=1)
    indices = np.arange(len(ds_temp.time))
    var = 'sla'
    fit_string = var + '_polyfit_coefficients'
    slope = np.array(ds_poly[fit_string][0]).flatten()
    intercept = np.array(ds_poly[fit_string][1]).flatten()
    lin_fit = np.zeros((len(ds_temp.time), len(slope)))
    for loc in range(len(slope)):
        lin_fit[:, loc] = slope[loc] * indices + intercept[loc]
    lin_fit = np.reshape(lin_fit, (len(ds_temp.time), len(ds_temp.latitude), len(ds_temp.longitude)))
    detrended_series = ds_temp[var] - lin_fit
    dataset[var] = detrended_series
    return (dataset)

def gs_index_joyce(dataset):
    """
    Calculate the Locations of Gulf Stream Indices using Terry Joyce's Maximum Standard Deviation Method (Pérez-Hernández and Joyce (2014))
    Inputs:
    - dataset: containing longitude, latitude, sla, sla_std
    Returns:
    - gsi_lon: longitudes of gulf stream index points
    - gsi_lat: latitudes of gulf stream index points
    - std_ts: time series of gulf stream index
    """
    # Load data array, trim longtiude to correct window
    ds = dataset.sel(longitude=slice(290, 308))
    # Coarsen data array to nominal 1 degree (in longitude coordinate)
    crs_factor = int(1 / (ds.longitude.data[1] - ds.longitude.data[
        0]))  # Calculate factor needed to coarsen by based on lon. resolution
    if crs_factor != 0:
        ds = ds.coarsen(longitude=crs_factor,
                        boundary='pad').mean()  # Coarsen grid in longitude coord using xarray built in
    gsi_lon = ds.longitude.data  # Save gsi longitudes in array
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        # Take time average to produce a single field
        mn_std = ds.sla_std.data
        mn_sla = np.nanmean(ds.sla.data, axis=0)
        # Calculate location (latitude) of maximum standard deviation
        gsi_lat_idx = np.nanargmax(mn_std, axis=0)
        gsi_lat = ds.latitude[gsi_lat_idx].data
        sla_flt = ds.sla.data.reshape(len(ds.time), len(ds.latitude), len(ds.longitude))
        temp = np.zeros(len(ds.longitude))
        sla_ts = np.zeros(len(ds.time))
        sla_ts_std = np.zeros(len(ds.time))

        for t in range((len(ds.time))):
            for lon in range(len(ds.longitude)):
                temp[lon] = sla_flt[t, gsi_lat_idx[lon], lon]
            sla_ts[t] = np.nanmean(temp)
            sla_ts_std[t] = np.nanstd(temp)
    return (gsi_lon, gsi_lat, sla_ts, sla_ts_std)


def gs_index(dataset, adt, alt = False):
    # Get Gulf Stream Index Locations (Contour w Max Standard Deviation)
    ds = dataset
    ds = ds.sel(latitude = slice(33,45), longitude = slice(288,310))

    if alt == True:
        x, y, std  = get_contour_info(ds,contour_to_get=get_max_contour(ds, adt,contours_to_try = np.linspace(30,40,11)))
    else:
        #x, y, std = get_contour_info(ds, contour_to_get=get_max_contour(ds, adt,contours_to_try = np.linspace(-45, 0, 46)))
        x, y, std = get_contour_info(ds, contour_to_get=get_max_contour(ds, adt,contours_to_try = np.linspace(-5,12,18)))
    subset_ind = []
    for k in np.linspace(-70, -55, 16):
        subset_ind.append(np.nanargmin(abs((x - 360) - k)))
    #for k in np.linspace(-65,-55,11):
    #    subset_ind.append(np.nanargmin(abs((x - 360) - k)))

    gsi_lon, gsi_lat = x[subset_ind], y[subset_ind]

    sla_ts = np.zeros(len(ds.time))
    sla_ts_std = np.zeros(len(ds.time))
    temp = np.zeros(len(gsi_lon))

    for t in range((len(ds.time))):
        for x in range(len(temp)):
            temp[x] = ds['sla'][
                t, np.nanargmin(abs(ds.latitude.data - gsi_lat[x])), np.nanargmin(abs(ds.longitude.data - gsi_lon[x]))]
        sla_ts[t] = np.nanmean(temp)
        sla_ts_std[t] = np.nanstd(temp)
    return (gsi_lon, gsi_lat, sla_ts, sla_ts_std)

def get_max_contour(dataset, adt, contours_to_try = np.linspace(5, 30, 26)):
    ds = dataset.sel(longitude=slice(289, 308), latitude=slice(36, 45))
    std_contours = np.zeros(len(contours_to_try))
    for c in range(len(contours_to_try)):
        x_temp, y_temp, std_contours[c] = get_contour_info(ds, contour_to_get=int(contours_to_try[c]))
    contour_to_use = int(contours_to_try[np.nanargmax(std_contours)])
    print(contour_to_use)
    return (contour_to_use)
    
def get_contour_info(ds, contour_to_get=40):
    ds = ds.sel(longitude=slice(289, 308), latitude=slice(36, 42))
    x_data = ds.longitude
    y_data = ds.latitude
    std_field = np.nanstd(ds.sla, axis=0)

    contours = plt.contour(x_data, y_data,
                           (ds.adt),
                           levels=[contour_to_get], colors='k', zorder=10, linewidths=0.75)
    x, y = [], []
    for item in contours.collections:
        for i in item.get_paths():
            v = i.vertices
            x_temp = (v[:, 0])
            y_temp = (v[:, 1])
            if len(x_temp) > len(x):
                x = x_temp
                y = y_temp
            temp = 0
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                for lon in range(len(x)):
                    temp = temp + std_field[
                        np.nanargmin(abs(ds.latitude.data - y[lon])), np.nanargmin(abs(ds.longitude.data - x[lon]))]
                std_contour = temp/len(x)
    for i in range(0, len(x)):
        for j in range(i + 1, len(x)):
            if (x[i] - 0.1 < x[j] < x[i] + 0.1):
                if y[j] > y[i]:
                    x[i] = np.nan
                    y[i] = np.nan
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]
    #x_lon = np.array([289.875, 290.375, 290.875, 291.375, 291.875, 292.375, 292.875, 293.375, 293.875, 294.375, 294.875])
    #x_lon = np.array([299.875, 300.375, 300.875, 301.375, 301.875, 302.375, 302.875, 303.375, 303.875, 304.375, 304.875])
    #x_lon = np.array([289.875, 290.875, 291.875, 292.875, 293.875, 294.875, 295.875, 296.875, 297.875,
    #     298.875, 299.875, 300.875, 301.875, 302.875,303.875, 304.875])
    #x_lon = (np.arange(295,306))
    x_lon = (np.arange(290,306))
    check_x_ord = x[0] > x[-1]
    if x[0] > x[-1]:
        x = np.flip(x)
        y = np.flip(y)
    y_lat = np.interp(x_lon,x,y)
    plt.close()
    return (x_lon, y_lat, std_contour)
    
def low_pass_weights(window, cutoff):
    order = ((window - 1) // 2 ) + 1
    nwts = 2 * order + 1
    w = np.zeros([nwts])
    n = nwts // 2
    w[n] = 2 * cutoff
    k = np.arange(1., n)
    sigma = np.sin(np.pi * k / n) * n / (np.pi * k)
    firstfactor = np.sin(2. * np.pi * cutoff * k) / (np.pi * k)
    w[n-1:0:-1] = firstfactor * sigma
    w[n+1:-1] = firstfactor * sigma
    return w[1:-1]

def lanczos_bandpass(array, low_freq, high_freq, window):
    hfw = low_pass_weights(window, 1/(low_freq))
    lfw = low_pass_weights(window, 1/(high_freq))
    weight_high = xr.DataArray(hfw, dims = ['window'])
    weight_low = xr.DataArray(lfw, dims = ['window'])
    lowpass_hf = array.rolling(time = len(hfw), center = True).construct('window').dot(weight_high)
    lowpass_lf = array.rolling(time = len(lfw), center = True).construct('window').dot(weight_low)
    array_filt = lowpass_hf - lowpass_lf
    return(array_filt)
def lagged_corr(var_one, var_two, n_lags = 12):
    corrs  = []
    lags   = np.arange(0,n_lags+1)
    # Negative Lags: Var One Leads 
    for l in range(len(lags)-1):
        corrs.append(np.corrcoef(var_one[:-lags[::-1][l]],var_two[lags[::-1][l]:])[0,1])    
    # Positive Lags First: Var Two Leads 
    for l in range(len(lags)):
        if l == 0:
            corrs.append(np.corrcoef(var_one,var_two)[0,1]) 
        else:
            corrs.append(np.corrcoef(var_one[l:],var_two[:-l])[0,1])
    return(corrs)
    
def get_cis(forcing_pc,gsi_ts,n_lags = 31,jfm = False):
    n_sim  = 10000
    uci = np.zeros(n_lags+1)
    lci = np.zeros(n_lags+1)
    for lag in range(n_lags+1):
        len_ts = len(forcing_pc) - lag
        x_scram = phase_scramble_bs(forcing_pc[lag:-1],n_sim=n_sim)
        y_scram = phase_scramble_bs(gsi_ts[0:(-(lag+1))],n_sim=n_sim)
        if jfm == True:
            y_scram = phase_scramble_bs(gsi_ts[0:(30-(lag+1))],n_sim=n_sim)
        coefs = np.zeros(n_sim)
        for n in range(n_sim):
            coefs[n] = np.corrcoef(x_scram[:,n],y_scram[:,n])[0,1]
        uci[lag] = np.quantile(coefs,0.95)
        lci[lag] = np.quantile(coefs,0.05)
        #uci[lag] = np.nanmean(coefs) + 1.96*(np.nanstd(coefs))
        #lci[lag] = np.nanmean(coefs) - 1.96*(np.nanstd(coefs))
        #print(np.sqrt(len(coefs)))
    uci_r = np.array(uci[::-1][:-1])
    lci_r = np.array(lci[::-1][:-1])
    ci_upper = np.concatenate((uci_r,uci))
    ci_lower = np.concatenate((lci_r,lci))
    if jfm == True:
        ci_upper = np.interp(x,xf,ci_upper)/1.5
        ci_lower = np.interp(x,xf,ci_lower)/1.5
    return(ci_lower,ci_upper)

def phase_scramble_bs(x, n_sim=10000):
    n_frms = len(x)
    if n_frms % 2 == 0:
        n_frms = n_frms - 1
        x = x[0:n_frms]

    blk_sz = int((n_frms - 1) / 2)
    blk_one = np.arange(1, blk_sz + 1)
    blk_two = np.arange(blk_sz + 1, n_frms)

    fft_x = np.fft.fft(x)
    ph_rnd = np.random.random((blk_sz, n_sim))

    ph_blk_one = np.exp(2 * np.pi * 1j * ph_rnd)
    ph_blk_two = np.conj(np.flipud(ph_blk_one))

    fft_x_surr = np.tile(fft_x[:, None], (1, n_sim))
    fft_x_surr[blk_one, :] = fft_x_surr[blk_one] * ph_blk_one
    fft_x_surr[blk_two, :] = fft_x_surr[blk_two] * ph_blk_two

    scrambled = np.real(np.fft.ifft(fft_x_surr, axis=0))
    return scrambled    

def calc_eofs(array, num_modes=3):
    if 'latitude' in array.dims:
        coslat = np.cos(np.deg2rad(array.coords['latitude'].values))
        wgts = np.sqrt(coslat)[..., np.newaxis]
        solver = Eof(array, weights=wgts)
    else:
        solver = Eof(array)
    eofs = np.squeeze(solver.eofs(neofs=num_modes))
    pcs = np.squeeze(solver.pcs(npcs=num_modes, pcscaling=1))
    per_var = solver.varianceFraction()
    eigs = solver.eigenvalues()
    return (eofs, pcs, per_var,eigs)

def lagged_corr(var_one, var_two, n_lags = 12):
    corrs  = []
    lags   = np.arange(0,n_lags+1)
    # Negative Lags: Var One Leads 
    for l in range(len(lags)-1):
        corrs.append(np.corrcoef(var_one[:-lags[::-1][l]],var_two[lags[::-1][l]:])[0,1])    
    # Positive Lags First: Var Two Leads 
    for l in range(len(lags)):
        if l == 0:
            corrs.append(np.corrcoef(var_one,var_two)[0,1]) 
        else:
            corrs.append(np.corrcoef(var_one[l:],var_two[:-l])[0,1])
    return(corrs)

def is_winter(month):
    return (month >= 1) & (month <= 3)
def is_spring(month):
    return (month >= 4) & (month <= 6)
def is_summer(month):
    return (month >= 7) & (month <= 9)
def is_fall(month):
    return (month >= 10) & (month <= 12)
