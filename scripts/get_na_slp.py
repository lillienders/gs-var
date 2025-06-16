import cdsapi

c = cdsapi.Client()

c.retrieve(
    'reanalysis-era5-single-levels-monthly-means',
    {
        'product_type': 'monthly_averaged_reanalysis',
        'variable': 'mean_sea_level_pressure',
        'grid': [0.25, 0.25],  # High resolution; can also use [1.0, 1.0] to reduce file size
        'area': [80, -100, 20, 0],  # North, West, South, East
        'format': 'netcdf',
        'year': [str(y) for y in range(1940, 2024)],
        'month': [
            '01', '02', '03', '04', '05', '06',
            '07', '08', '09', '10', '11', '12'
        ],
        'time': '00:00',
    },
    'era5_north_atlantic_slp_monthly_1940_2023.nc'
)
