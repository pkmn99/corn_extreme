import numpy as np
import pandas as pd
import cartopy.io.shapereader as shpreader
from affine import Affine

"""
Extract the center location of each county
"""
def get_county_latlon(rerun=False):
    if rerun:
        county_shapes = shpreader.Reader('../data/US_county_gis/counties_contiguous.shp')
        state_shapes = shpreader.Reader('../data/US_county_gis/states_contiguous.shp')
        county_rec = list(county_shapes.records())
        county_geo = list(county_shapes.geometries())
        
        fips = [county_rec[f].attributes['FIPS'] for f in range(len(county_rec))]
        lon = [(county_geo[i].bounds[0] + county_geo[i].bounds[2])/2 for i in range(len(county_rec))]
        lat = [(county_geo[i].bounds[1] + county_geo[i].bounds[3])/2 for i in range(len(county_rec))]
        area = [county_rec[f].attributes['AREA'] for f in range(len(county_rec))]
        
        d = {'FIPS': fips,
         'lat': lat,
         'lon':lon,
         'county_area':area}

        df = pd.DataFrame(d, columns=['FIPS', 'lat', 'lon', 'county_area'])

        # Define Affine of 0.5 degree
        a = Affine(0.5,0,-180,0,-0.5,90)
        # get col and row number
        df['col'], df['row'] = ~a * (df['lon'], df['lat']) 
        # need to floor to get integer col and row

        df['col'] = df['col'].apply(np.floor).astype(int)
        df['row'] = df['row'].apply(np.floor).astype(int)

        df.to_csv('../data/result/county_latlon.csv', index=False)
        
        print 'Extracting lat lon for each county done. File county_latlon.csv saved'
    else:
        df = pd.read_csv('../data/result/county_latlon.csv',dtype={'FIPS':str})
    return df
