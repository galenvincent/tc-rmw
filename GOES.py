import numpy as np
import pandas as pd
import xarray as xa
import datetime as dt
import re
import requests
from bs4 import BeautifulSoup
import datetime
import warnings
from pydap.client import open_url
from pydap.cas.urs import setup_session

pd.options.mode.chained_assignment = None  # default='warn'


### UTILITY FUNCTIONS ###

def distance_from_coord(lat1, lat2, lon1, lon2):
    """
    Return distance in KM given a pair of lat-lon coordinates.

    :param lat1: Starting latitude
    :param lat2: Ending latitude
    :param lon1: Starting Longitude
    :param lon2: Ending longitude
    :return: Arc length of great circle
    """
    # https://www.movable-type.co.uk/scripts/latlong.html
    R = 6371
    latDel = lat1 - lat2
    lonDel = lon1 - lon2
    a = (
            np.sin(latDel / 2 * np.pi / 180) ** 2 +
            np.cos(lat1 * np.pi / 180) * np.cos(lat2 * np.pi / 180) *
            np.sin(lonDel / 2 * np.pi / 180) ** 2
    )
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c


def coord_from_distance(lat, lon, distance, ang_step = 45):
    """
    Find the bounding box for the stamp based on center coord and distance

    :param lat: Latitude of stamp center
    :param lon: Longitude of stamp center
    :param distance: Stamp radius
    :return: Dictionary of bounding latitudes and longitudes for stamp
    """
    # https://www.movable-type.co.uk/scripts/latlong.html
    R = 6371
    bearings = list(range(0, 360, ang_step))
    latEnd = [float()] * len(bearings)
    lonEnd = [float()] * len(bearings)
    distance = float(distance)
    ii = 0
    for bearing in bearings:
        latEnd[ii] = np.arcsin(np.sin(lat * np.pi / 180) * np.cos(distance / R) +
                               np.cos(lat * np.pi / 180) * np.sin(distance / R) *
                               np.cos(bearing * np.pi / 180)
                               ) * 180 / np.pi
        lonEnd[ii] = lon + np.arctan2(
            np.sin(bearing * np.pi / 180) * np.sin(distance / R) * np.cos(lat * np.pi / 180),
            np.cos(distance / R) - np.sin(lat * np.pi / 180) * np.sin(latEnd[ii] * np.pi / 180)
        ) * 180 / np.pi
        ii += 1
    return ({
        'latHi': np.max(latEnd),
        'latLo': np.min(latEnd),
        'lonHi': np.max(lonEnd),
        'lonLo': np.min(lonEnd)
    })


def listFD(url, ext=''):
    """
    Parse the files on an html catalog.

    GridSat file names on the NCEI THREDDS server contain the name of the GOES satellite used to capture the imagery.
    This function can be used to parse the catalog for a given month; this allows us to check file names ahead of time
    rather than testing possible names.

    :param url: URL for the page to be parsed, such as a THREDDS catalog for a given month of GridSat-GOES
    :param ext: Limit the returned entries to a given extension
    :return: A list of urls
    """
    page = requests.get(url).text
    soup = BeautifulSoup(page, 'html.parser')
    return [url + '/' +
            node.get('href') for node in soup.find_all('a')
            if node.get('href').endswith(ext)]


def collect_file_names(stampList, url='NCEI'):
    """
    Collect the list of filenames on pages with month/year combinations present in the list of desired stamps.

    :param url: The general URL for NCEI THREDDS catalog of NetCDF files to be downloaded, with $YEAR$ and $MONTH$
                standing in for the year and month folder names.
    :param stampList: A pandas frame containing timestamps for desired stamps.
    :return: A list of netCDF URLS for all files in relevant months.
    """

    if url == 'NCEI':
        url = 'https://www.ncei.noaa.gov/thredds/catalog/satellite/gridsat-goes-full-disk/$YEAR$/$MONTH$/catalog.html'
        option = 'month'
        ext = 'nc'
    elif url == 'MERGEIR':
        url = 'https://disc2.gesdisc.eosdis.nasa.gov/opendap/MERGED_IR/GPM_MERGIR.1/$YEAR$/$DAY$/contents.html'
        option = 'day'
        ext = 'nc4'
    else:
        option = 'month'
        ext = 'nc'

    stampList['YEAR'] = pd.DatetimeIndex(stampList['DATETIME']).year.astype(str)
    if option == 'month':
        stampList['MONTH'] = pd.DatetimeIndex(stampList['DATETIME']).month.astype(str)
    elif option == 'day':
        stampList['DAY'] = pd.DatetimeIndex(stampList['DATETIME']).dayofyear.astype(str)

    files = []
    if option == 'month':
        for row in stampList[['YEAR', 'MONTH']].drop_duplicates().itertuples():
            FDURL = re.sub('\$YEAR\$', row[1], url)
            FDURL = re.sub('\$MONTH\$', row[2].zfill(2), FDURL)

            files = files + listFD(FDURL, ext=ext)
    elif option == 'day':
        for row in stampList[['YEAR', 'DAY']].drop_duplicates().itertuples():
            FDURL = re.sub('\$YEAR\$', row[1], url)
            FDURL = re.sub('\$DAY\$', row[2].zfill(3), FDURL)

            files = files + listFD(FDURL, ext=ext)

    return files


def linear_interp(stampList, timestamp):
    """
    Shift the center of the requested stamp via linear interpolation to account for time of satellite observation.

    GridSat-GOES observations may not occur exactly at the nominal time. This function approximately corrects for the
    motion of TCs between stamps (used for 'centered' stamps only).

    :param stampList: The pandas frame containing the locations of stamp centers at each timestamp
    :param time: Time at which GridSat-GOES data is available
    :return: Data frame row containing interpolated stamp center.
    """
    # Cannot interpolate without additional entries
    if len(stampList) <= 1:
        return stampList

    # Equal case must return here to avoid divide by 0
    if len(stampList[stampList['DATETIME'] == timestamp]) > 0:
        return stampList[stampList['DATETIME'] == timestamp]

    timestamp = pd.to_datetime(timestamp)
    # Handle GridSat-GOES image from before/after start/end of list
    if len(stampList[stampList['DATETIME'] <= timestamp]) == 0:
        last_ts = stampList[stampList['DATETIME'] >= timestamp].iloc[[0]]
        next_ts = stampList[stampList['DATETIME'] >= timestamp].iloc[[1]]
    elif len(stampList[stampList['DATETIME'] >= timestamp]) == 0:
        last_ts = stampList[stampList['DATETIME'] <= timestamp].iloc[[-2]]
        next_ts = stampList[stampList['DATETIME'] <= timestamp].iloc[[-1]]
    else:
        last_ts = stampList[stampList['DATETIME'] <= timestamp].iloc[[-1]]
        next_ts = stampList[stampList['DATETIME'] >= timestamp].iloc[[0]]

    fraction = (
            np.abs(timestamp - last_ts.iloc[0]['DATETIME']) /
            (
                    np.abs(next_ts.iloc[0]['DATETIME'] - timestamp) +
                    np.abs(timestamp - last_ts.iloc[0]['DATETIME'])
            )
    )
    out = last_ts
    for col in ['LAT', 'LON', 'WIND']:
        dif = fraction * (next_ts.iloc[0][col] - last_ts.iloc[0][col])
        out[col] = last_ts.iloc[0][col] + dif

    out['DATETIME'] = timestamp
    out['DATE'] = timestamp.strftime(format='%Y%m%d')
    out['TIME'] = timestamp.strftime(format='%H%M')

    return (out)


class Stamp:
    def __init__(self, data):
        self.data = data

    @classmethod
    def from_list_NCEI(cls, stampList, basin='NAL', mode='centered'):
        "When writing documentation, remember for centered, RADIUS should also include blur radius."
        files = collect_file_names(stampList, url='NCEI')

        # Contruct URLs to desired netcdf files
        stampList['URL'] = ''
        for ii in stampList.index:
            year = str(pd.to_datetime(stampList.at[ii, 'DATETIME']).year)
            month = str(pd.to_datetime(stampList.at[ii, 'DATETIME']).month).zfill(2)
            day = str(pd.to_datetime(stampList.at[ii, 'DATETIME']).day).zfill(2)
            hour = str(pd.to_datetime(stampList.at[ii, 'DATETIME']).hour)
            if len(hour) <= 2:
                hour = hour.zfill(2).ljust(4, '0')
            elif len(hour) == 3:
                hour = hour.zfill(4)

            # Construct call to THREDDS
            if basin == 'NAL':
                goesnums = ['10', '12', '14', '13', '16', '08']
            elif basin == 'ENP':
                goesnums = ['10', '09', '11', '15', '17']

            full_url = 'https://www.ncei.noaa.gov/thredds/dodsC/satellite/' \
                       'gridsat-goes-full-disk/$YEAR$/$MONTH$/GridSat-GOES.' \
                       '$FILE$.v01.nc'

            file = 'goes\d\d.' + year + '.' + month + '.' + day + '.' + hour
            final_file = []
            for line in files:
                if bool(re.search(file, line)) & any(goesnum in line[165:167] for goesnum in goesnums):
                    final_file = final_file + [line]

            full_url = re.sub('\$YEAR\$', year, full_url)
            full_url = re.sub('\$MONTH\$', month, full_url)
            full_url = re.sub('\$DAY\$', day, full_url)
            full_url = re.sub('\$HOUR\$', hour, full_url)

            file = 'goes' + final_file[0][165:167] + '.' + year + '.' + month + '.' + day + '.' + hour
            full_url = re.sub('\$FILE\$', file, full_url)

            stampList.at[ii, 'URL'] = full_url

        # Collect data from THREDDS
        if mode == 'centered':
            stampList['LATLO'] = 0.0
            stampList['LATHI'] = 0.0
            stampList['LONLO'] = 0.0
            stampList['LONHI'] = 0.0
        totalFiles = len(stampList)

        # Collect first stamp separately to initialize arrays
        print('Initializing arrays.')
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            thredds = [xa.open_dataset(stampList.iloc[0]['URL'], decode_cf=True, engine='pydap')] * totalFiles

        ii = stampList.index[0]
        if mode == 'centered':
            stampList.iloc[0] = linear_interp(stampList, thredds[0]['time'].values[0]).iloc[0]
            bound = coord_from_distance(stampList.at[ii, 'LAT'].astype(float),
                                        stampList.at[ii, 'LON'].astype(float),
                                        stampList.at[ii, 'RADIUS'].astype(float))
            stampList.at[ii, 'LATLO'] = bound['latLo']
            stampList.at[ii, 'LATHI'] = bound['latHi']
            stampList.at[ii, 'LONLO'] = bound['lonLo']
            stampList.at[ii, 'LONHI'] = bound['lonHi']

        stamps = [thredds[0]['ch4'].sel(lat=slice(stampList.at[ii, 'LATLO'], stampList.at[ii, 'LATHI']),
                                        lon=slice(stampList.at[ii, 'LONLO'], stampList.at[ii, 'LONHI']),
                                        time=thredds[0]['time'].values[0])] * len(stampList)
        lats = thredds[0].lat.values
        lons = thredds[0].lon.values
        for jj in range(totalFiles):
            print('Opening NetCDF file on THREDDS: {} of {}.'.format(jj + 1, totalFiles))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with xa.open_dataset(stampList.iloc[jj]['URL'], decode_cf=True, engine='pydap') as ds:
                    ii = stampList.index[jj]
                    if mode == 'centered':
                        stampList.iloc[jj] = linear_interp(stampList, ds['time'].values[0]).iloc[0]
                        bound = coord_from_distance(stampList.at[ii, 'LAT'].astype(float),
                                                    stampList.at[ii, 'LON'].astype(float),
                                                    stampList.at[ii, 'RADIUS'].astype(float))
                        stampList.at[ii, 'LATLO'] = bound['latLo']
                        stampList.at[ii, 'LATHI'] = bound['latHi']
                        stampList.at[ii, 'LONLO'] = bound['lonLo']
                        stampList.at[ii, 'LONHI'] = bound['lonHi']

                    print('Downloading stamp {} of {}.'.format(jj + 1, len(stampList)))
                    # Command to pull proper IR slice.
                    latidx = [x for x in range(len(lats)) if lats[x] > stampList.at[ii, 'LATLO'] and
                              lats[x] < stampList.at[ii, 'LATHI']]
                    lonidx = [x for x in range(len(lons)) if lons[x] > stampList.at[ii, 'LONLO'] and
                              lons[x] < stampList.at[ii, 'LONHI']]
                    data = ds['ch4'][0, latidx, lonidx].load(parallel=False)

                    # Adjust metadata
                    data.values += -273.15
                    data.attrs['units'] = 'Celsius'
                    data.attrs['standard_name'] = 'CTTb'
                    data.attrs['long_name'] = 'IR Brightness Cloud Top Temperature'

                    # Convert to dataset
                    data = data.to_dataset(name='temperature')

                    # Center Stamp
                    data = data.combine_first(
                        xa.Dataset({
                            'LATCENTER': ('time', [stampList.at[ii, 'LAT'].astype(float)]),
                            'LONCENTER': ('time', [stampList.at[ii, 'LON'].astype(float)]),
                            'LATSHIFT': ('time', [(data.lat.values[1::2].max() + data.lat.values[1::2].min()) / 2]),
                            'LONSHIFT': ('time', [(data.lon.values[1::2].max() + data.lon.values[1::2].min()) / 2])
                        }, coords={'time': [data.time.values]})
                    )
                    data = data.assign_coords({'lat': np.round(data.lat.values - (data.lat.values[1::2].max() +
                                                                                  data.lat.values[1::2].min()) / 2,
                                                               decimals=2),
                                               'lon': np.round(data.lon.values - (data.lon.values[1::2].max() +
                                                                                  data.lon.values[1::2].min()) / 2,
                                                               decimals=2)})

                    stamps[jj] = data

        data = xa.combine_nested(stamps, concat_dim='time')
        return cls(data)

    @classmethod
    def from_list_MERGEIR(cls, stampList, user, pwd, mode='centered', halfHour=False):
        "When writing documentation, remember for centered, RADIUS should also include blur radius."
        # TODO: Handle TCs moving out of MergeIR domain
        if halfHour:
            timeidx = 1
        else:
            timeidx = 0

        # Contruct URLs to desired netcdf files
        stampList['URL'] = ''
        for ii in stampList.index:
            year = str(pd.to_datetime(stampList.at[ii, 'DATETIME']).year)
            month = str(pd.to_datetime(stampList.at[ii, 'DATETIME']).month).zfill(2)
            day = str(pd.to_datetime(stampList.at[ii, 'DATETIME']).day).zfill(2)
            jday = str(pd.to_datetime(stampList.at[ii, 'DATETIME']).dayofyear).zfill(3)
            hour = str(pd.to_datetime(stampList.at[ii, 'DATETIME']).hour)
            if len(hour) <= 2:
                hour = hour.zfill(2)

            # Construct call to gesdisc
            full_url = 'https://disc2.gesdisc.eosdis.nasa.gov:443/opendap/MERGED_IR/GPM_MERGIR.1/' \
                       '$YEAR$/$JDAY$/merg_$FILE$_4km-pixel.nc4'

            full_url = re.sub('\$YEAR\$', year, full_url)
            full_url = re.sub('\$JDAY\$', jday, full_url)
            full_url = re.sub('\$FILE\$', year + month + day + hour, full_url)

            stampList.at[ii, 'URL'] = full_url

        # Collect data from THREDDS
        if mode == 'centered':
            stampList['LATLO'] = 0.0
            stampList['LATHI'] = 0.0
            stampList['LONLO'] = 0.0
            stampList['LONHI'] = 0.0
        totalFiles = len(stampList)

        # Collect first stamp separately to initialize arrays
        print('Initializing arrays.')
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            session = setup_session(user, pwd, check_url=stampList.iloc[0]['URL'])
            dataset = open_url(stampList.iloc[0]['URL'], session=session)
            time = np.datetime64(
                datetime.datetime.utcfromtimestamp(
                    np.array(dataset.time)[timeidx] * 3600 * 24
                ))

        ii = stampList.index[0]
        if mode == 'centered':
            stampList.iloc[0] = linear_interp(stampList, time).iloc[0]
            bound = coord_from_distance(stampList.at[ii, 'LAT'].astype(float),
                                        stampList.at[ii, 'LON'].astype(float),
                                        stampList.at[ii, 'RADIUS'].astype(float))
            stampList.at[ii, 'LATLO'] = bound['latLo']
            stampList.at[ii, 'LATHI'] = bound['latHi']
            stampList.at[ii, 'LONLO'] = bound['lonLo']
            stampList.at[ii, 'LONHI'] = bound['lonHi']

        lats = np.array(dataset.lat)
        lons = np.array(dataset.lon)
        latidx = [x for x in range(len(lats)) if lats[x] > stampList.at[ii, 'LATLO'] and
                  lats[x] < stampList.at[ii, 'LATHI']]
        lonidx = [x for x in range(len(lons)) if lons[x] > stampList.at[ii, 'LONLO'] and
                  lons[x] < stampList.at[ii, 'LONHI']]
        img = np.array(dataset.Tb[timeidx, min(latidx):max(latidx), min(lonidx):max(lonidx)])
        img = img.reshape(img.shape[1], img.shape[2])
        img[img == -9999] = np.nan
        stamp = xa.Dataset({
            'temperature': (('time', 'lat', 'lon'), [img])
        }, coords={'time': [np.datetime64(datetime.datetime.utcfromtimestamp(
                               np.array(dataset.time)[timeidx]*3600*24))],
                   'lat': lats[min(latidx):max(latidx)],
                   'lon': lons[min(lonidx):max(lonidx)]
        })

        stamps = [stamp] * len(stampList)
        for jj in range(totalFiles):
            print('Opening NetCDF file on Hydrax: {} of {}.'.format(jj + 1, totalFiles))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ds = open_url(stampList.iloc[jj]['URL'], session=session)
                time = np.datetime64(datetime.datetime.utcfromtimestamp(
                           np.array(ds.time)[timeidx]*3600*24))
                ii = stampList.index[jj]
                if mode == 'centered':
                    stampList.iloc[jj] = linear_interp(stampList, time).iloc[0]
                    bound = coord_from_distance(stampList.at[ii, 'LAT'].astype(float),
                                                stampList.at[ii, 'LON'].astype(float),
                                                stampList.at[ii, 'RADIUS'].astype(float))
                    stampList.at[ii, 'LATLO'] = bound['latLo']
                    stampList.at[ii, 'LATHI'] = bound['latHi']
                    stampList.at[ii, 'LONLO'] = bound['lonLo']
                    stampList.at[ii, 'LONHI'] = bound['lonHi']

                print('Downloading stamp {} of {}.'.format(jj + 1, len(stampList)))
                # Command to pull proper IR slice.
                latidx = [x for x in range(len(lats)) if lats[x] > stampList.at[ii, 'LATLO'] and
                          lats[x] < stampList.at[ii, 'LATHI']]
                lonidx = [x for x in range(len(lons)) if lons[x] > stampList.at[ii, 'LONLO'] and
                          lons[x] < stampList.at[ii, 'LONHI']]
                img = np.array(ds.Tb[timeidx, min(latidx):max(latidx), min(lonidx):max(lonidx)])[0, :, :]
                img[img == -9999] = np.nan
                data = xa.Dataset({
                    'temperature': (('time', 'lat', 'lon'), [img])
                }, coords={'time': [np.datetime64(datetime.datetime.utcfromtimestamp(
                    np.array(ds.time)[timeidx] * 3600 * 24))],
                    'lat': lats[min(latidx):max(latidx)],
                    'lon': lons[min(lonidx):max(lonidx)]
                })

                # Adjust metadata
                data.temperature.values += -273.15
                data.temperature.attrs['units'] = 'Celsius'
                data.temperature.attrs['standard_name'] = 'CTTb'
                data.temperature.attrs['long_name'] = 'IR Brightness Cloud Top Temperature'

                # Center Stamp
                data = data.combine_first(
                    xa.Dataset({
                        'LATCENTER': ('time', [stampList.at[ii, 'LAT'].astype(float)]),
                        'LONCENTER': ('time', [stampList.at[ii, 'LON'].astype(float)]),
                        'LATSHIFT': ('time', [(data.lat.values[1::2].max() + data.lat.values[1::2].min()) / 2]),
                        'LONSHIFT': ('time', [(data.lon.values[1::2].max() + data.lon.values[1::2].min()) / 2])
                    }, coords={'time': [data.time.values[0]]})
                )
                data = data.assign_coords({'lat': np.round(data.lat.values - (data.lat.values[1::2].max() +
                                                                              data.lat.values[1::2].min()) / 2,
                                                           decimals=2),
                                           'lon': np.round(data.lon.values - (data.lon.values[1::2].max() +
                                                                              data.lon.values[1::2].min()) / 2,
                                                           decimals=2)})

                stamps[jj] = data

        data = xa.combine_nested(stamps, concat_dim='time')
        return cls(data)

    @classmethod
    def from_filename(cls, file='data/saved_on_disk.nc'):
        data = xa.open_dataset(file)
        return cls(data)

    def recenter(self, shift, stampList):
        '''
        Correct stamp centers for sampling times of GOES

        :param shift: time to push predictions forward in minutes
        :param stampList: hurdat hourly list for TC
        '''
        times = self.data.time.values
        newlats = self.data.LATCENTER.values
        newlons = self.data.LONCENTER.values
        for ii in range(len(times)):
            adjustedCenter = linear_interp(stampList, times[ii] + np.timedelta64(shift, 'm'))
            newlats[ii] = adjustedCenter['LAT'].values[0]
            newlons[ii] = adjustedCenter['LON'].values[0]
        self.data = self.data.assign({
            'LATCENTER': (['time'], newlats),
            'LONCENTER': (['time'], newlons)
        })

        return self

    def save(self, file='data/saved_on_disk.nc'):
        self.data.to_netcdf(file)
