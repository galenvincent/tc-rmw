# COPIED FROM THE ORB GIT REPO - GO THERE FOR ANY UPDATES 

import copy
import gc
import seaborn as sb
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import random
import itertools
import collections
import glob
import re
import tempfile
import time

import pandas as pd
# good to never do chained assignment
pd.set_option('mode.chained_assignment', 'raise')
#pd.set_option('mode.chained_assignment', 'warn') # default
pd.options.display.float_format = '{:,.4f}'.format
pd.set_option('max_columns', 999)
pd.set_option('max_rows', 40)

import torch
from torch import distributions
from torch import nn
import torch.utils.data as utils
from torch.nn import functional as F
from torch import optim
from torch.nn import functional as F
from torch.optim import lr_scheduler
from torch.distributions.multivariate_normal import MultivariateNormal

import sys
sys.path.append("pytorch-generative")
from pytorch_generative import nn as pg_nn
from pytorch_generative.models import base
from pytorch_generative import datasets
from pytorch_generative import models
from pytorch_generative import trainer
from pytorch_generative.trainer import Trainer


def _get_orb_files(file_dir, location_prefix, orb_suffix, min_year=2000, max_year=2005):
    years = [str(i) for i in range(min_year, max_year + 1)]
    files = os.listdir(file_dir)
    rad_files = []
    for f in files:
        is_orb = orb_suffix in f
        year_valid = False
        location_valid = f.startswith(location_prefix)
        for year in years:
            year_valid = year_valid or (year in f)
        if is_orb and year_valid and location_valid:
            rad_files.append(f)
    return rad_files

def _elu_conv_elu(conv, x):
    return F.elu(conv(F.elu(x)))


class PrimaryData:
    '''
    Class for the initial processing of data prior to input to torch.
    '''
    def __init__(self, tc_dir, rad_dir=None, basin='AL',
                 minyear=2000, maxyear=2020,
                 radrange=(0, 400), radby=5):
        self.tc_dir = tc_dir
        self.rad_dir = rad_dir
        if self.rad_dir is None:
            self.rad_dir = self.tc_dir
        self.basin = basin
        self.minyear = minyear
        self.maxyear = maxyear
        self.radrange = radrange
        self.radby = radby
        self.radcols = [str(i) for i in range(self.radrange[0] + self.radby,
                                   self.radrange[1] + self.radby,
                                   self.radby)]
        self.allradcols = [b+a for a,b in itertools.product(['_NE', '_NW', '_SE', '_SW'], self.radcols)]

        self.data = self._collect_data()
        self.nans = self._collect_nan()
        self._nan_total_()
        self.tc_ids = sorted(self.data['ID'].unique())

    def _rad_loader(self, rad_files):
        dfs = []
        for f in rad_files:
            path = f'{self.rad_dir}/{f}'
            storm_id = f.split('_')[0]
            df = pd.read_csv(path, header=0, skiprows=[0, 2])
            df.rename(columns={'radius': 'timestamp'}, inplace=True)
            df['timestamp'] = pd.to_datetime(df['timestamp']).dt.round('min')
            df['ID'] = storm_id
            df.rename(columns=dict([(str(float(i)), str(i)) for i in range(5, 400 + 5, 5)]), inplace=True)
            df.sort_values('timestamp', inplace=True)
            df.reset_index(drop=True, inplace=True)
            dfs.append(df)
        return pd.concat(dfs)

    def _collect_rad(self):
        NE_files = _get_orb_files(self.rad_dir, self.basin, '_radNE.csv',
                                  min_year=self.minyear, max_year=self.maxyear)
        NW_files = _get_orb_files(self.rad_dir, self.basin, '_radNW.csv',
                                  min_year=self.minyear, max_year=self.maxyear)
        SE_files = _get_orb_files(self.rad_dir, self.basin, '_radSE.csv',
                                  min_year=self.minyear, max_year=self.maxyear)
        SW_files = _get_orb_files(self.rad_dir, self.basin, '_radSW.csv',
                                  min_year=self.minyear, max_year=self.maxyear)


        df_NE = self._rad_loader(NE_files)
        df_NE.columns = df_NE.columns.map(lambda x: x + '_NE' if x in self.radcols else x)

        df_NW = self._rad_loader(NW_files)
        df_NW.columns = df_NW.columns.map(lambda x: x + '_NW' if x in self.radcols else x)

        df_SE = self._rad_loader(SE_files)
        df_SE.columns = df_SE.columns.map(lambda x: x + '_SE' if x in self.radcols else x)

        df_SW = self._rad_loader(SW_files)
        df_SW.columns = df_SW.columns.map(lambda x: x + '_SW' if x in self.radcols else x)

        return(df_NE, df_NW, df_SE, df_SW)

    def _collect_nan(self):
        nan_files = _get_orb_files(self.rad_dir, self.basin, '_nan.csv',
                                   min_year=self.minyear, max_year=self.maxyear)

        dfs = []
        for f in nan_files:
            path = f'{self.rad_dir}/{f}'
            storm_id = f.split('_')[0]
            df = pd.read_csv(path, header=0, skiprows=[0, 2])
            df.rename(columns={'radius': 'timestamp'}, inplace=True)
            df['timestamp'] = pd.to_datetime(df['timestamp']).dt.round('min')
            df['ID'] = storm_id
            df.rename(columns=dict([(str(float(i)), str(i)) for i in range(5, 400 + 5, 5)]), inplace=True)
            df.sort_values('timestamp', inplace=True)
            df.reset_index(drop=True, inplace=True)
            dfs.append(df)
        df_nan = pd.concat(dfs)

        return (df_nan)

    def _nan_total_(self):
        self.nans['total'] = self.nans[self.radcols].apply(func=np.mean, axis=1)

    def _collect_tc(self):
        tc_files = _get_orb_files(self.tc_dir, self.basin, '_TCdata.csv',
                                  min_year=self.minyear, max_year=self.maxyear)

        dfs = []
        for f in tc_files:
            path = f'{self.tc_dir}/{f}'
            storm_id = f.split('_')[0]
            df = pd.read_csv(path, header=0)
            df.rename(columns={'TIMESTAMP': 'timestamp'}, inplace=True)
            df['timestamp'] = pd.to_datetime(df['timestamp']).dt.round('min')
            # interpolates latitude and wind speed between snapshots
            # snapshots not every 30 mins -> interpolate needed
            df = df.resample('0.5H', on='timestamp').mean().reset_index()
            df[['LAT', 'LONG', 'WIND']] = df[['LAT', 'LONG', 'WIND']].interpolate()
            # round latitude to tenths place - used to compute storm image area
            df['LAT'] = df['LAT'].round(1)
            df['ID'] = storm_id
            dfs.append(df[['ID', 'timestamp', 'LAT', 'WIND']])
        df_tc = pd.concat(dfs)

        return(df_tc)

    def _collect_data(self):
        df_NE, df_NW, df_SE, df_SW = self._collect_rad()
        df_rad = df_NE.merge(df_NW, how='inner', on=['timestamp', 'ID'], suffixes=('_NE', '_NW'))
        del df_NE, df_NW
        df_rad = df_rad.merge(df_SW, how='inner', on=['timestamp', 'ID'], suffixes=('', '_SW'))
        del df_SW
        df_rad = df_rad.merge(df_SE, how='inner', on=['timestamp', 'ID'], suffixes=('', '_SE'))
        del df_SE

        df_tc = self._collect_tc()
        df_fin = df_rad.merge(df_tc, how='inner', on=['timestamp', 'ID'], suffixes=('', '_tc'))
        del df_tc
        gc.collect()

        return(df_fin)

    def interpolate_lone_pixels_(self):
        for tcid in self.tc_ids:
            idxmap = self.data[self.data['ID'] == tcid].index
            nans = self.data[self.data['ID'] == tcid].isnull().any(axis=1)
            if nans.__len__() == 0:
                continue
            if not nans.any():
                continue

            for idx in [idx + 1 for idx in range(nans.__len__() - 2)]:
                if not nans.iloc[idx]:
                    continue
                else:
                    mappedidx = idxmap[idx]
                    row = self.data.iloc[mappedidx, [self.data.columns.get_loc(c) for c in self.allradcols]].values
                    fillidx = [idx for idx in range(row.__len__()) if np.isnan(row[idx])]
                    for fill in fillidx:
                        if fill % (row.__len__()/4) == 0:
                            row[fill] = row[fill+1]
                        elif fill % (row.__len__()/4) == (row.__len__()/4 - 1):
                            row[fill] = row[fill-1]
                        else:
                            row[fill] = (row[fill-1] + row[fill+1])/2
                self.data.iloc[mappedidx, [self.data.columns.get_loc(c) for c in self.allradcols]] = row

    def interpolate_neighborhood_(self, threshold=.1):
        kern = np.array([[1,2,1],[2,4,2],[1,2,1]])

        for tcid in self.tc_ids:
            idxmap = self.data[self.data['ID'] == tcid].index
            nans = self.data[self.data['ID'] == tcid].isnull().mean(axis=1)
            if nans.__len__() == 0:
                continue
            if not nans.mean() == 0:
                continue

            for idx in [idx + 1 for idx in range(nans.__len__() - 2)]:
                if nans.iloc[idx] == 0:
                    continue
                elif nans.iloc[idx] > threshold:
                    continue
                else:
                    mappedidx = idxmap[idx]
                    row = self.data.iloc[mappedidx, [self.data.columns.get_loc(c) for c in self.allradcols]].values
                    cols = row.__len__()
                    neighborhood = self.data.iloc[(mappedidx-1):(mappedidx+2),
                                                  [self.data.columns.get_loc(c) for c in self.allradcols]].values
                    fillidx = [idx for idx in range(cols) if np.isnan(row[idx])]
                    for fill in fillidx:
                        if fill % (cols / 4) == 0:
                            neighbors = neighborhood[:, fill:(fill+2)]
                            A = np.multiply(neighbors, kern[:, 1:])
                            B = np.multiply(1-np.isnan(neighbors), kern[:, 1:])
                        elif fill % (cols / 4) == (cols / 4 - 1):
                            neighbors = neighborhood[:, (fill-1):(fill+1)]
                            A = np.multiply(neighbors, kern[:, :-2])
                            B = np.multiply(1-np.isnan(neighbors), kern[:, :-2])
                        else:
                            neighbors = neighborhood[:, (fill-1):(fill+2)]
                            A = np.multiply(neighbors, kern)
                            B = np.multiply(1-np.isnan(neighbors), kern)
                        row[fill] = np.nansum(A)/np.sum(B)
                    self.data.iloc[mappedidx, [self.data.columns.get_loc(c) for c in self.allradcols]] = row

    def interpolate_lone_rows_(self):
        for tcid in self.tc_ids:
            idxmap = self.data[self.data['ID'] == tcid].index
            nans = self.data[self.data['ID'] == tcid].isnull().any(axis=1)
            if nans.__len__() == 0:
                continue
            if not nans.any():
                continue

            nans = self.data[self.data['ID'] == tcid].isnull().any(axis=1)
            for idx in [idx + 1 for idx in range(nans.__len__() - 2)]:
                if not nans.iloc[idx]:
                    continue
                elif nans.iloc[idx+1] or nans.iloc[idx-1]:
                    continue
                else:
                    mappedidx = idxmap[idx]
                    self.data.iloc[mappedidx, [self.data.columns.get_loc(c) for c in self.allradcols]] = (
                        self.data[self.allradcols].iloc[mappedidx-1] + self.data[self.allradcols].iloc[mappedidx+1]
                                                                                                         ) / 2

    def _max_resample(self, frame, nanframe, target, threshold=0.0):
        nans = nanframe[nanframe['timestamp'].isin(frame['timestamp'])][self.radcols].max(axis=1)
        options = (nans - nans.min()) <= threshold
        options = options.reset_index(drop=True)

        row_options = frame.reset_index(drop=True).loc[options]
        row_distances = (row_options['timestamp'] - target).abs()

        idx = (row_distances == row_distances.min()).tolist()
        out = row_options.loc[idx]

        return out.iloc[0]

    def _mean_resample(self, frame, nanframe, target, threshold=0.0):
        nans = nanframe[nanframe['timestamp'].isin(frame['timestamp'])][self.radcols].mean(axis=1)
        options = (nans - nans.min()) <= threshold
        options = options.reset_index(drop=True)

        row_options = frame.reset_index(drop=True).loc[options]
        row_distances = (row_options['timestamp'] - target).abs()

        idx = (row_distances == row_distances.min()).tolist()
        out = row_options.loc[idx]

        return out.iloc[0]

    def downscale_(self, resampler, resolution='3H', window='1.5H', threshold=0.0):
        tcids = self.data['ID'].unique()

        downscaled_dfs = []
        downscaled_dfs_nan = []
        for tcid in tcids:
            df = self.data[self.data['ID'] == tcid]
            df_down = df.set_index('timestamp').resample(resolution).mean()

            nan = self.nans[self.nans['ID'] == tcid]

            up = pd.DatetimeIndex(df_down.index.values + pd.Timedelta(window))
            down = pd.DatetimeIndex(df_down.index.values - pd.Timedelta(window))
            bounds = zip(up, down, df_down.index.values)

            dfs = [
                resampler(
                    frame=df[df['timestamp'].between(b, a)],
                    nanframe=nan[nan['timestamp'].between(b, a)],
                    target=c,
                    threshold=threshold
                ) for a, b, c in bounds
            ]
            output_frame = pd.DataFrame(dfs)
            output_frame['target'] = df_down.index
            output_frame['lag'] = output_frame['timestamp'] - output_frame['target']
            downscaled_dfs.append(output_frame)

            nan = pd.merge(output_frame, nan, on=['ID', 'timestamp'])
            nan.drop(list(nan.filter(regex='_|WIND|LAT')), axis=1, inplace=True)
            nan['target'] = df_down.index
            nan['lag'] = nan['timestamp'] - nan['target']
            downscaled_dfs_nan.append(nan)

        self.data = pd.concat(downscaled_dfs).reset_index(drop=True)
        self.nans = pd.concat(downscaled_dfs_nan).reset_index(drop=True)

    def plot_hovmoller(self, tcid, fixrange=(-100,30), len=20, wid=10, nan=False, skip=12, skipx=10):
        if tcid in [str(2000+ii) for ii in range(21)]:
            xticks = [''] * self.radcols.__len__()
            xticks[::skipx] = [str(int(x) - 5) for x in self.radcols[::skipx]]

            yticks = 'auto'

            idxsnan = self.nans['ID'].str.contains(tcid)
            idxs = self.data['ID'].str.contains(tcid)
        else:
            xticks = [''] * self.radcols.__len__()
            xticks[::skipx] = [str(int(x) - 5) for x in self.radcols[::skipx]]

            yticks = [''] * self.nans[self.nans['ID'] == tcid].shape[0]
            yticks[::skip] = self.nans.loc[self.nans['ID'] == tcid]['timestamp'].dt.strftime(
                "%m/%d/%Y, %H:%M:%S").values[::skip]

            idxsnan = self.nans['ID'] == tcid
            idxs = self.data['ID'] == tcid

        if nan:
            fig, axs = plt.subplots(2, 3)
            fig.set_size_inches(wid*1.5, len)
            pos = [1, 0, 4, 3]

            for ii in [2, 5]:
                sb.heatmap(self.nans.loc[idxsnan][self.radcols].values,
                           ax=axs.flat[ii], square=True, vmin=0, vmax=.2,
                           xticklabels=xticks, yticklabels=yticks)
                axs.flat[ii].set_title('Nan Fraction')
        else:
            fig, axs = plt.subplots(2, 2)
            fig.set_size_inches(wid, len)
            pos = [1, 0, 3, 2]
        vmin = fixrange[0]
        vmax = fixrange[1]
        names = ['NE', 'NW', 'SE', 'SW']
        for ii in range(4):
            sb.heatmap(self.data.loc[idxs].filter(regex=names[ii]).values,
                       ax=axs.flat[pos[ii]], square=True,
                       vmin=vmin, vmax=vmax,
                       xticklabels=xticks, yticklabels=yticks)
            axs.flat[pos[ii]].set_title(f'{names[ii]}')
        plt.tight_layout()
        plt.show()

    def plot_lag(self, tcid):
        plotable = self.data[self.data['ID'] == tcid][['timestamp', 'lag']]
        fig, axs = plt.subplots(1, 1)
        fig.set_size_inches(15, 5)
        axs.plot(plotable.timestamp, plotable.lag.astype('timedelta64[m]'))
        axs.set_xlabel('Time')
        axs.set_ylabel('Lag (minutes)')
        axs.set_title(tcid + ' profile lag due to missingness')

    def plot_nan(self, tcid):
        plotable = self.nans[self.nans['ID'] == tcid][['timestamp', 'total']]
        fig, axs = plt.subplots(1, 1)
        fig.set_size_inches(15, 5)
        axs.plot(plotable.timestamp, plotable.total)
        axs.set_xlabel('Time')
        axs.set_ylabel('Average Missingness [%]')
        axs.set_title(tcid + ' average missingness over all radii')


class StructuralTrajectories:
    '''
    Class for creating the input tensors for the deep learning models.
    '''
    def __init__(self, primary):
        self.data = None
        self.basin = primary.basin
        self.minyear = primary.minyear
        self.maxyear = primary.maxyear
        self.radrange = primary.radrange
        self.radby = primary.radby
        self.radcols = primary.radcols
        self.allradcols = primary.allradcols
        self.tc_ids = primary.tc_ids

    def radcol(self, affix):
        return [i + affix for i in self.radcols]

    def _construct_storm(self, tcid, primary, length):
        rad_seq = None
        feat_seq = None
        time_seq = None

        # sorted dataframe with just single storm
        storm_df = primary.data.loc[primary.data['ID'] == tcid, :].sort_values('timestamp')

        NE_mtx = storm_df[self.radcol('_NE')].values
        NW_mtx = storm_df[self.radcol('_NW')].values
        SE_mtx = storm_df[self.radcol('_SE')].values
        SW_mtx = storm_df[self.radcol('_SW')].values

        wind_vec = storm_df['WIND'].values
        wind_diff_vec = storm_df['WIND'].diff().values

        times_vec = storm_df['timestamp'].values
        target_vec = storm_df['target'].values

        # iterate over all `lead` row images
        assert (NE_mtx.shape == NW_mtx.shape
                and NW_mtx.shape == SE_mtx.shape
                and SE_mtx.shape == SW_mtx.shape)
        for jj in range(NE_mtx.shape[0] - length):
            # create "slices" of 60 rows
            s_NE = NE_mtx[jj:(jj + length)]
            s_NW = NW_mtx[jj:(jj + length)]
            s_SE = SE_mtx[jj:(jj + length)]
            s_SW = SW_mtx[jj:(jj + length)]

            s_wind = wind_vec[jj:(jj + length)]
            s_wind_diff = wind_diff_vec[jj:(jj + length)]

            s_times = times_vec[jj:(jj + length)]
            s_target = target_vec[jj:(jj + length)]

            # exclude any images with nulls
            if (np.isnan(s_NE).any()
                    or np.isnan(s_NW).any()
                    or np.isnan(s_SE).any()
                    or np.isnan(s_SW).any()
                    or np.isnan(s_wind).any()
                    or np.isnan(s_wind_diff).any()
            ):
                continue

            # stack into proper shape
            s_rad = np.concatenate([np.expand_dims(s_NE, 2),
                                    np.expand_dims(s_NW, 2),
                                    np.expand_dims(s_SE, 2),
                                    np.expand_dims(s_SW, 2)], axis=2)
            s_feat = np.concatenate([np.expand_dims(s_wind, 1),
                                     np.expand_dims(s_wind_diff, 1)], axis=1)
            s_time = np.concatenate([np.expand_dims(s_times, 1),
                                     np.expand_dims(s_target, 1)], axis=1)

            if rad_seq is None:
                rad_seq = np.expand_dims(s_rad, 3)
            else:
                rad_seq = np.concatenate([rad_seq, np.expand_dims(s_rad, 3)], axis=3)

            if feat_seq is None:
                feat_seq = np.expand_dims(s_feat, 2)
            else:
                feat_seq = np.concatenate([feat_seq, np.expand_dims(s_feat, 2)], axis=2)

            if time_seq is None:
                time_seq = np.expand_dims(s_time, 2)
            else:
                time_seq = np.concatenate([time_seq, np.expand_dims(s_time, 2)], axis=2)

        # Return the resulting sequence for the TC
        if rad_seq is None or feat_seq is None or time_seq is None:
            return
        else:
            return {'profiles': np.transpose(rad_seq, axes=(3, 2, 0, 1)),
                    'features': np.transpose(feat_seq, axes=(2, 1, 0)),
                    'times': np.transpose(time_seq, axes=(2, 1, 0))}

    def ingest_primary_(self, primary, length):
        self.data = {tcid: self._construct_storm(tcid, primary, length) for tcid in self.tc_ids}


class QuadDataset(torch.utils.data.Dataset):
    '''
    Custom pytorch dataset which references the tcdata
    '''

    def __init__(self, primary, length, forecast=3):
        self.trajectory_librarian = StructuralTrajectories(primary)
        self.trajectory_librarian.ingest_primary_(primary, length)
        self.tensordict = self.trajectory_librarian.data
        self.forecast = length - forecast
        self.length = length
        self.tc_ids = self.trajectory_librarian.tc_ids

        for tcid in self.tensordict.keys():
            if self.tensordict[tcid] is not None:
                self.tensordict[tcid]['profiles'] = torch.from_numpy(
                    self.tensordict[tcid]['profiles']
                ).type(torch.Tensor)
                self.tensordict[tcid]['features'] = torch.from_numpy(
                    self.tensordict[tcid]['features']
                ).type(torch.Tensor)

        self.tc_ids = [
            tcid for tcid in self.tc_ids if self.tensordict[tcid] is not None
        ]

        sizes = {
            tcid: self.tensordict[tcid]['profiles'].shape[0] for tcid in self.tc_ids
        }

        self.tc_ids.sort()
        self.sizes = [sizes[tcid] for tcid in self.tc_ids]
        self.starts = np.cumsum(self.sizes)

    def __getitem__(self, index):
        '''
        Returns the four quadrants at index, feature history
        '''
        tc_index = np.min(np.where(self.starts > index))
        tc_id = self.tc_ids[np.min(np.where(self.starts > index))]
        new_index = index - self.starts[tc_index] + self.sizes[tc_index]
        sequence = self.tensordict[tc_id]['profiles'][new_index, :, 0:self.forecast, :]

        features = self.tensordict[tc_id]['features'][new_index, :, 0:self.forecast]

        response = np.array(self.tensordict[tc_id]['features'][new_index, 0, self.length-1])

        return sequence, features, response

    def getmeta(self, index):
        tc_index = np.min(np.where(self.starts > index))
        tc_id = self.tc_ids[np.min(np.where(self.starts > index))]
        new_index = index - self.starts[tc_index] + self.sizes[tc_index]

        return tc_id, new_index, self.tensordict[tc_id]['times'][new_index, :, :]

    def __len__(self):
        return sum(self.sizes)


class ResidualBlock(nn.Module):
    """Residual block with a gated activation function."""

    def __init__(self, n_channels):
        """Initializes a new ResidualBlock.
        Args:
            n_channels: The number of input and output channels.
        """
        super().__init__()
        self._input_conv = nn.Conv2d(
            in_channels=n_channels, out_channels=n_channels, kernel_size=2, padding=1
        )
        self._output_conv = nn.Conv2d(
            in_channels=n_channels,
            out_channels=2 * n_channels,
            kernel_size=2,
            padding=1,
        )
        self._activation = pg_nn.GatedActivation(activation_fn=nn.Identity())

    def forward(self, x):
        _, c, h, w = x.shape
        out = _elu_conv_elu(self._input_conv, x)[:, :, :h, :w]
        out = self._activation(self._output_conv(out)[:, :, :h, :w])
        return x + out


class PixelSNAILBlock(nn.Module):
    """Block comprised of a number of residual blocks plus one attention block.
    Implements Figure 5 of [1].
    """

    def __init__(
        self,
        n_channels,
        input_img_channels=1,
        n_residual_blocks=2,
        attention_key_channels=4,
        attention_value_channels=32,
    ):
        """Initializes a new PixelSnailBlock instance.
        Args:
            n_channels: Number of input and output channels.
            input_img_channels: The number of channels in the original input_img. Used
                for the positional encoding channels and the extra channels for the key
                and value convolutions in the attention block.
            n_residual_blocks: Number of residual blocks.
            attention_key_channels: Number of channels (dims) for the attention key.
            attention_value_channels: Number of channels (dims) for the attention value.
        """
        super().__init__()

        def conv(in_channels):
            return nn.Conv2d(in_channels, out_channels=n_channels, kernel_size=1)

        self._residual = nn.Sequential(
            *[ResidualBlock(n_channels) for _ in range(n_residual_blocks)]
        )
        self._attention = pg_nn.CausalAttention(
            in_channels=n_channels + 2 * input_img_channels,
            embed_channels=attention_key_channels,
            out_channels=attention_value_channels,
            mask_center=True,
            extra_input_channels=input_img_channels,
        )
        self._residual_out = conv(n_channels)
        self._attention_out = conv(attention_value_channels)
        self._out = conv(n_channels)

    def forward(self, x, input_img):
        """Computes the forward pass.
        Args:
            x: The input.
            input_img: The original image only used as input to the attention blocks.
        Returns:
            The result of the forward pass.
        """
        res = self._residual(x)
        pos = pg_nn.image_positional_encoding(input_img.shape).to(res.device)
        attn = self._attention(torch.cat((pos, res), dim=1), input_img)
        res, attn = (
            _elu_conv_elu(self._residual_out, res),
            _elu_conv_elu(self._attention_out, attn),
        )
        return _elu_conv_elu(self._out, res + attn)


class PixelSNAIL(base.AutoregressiveModel):
    """The PixelSNAIL model.
    Unlike [1], we implement skip connections from each block to the output.
    We find that this makes training a lot more stable and allows for much faster
    convergence.
    """

    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        n_channels=64,
        n_pixel_snail_blocks=8,
        n_residual_blocks=2,
        attention_key_channels=4,
        attention_value_channels=32,
        sample_fn=None,
    ):
        """Initializes a new PixelSNAIL instance.
        Args:
            in_channels: Number of input channels.
            out_channels: Number of output_channels.
            n_channels: Number of channels to use for convolutions.
            n_pixel_snail_blocks: Number of PixelSNAILBlocks.
            n_residual_blocks: Number of ResidualBlock to use in each PixelSnailBlock.
            attention_key_channels: Number of channels (dims) for the attention key.
            attention_value_channels: Number of channels (dims) for the attention value.
            sample_fn: See the base class.
        """
        super().__init__(sample_fn)
        self._input = pg_nn.CausalConv2d(
            mask_center=True,
            in_channels=in_channels,
            out_channels=n_channels,
            kernel_size=3,
            padding=1,
        )
        self._pixel_snail_blocks = nn.ModuleList(
            [
                PixelSNAILBlock(
                    n_channels=n_channels,
                    input_img_channels=in_channels,
                    n_residual_blocks=n_residual_blocks,
                    attention_key_channels=attention_key_channels,
                    attention_value_channels=attention_value_channels,
                )
                for _ in range(n_pixel_snail_blocks)
            ]
        )
        self._output = nn.Sequential(
            nn.Conv2d(
                in_channels=n_channels, out_channels=n_channels // 2, kernel_size=1
            ),
            nn.Conv2d(
                in_channels=n_channels // 2, out_channels=out_channels, kernel_size=1
            ),
        )
        self._device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def forward(self, x):
        input_img = x
        x = self._input(x)
        for block in self._pixel_snail_blocks:
            x = x + block(x, input_img)
        return self._output(x)


class TrainerPixelQuad(Trainer):
    """
    An object which encapsulates the training and evaluation loop, simply extending the Trainer class.
    """

    def __init__(
            self, model, loss_fn, optimizer, train_loader, eval_loader,
            lr_scheduler=None, clip_grad_norm=None, skip_grad_norm=None,
            sample_epochs=None, sample_fn=None, log_dir=None,
            save_checkpoint_epochs=1, n_gpus=0, device_id=None
    ):
        super().__init__(
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            train_loader=train_loader,
            eval_loader=eval_loader,
            lr_scheduler=lr_scheduler,
            clip_grad_norm=clip_grad_norm,
            skip_grad_norm=skip_grad_norm,
            sample_epochs=sample_epochs,
            sample_fn=sample_fn,
            log_dir=log_dir,
            save_checkpoint_epochs=save_checkpoint_epochs,
            n_gpus=n_gpus,
            device_id=device_id
        )

    def interleaved_train_and_eval(self, max_epochs, restore=True):
        """Trains and evaluates (after each epoch).
        Args:
            max_epochs: Maximum number of epochs to train for.
            restore: Wether to continue training from an existing checkpoint in
                self.log_dir.
        """
        if restore:
            try:
                self.restore_checkpoint()
            except FileNotFoundError:
                pass  # No checkpoint found in self.log_dir; train from scratch.

        for _ in range(max_epochs - self._epoch):
            start_time = time.time()

            if self._epoch % 1 == 0:
                torch.save(self.model.state_dict(), self._path(f"model_state_{self._epoch}"))

            # Train.
            for i, batch in enumerate(self.train_loader):
                batch = batch if isinstance(batch, (tuple, list)) else (batch, None)
                x, _, y = batch
                self._examples_processed += x.shape[0]
                lrs = {
                    f"group_{i}": param["lr"]
                    for i, param in enumerate(self.optimizer.param_groups)
                }
                self._summary_writer.add_scalars("metrics/lr", lrs, self._step)
                metrics = self._train_one_batch(x, y)
                self._log_metrics(metrics, training=True)

                self._time_taken += time.time() - start_time
                start_time = time.time()
                self._summary_writer.add_scalar(
                    "speed/examples_per_sec",
                    self._examples_processed / self._time_taken,
                    self._step,
                )
                self._summary_writer.add_scalar(
                    "speed/millis_per_example",
                    self._time_taken / self._examples_processed * 1000,
                    self._step,
                )
                self._summary_writer.add_scalar("speed/epoch", self._epoch, self._step)
                self._summary_writer.add_scalar("speed/step", self._step, self._step)
                self._step += 1

            # Evaluate
            n_examples, sum_metrics = 0, collections.defaultdict(float)
            for batch in self.eval_loader:
                batch = batch if isinstance(batch, (tuple, list)) else (batch, None)
                x, _, y = batch
                n_batch_examples = x.shape[0]
                n_examples += n_batch_examples
                for key, metric in self._eval_one_batch(x, y).items():
                    sum_metrics[key] += metric * n_batch_examples
            metrics = {key: metric / n_examples for key, metric in sum_metrics.items()}
            self._log_metrics(metrics, training=False)

            print(f"Finished epoch {self._epoch}. Checkpointing...")
            self._epoch += 1
            self._save_checkpoint()
            if self.sample_epochs and self._epoch % self.sample_epochs == 0:
                self.model.eval()
                with torch.no_grad():
                    tensor = self.sample_fn(self.model)
                self._summary_writer.add_images("sample", tensor, self._step)

        self._summary_writer.close()


class PixelQuadManager:
    def __init__(self, model, train_loader, test_loader, batch_size=8):
        self.model = model.to(model._device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.batch_size = batch_size

    def loss_fn(self, x, _, preds):
        loc = torch.sigmoid(preds[:, 0:4, :, :].transpose(1, 3).to(self.model._device))
        loc = torch.add(torch.mul(loc, 130), -100)
        diag = torch.exp(F.elu(preds[:, 4:8, :, :].transpose(1, 3))).to(self.model._device)
        tri = preds[:, 8:14, :, :].transpose(1, 3).to(self.model._device)
        z = torch.zeros(size=loc.shape[0:3], device=self.model._device).to(self.model._device)
        scale_tril = torch.stack([
            diag[:, :, :, 0], z, z, z,
            tri[:, :, :, 0], diag[:, :, :, 1], z, z,
            tri[:, :, :, 1], tri[:, :, :, 2], diag[:, :, :, 2], z,
            tri[:, :, :, 3], tri[:, :, :, 4], tri[:, :, :, 5], diag[:, :, :, 3]
        ], dim=-1).view(loc.shape[0], loc.shape[1], loc.shape[2], 4, 4).to(self.model._device)

        dist = MultivariateNormal(loc=loc, scale_tril=scale_tril)

        return -dist.log_prob(x.transpose(1, 3)).mean()

    def train_model(
            self,
            n_epochs=1000,
            log_dir="/tmp/run",
            n_gpus=1,
            device_id=0,
            lr=1e-5,
            decay=3.3e-5
    ):
        """Training script with defaults to reproduce results.
        The code inside this function is self contained and can be used as a top level
        training script, e.g. by copy/pasting it into a Jupyter notebook.
        Args:
            n_epochs: Number of epochs to train for.
            batch_size: Batch size to use for training and evaluation.
            log_dir: Directory where to log trainer state and TensorBoard summaries.
            n_gpus: Number of GPUs to use for training the model. If 0, uses CPU.
            device_id: The device_id of the current GPU when training on multiple GPUs.
            debug_loader: Debug DataLoader which replaces the default training and
                evaluation loaders if not 'None'. Do not use unless you're writing unit
                tests.
        """
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        scheduler = lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lambda _: 1-decay)

        trainer = TrainerPixelQuad(
            model=self.model,
            loss_fn=self.loss_fn,
            optimizer=optimizer,
            clip_grad_norm=1,
            train_loader=self.train_loader,
            eval_loader=self.test_loader,
            lr_scheduler=scheduler,
            log_dir=log_dir,
            save_checkpoint_epochs=1,
            n_gpus=n_gpus,
            device_id=device_id,
        )
        trainer.interleaved_train_and_eval(n_epochs)

    def pred_transforms(self, pred, shape):
        loc = torch.sigmoid(pred[:, 0:4, :, :].transpose(1, 3).to(self.model._device))
        loc = torch.add(torch.mul(loc, 130), -100)
        diag = torch.exp(nn.functional.elu(pred[:, 4:8, :, :])).transpose(1, 3).to(self.model._device)
        tri = pred[:, 8:14, :, :].transpose(1, 3).to(self.model._device)
        z = torch.zeros(size=loc.shape[0:3], device=self.model._device).to(self.model._device)
        scale_tril = torch.stack([
            diag[:, :, :, 0], z, z, z,
            tri[:, :, :, 0], diag[:, :, :, 1], z, z,
            tri[:, :, :, 1], tri[:, :, :, 2], diag[:, :, :, 2], z,
            tri[:, :, :, 3], tri[:, :, :, 4], tri[:, :, :, 5], diag[:, :, :, 3]
        ], dim=-1).view(loc.shape[0], loc.shape[1], loc.shape[2], 4, 4).to(self.model._device)
        cov = torch.matmul(
            scale_tril[0,].view(-1, 4, 4),
            scale_tril[0,].view(-1, 4, 4).transpose(1, 2)
        ).view(shape[2], shape[1], 4, 4)

        return loc.transpose(1, 2), cov.transpose(0, 1)

    def plot_forward_pass(self, data, plot_cov=False):
        pred = self.model.forward(
            torch.Tensor(torch.unsqueeze(data, 0)).to(self.model._device)
        )
        loc, cov = self.pred_transforms(pred, data.shape)
        img = data

        if plot_cov:
            fig, axs = plt.subplots(4, 4)

            names = ['NE', 'NW', 'SE', 'SW']
            for ii in range(4):
                for jj in range(4):
                    if ii == jj:
                        vmin, vmax = -10, 10
                    else:
                        vmin, vmax = -3, 3
                    sb.heatmap(cov.cpu().detach().numpy()[:, :, ii, jj],
                               ax=axs[ii, jj], center=0, vmin=vmin, vmax=vmax)
                    if ii == 0:
                        axs[ii, jj].set_title(f'{names[jj]}')

            fig.set_size_inches(16, 16)
            plt.tight_layout()
            plt.show()
        else:
            fig, axs = plt.subplots(2, 4)
            vmin = -100  # np.min(img)
            vmax = 30  # np.max(img)

            names = ['NE', 'NW', 'SE', 'SW']
            pos = [1, 0, 3, 2]
            for ii in range(4):
                sb.heatmap(img[ii, :, :],
                           ax=axs.flat[2 * pos[ii]], vmin=vmin, vmax=vmax)
                axs.flat[2 * pos[ii]].set_title(f'True {names[ii]}')

            for ii in range(4):
                sb.heatmap(loc[0, :, :, ii].cpu().detach().numpy(),
                           ax=axs.flat[2 * pos[ii] + 1], vmin=vmin, vmax=vmax)
                axs.flat[2 * pos[ii] + 1].set_title(f"Fitted mean {names[ii]}")

            plt.tight_layout()
            plt.show()

    def _simulate_pixel(self, data, row, pixel):
        with torch.no_grad():
            pred = self.model.forward(
                torch.Tensor(torch.unsqueeze(data, 0)).to(self.model._device)
            )
        loc, cov = self.pred_transforms(pred, data.shape)
        dist = MultivariateNormal(loc=loc[0, row, pixel, :],
                                  covariance_matrix=cov[row, pixel, :, :])

        return dist.sample()

    def _simulate_row(self, data, row):
        for col in range(data.shape[2]):
            data[:, row, col] = self._simulate_pixel(data, row, col).cpu().detach()

        return data

    def simulate_image(self, data, start_row=13):
        data_sim = copy.deepcopy(data)

        row = start_row
        while row < data_sim.shape[1]:
            print(f'Simulating row {row}...')
            data_sim = self._simulate_row(data_sim, row)
            row += 1

        return data_sim

    def plot_sim_comparison(self, img, sim, start_row=13):
        fig, axs = plt.subplots(2, 4)
        vmin = -100  # np.min(img)
        vmax = 30  # np.max(img)

        names = ['NE', 'NW', 'SE', 'SW']
        pos = [1, 0, 3, 2]
        for ii in range(4):
            sb.heatmap(img[ii, :, :], ax=axs.flat[2 * pos[ii]], vmin=vmin, vmax=vmax)
            axs.flat[2 * pos[ii]].set_title(f'True {names[ii]}')
            axs.flat[2 * pos[ii]].axhline(start_row)

        for ii in range(4):
            sb.heatmap(sim[ii, :, :], ax=axs.flat[2 * pos[ii] + 1], vmin=vmin, vmax=vmax)
            axs.flat[2 * pos[ii] + 1].set_title(f"Simulated {names[ii]}")
            axs.flat[2 * pos[ii] + 1].axhline(start_row)

        plt.tight_layout()
        plt.show()