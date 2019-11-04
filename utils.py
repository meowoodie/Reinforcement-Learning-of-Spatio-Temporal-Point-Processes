#!/usr/bin/env python
# -*- coding: utf-8 -*-
import branca
import folium
import geopandas
import pandas as pd
import numpy as np
import seaborn as sns
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import animation
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.axes_grid1 import make_axes_locatable
from shapely.geometry import Polygon

def lebesgue_measure(S):
    """
    A helper function for calculating the Lebesgue measure for a space.
    It actually is the length of an one-dimensional space, and the area of
    a two-dimensional space.
    """
    sub_lebesgue_ms = [ sub_space[1] - sub_space[0] for sub_space in S ]
    return np.prod(sub_lebesgue_ms)



def l2_norm(x, y):
    """
    This helper function calculates distance (l2 norm) between two arbitrary data points from tensor x and 
    tensor y respectively, where x and y have the same shape [length, data_dim].
    """
    x     = tf.cast(x, dtype=tf.float32)
    y     = tf.cast(y, dtype=tf.float32)
    x_sqr = tf.expand_dims(tf.reduce_sum(x * x, 1), -1) # [length, 1]
    y_sqr = tf.expand_dims(tf.reduce_sum(y * y, 1), -1) # [length, 1]
    xy    = tf.matmul(x, tf.transpose(y))               # [length, length]
    dist_mat = x_sqr + tf.transpose(y_sqr) - 2 * xy
    return dist_mat



def plot_spatial_kernel(path, kernel, S, grid_size,
        sigma_x_clim=None, sigma_y_clim=None, rho_clim=None):
    """
    Plot spatial kernel parameters over the spatial region, including 
    sigma_x, sigma_x, and rho. 
    """
    assert len(S) == 2, '%d is an invalid dimension of the space.' % len(S)
    # define the span for space region
    x_span = np.linspace(S[0][0], S[0][1], grid_size+1)[:-1]
    y_span = np.linspace(S[1][0], S[1][1], grid_size+1)[:-1]
    # map initialization
    sigma_x_map = np.zeros((grid_size, grid_size))
    sigma_y_map = np.zeros((grid_size, grid_size))
    rho_map     = np.zeros((grid_size, grid_size))
    # grid entris calculation
    s = np.array([ [x_span[x_idx], y_span[y_idx]] 
        for x_idx in range(grid_size) for y_idx in range(grid_size) ])
    mu_xs, mu_ys, sigma_xs, sigma_ys, rhos = kernel.nonlinear_mapping(s)
    # mu_xs, mu_ys, sigma_xs, sigma_ys, rhos = \
    #     kernel.mu_x(s[:,0], s[:,1]),\
    #     kernel.mu_y(s[:,0], s[:,1]),\
    #     kernel.sigma_x(s[:,0], s[:,1]),\
    #     kernel.sigma_y(s[:,0], s[:,1]),\
    #     kernel.rho(s[:,0], s[:,1])
    indices = [ [x_idx, y_idx] 
        for x_idx in range(grid_size) for y_idx in range(grid_size) ]
    for i in range(len(indices)):
        sigma_x_map[indices[i][0]][indices[i][1]] = sigma_xs[i]
        sigma_y_map[indices[i][0]][indices[i][1]] = sigma_ys[i]
        rho_map[indices[i][0]][indices[i][1]]     = rhos[i]
    # plotting
    plt.rc('text', usetex=True)
    plt.rc("font", family="serif")
    # plot as a pdf file
    with PdfPages(path) as pdf:
        fig, axs = plt.subplots(1, 3)
        cmap     = matplotlib.cm.get_cmap('viridis')
        im_0     = axs[0].imshow(sigma_x_map, interpolation='nearest', origin='lower', cmap=cmap)
        im_1     = axs[1].imshow(sigma_y_map, interpolation='nearest', origin='lower', cmap=cmap)
        im_2     = axs[2].imshow(rho_map, interpolation='nearest', origin='lower', cmap=cmap)
        sigma_x_clim = [sigma_x_map.min(), sigma_x_map.max()] if sigma_x_clim is None else sigma_x_clim
        sigma_y_clim = [sigma_y_map.min(), sigma_y_map.max()] if sigma_y_clim is None else sigma_y_clim
        rho_clim     = [rho_map.min(), rho_map.max()] if rho_clim is None else rho_clim
        print(sigma_x_map.min(), sigma_x_map.max())
        print(sigma_y_map.min(), sigma_y_map.max())
        print(rho_map.min(), rho_map.max())
        # ticks for colorbars
        im_0.set_clim(*sigma_x_clim)
        im_1.set_clim(*sigma_y_clim)
        im_2.set_clim(*rho_clim)
        tick_0 = np.linspace(sigma_x_clim[0], sigma_x_clim[1], 5).tolist()
        tick_1 = np.linspace(sigma_y_clim[0], sigma_y_clim[1], 5).tolist()
        tick_2 = np.linspace(rho_clim[0], rho_clim[1], 5).tolist()
        # set x, y labels for subplots
        axs[0].set_xlabel(r'$x$')
        axs[1].set_xlabel(r'$x$')
        axs[2].set_xlabel(r'$x$')
        axs[0].set_ylabel(r'$y$')
        axs[1].set_ylabel(r'$y$')
        axs[2].set_ylabel(r'$y$')
        # remove x, y ticks
        axs[0].get_xaxis().set_ticks([])
        axs[1].get_xaxis().set_ticks([])
        axs[2].get_xaxis().set_ticks([])
        axs[0].get_yaxis().set_ticks([])
        axs[1].get_yaxis().set_ticks([])
        axs[2].get_yaxis().set_ticks([])
        # set subtitle for subplots
        axs[0].set_title(r'$\sigma_x$')
        axs[1].set_title(r'$\sigma_y$')
        axs[2].set_title(r'$\rho$')
        # plot colorbar
        cbar_0 = fig.colorbar(im_0, ax=axs[0], ticks=tick_0, fraction=0.046, pad=0.08, orientation="horizontal")
        cbar_1 = fig.colorbar(im_1, ax=axs[1], ticks=tick_1, fraction=0.046, pad=0.08, orientation="horizontal")
        cbar_2 = fig.colorbar(im_2, ax=axs[2], ticks=tick_2, fraction=0.046, pad=0.08, orientation="horizontal")
        # set font size of the ticks of the colorbars
        cbar_0.ax.tick_params(labelsize=5) 
        cbar_1.ax.tick_params(labelsize=5) 
        cbar_2.ax.tick_params(labelsize=5) 
        # adjust the width of the gap between subplots
        plt.subplots_adjust(wspace=0.2)
        pdf.savefig(fig)



def spatial_intensity_on_map(
    path,    # html saving path
    da,      # data adapter object defined in utils.py
    lam,     # lambda object defined in stppg.py
    data,    # a sequence of data points [seq_len, 3] happened in the past
    seq_ind, # index of sequence for visualization 
    t,       # normalized observation moment (t)
    xlim,    # real observation x range
    ylim,    # real observation y range
    ngrid=100):
    """Plot spatial intensity at time t over the entire map given its coordinates limits."""
    # data preparation
    # - remove the first element in the seq, since t_0 is always 0, 
    #   which will cause numerical issue when computing lambda value
    seqs = da.normalize(data)[:, 1:, :] 
    seq  = seqs[seq_ind]                    # visualize the sequence indicated by seq_ind
    seq  = seq[np.nonzero(seq[:, 0])[0], :] # only retain nonzero values
    print(seq)
    seq_t, seq_s = seq[:, 0], seq[:, 1:] 
    sub_seq_t = seq_t[seq_t < t]            # only retain values before time t.
    sub_seq_s = seq_s[:len(sub_seq_t)]
    # generate spatial grid polygons
    xmin, xmax, width       = xlim[0], xlim[1], xlim[1] - xlim[0] 
    ymin, ymax, height      = ylim[0], ylim[1], ylim[1] - ylim[0]
    grid_height, grid_width = height / ngrid, width / ngrid
    x_left_origin   = xmin
    x_right_origin  = xmin + grid_width
    y_top_origin    = ymax
    y_bottom_origin = ymax - grid_height
    polygons        = [] # spatial polygons
    lam_dict        = {} # spatial intensity
    _id             = 0
    for i in range(ngrid):
        y_top    = y_top_origin
        y_bottom = y_bottom_origin
        for j in range(ngrid):
            # append the intensity value to the list
            s = da.normalize_location((x_left_origin + x_right_origin) / 2., (y_top + y_bottom) / 2.)
            v = lam.value(t, sub_seq_t, s, sub_seq_s)
            lam_dict[str(_id)] = np.log(v)
            _id += 1
            # append polygon to the list
            polygons.append(Polygon(
                [(y_top, x_left_origin), (y_top, x_right_origin), (y_bottom, x_right_origin), (y_bottom, x_left_origin)])) 
            # update coordinates
            y_top    = y_top - grid_height
            y_bottom = y_bottom - grid_height
        x_left_origin  += grid_width
        x_right_origin += grid_width
    # convert polygons to geopandas object
    geo_df = geopandas.GeoSeries(polygons) 
    # init map
    _map   = folium.Map(location=[sum(xlim)/2., sum(ylim)/2.], zoom_start=11, zoom_control=True)
    # _map   = folium.Map(location=[sum(xlim)/2., sum(ylim)/2.], zoom_start=6, zoom_control=True, tiles='Stamen Terrain')
    # plot polygons on the map
    print(min(lam_dict.values()), max(lam_dict.values()))
    lam_cm = branca.colormap.linear.YlOrRd_09.scale(np.log(5), np.log(5000)) # scale(10, 5000) # colorbar for intensity values
    poi_cm = branca.colormap.linear.PuBu_09.scale(min(sub_seq_t), max(sub_seq_t)) # colorbar for lasting time of points
    folium.GeoJson(
        data = geo_df.to_json(),
        style_function = lambda feature: {
            'fillColor':   lam_cm(lam_dict[feature['id']]),
            'fillOpacity': .5,
            'weight':      0.}).add_to(_map)
    # plot markers on the map
    for i in range(len(sub_seq_t)):
        x, y = da.restore_location(*sub_seq_s[i])
        folium.Circle(
            location=[x, y],
            radius=10, # sub_seq_t[i] * 100,
            color=poi_cm(sub_seq_t[i]),
            fill=True,
            fill_color='blue').add_to(_map)
    # save the map
    _map.save(path)



class DataAdapter():
    """
    A helper class for normalizing and restoring data to the specific data range.
    
    init_data: numpy data points with shape [batch_size, seq_len, 3] that defines the x, y, t limits
    S:         data spatial range. eg. [[-1., 1.], [-1., 1.]]
    T:         data temporal range.  eg. [0., 10.]
    """
    def __init__(self, init_data, S=[[-1, 1], [-1, 1]], T=[0., 10.], xlim=None, ylim=None):
        self.data = init_data
        self.T    = T
        self.S    = S
        self.tlim = [ init_data[:, :, 0].min(), init_data[:, :, 0].max() ]
        mask      = np.nonzero(init_data[:, :, 0])
        x_nonzero = init_data[:, :, 1][mask]
        y_nonzero = init_data[:, :, 2][mask]
        self.xlim = [ x_nonzero.min(), x_nonzero.max() ] if xlim is None else xlim
        self.ylim = [ y_nonzero.min(), y_nonzero.max() ] if ylim is None else ylim
        print(self.tlim)
        print(self.xlim)
        print(self.ylim)

    def normalize(self, data):
        """normalize batches of data points to the specified range"""
        rdata = np.copy(data)
        for b in range(len(rdata)):
            # scale x
            rdata[b, np.nonzero(rdata[b, :, 0]), 1] = \
                (rdata[b, np.nonzero(rdata[b, :, 0]), 1] - self.xlim[0]) / \
                (self.xlim[1] - self.xlim[0]) * (self.S[0][1] - self.S[0][0]) + self.S[0][0]
            # scale y
            rdata[b, np.nonzero(rdata[b, :, 0]), 2] = \
                (rdata[b, np.nonzero(rdata[b, :, 0]), 2] - self.ylim[0]) / \
                (self.ylim[1] - self.ylim[0]) * (self.S[1][1] - self.S[1][0]) + self.S[1][0]
            # scale t 
            rdata[b, np.nonzero(rdata[b, :, 0]), 0] = \
                (rdata[b, np.nonzero(rdata[b, :, 0]), 0] - self.tlim[0]) / \
                (self.tlim[1] - self.tlim[0]) * (self.T[1] - self.T[0]) + self.T[0]
        return rdata

    def restore(self, data):
        """restore the normalized batches of data points back to their real ranges."""
        ndata = np.copy(data)
        for b in range(len(ndata)):
            # scale x
            ndata[b, np.nonzero(ndata[b, :, 0]), 1] = \
                (ndata[b, np.nonzero(ndata[b, :, 0]), 1] - self.S[0][0]) / \
                (self.S[0][1] - self.S[0][0]) * (self.xlim[1] - self.xlim[0]) + self.xlim[0]
            # scale y
            ndata[b, np.nonzero(ndata[b, :, 0]), 2] = \
                (ndata[b, np.nonzero(ndata[b, :, 0]), 2] - self.S[1][0]) / \
                (self.S[1][1] - self.S[1][0]) * (self.ylim[1] - self.ylim[0]) + self.ylim[0]
            # scale t 
            ndata[b, np.nonzero(ndata[b, :, 0]), 0] = \
                (ndata[b, np.nonzero(ndata[b, :, 0]), 0] - self.T[0]) / \
                (self.T[1] - self.T[0]) * (self.tlim[1] - self.tlim[0]) + self.tlim[0]
        return ndata

    def normalize_location(self, x, y):
        """normalize a single data location to the specified range"""
        _x = (x - self.xlim[0]) / (self.xlim[1] - self.xlim[0]) * (self.S[0][1] - self.S[0][0]) + self.S[0][0]
        _y = (y - self.ylim[0]) / (self.ylim[1] - self.ylim[0]) * (self.S[1][1] - self.S[1][0]) + self.S[1][0]
        return np.array([_x, _y])

    def restore_location(self, x, y):
        """restore a single data location back to the its original range"""
        _x = (x - self.S[0][0]) / (self.S[0][1] - self.S[0][0]) * (self.xlim[1] - self.xlim[0]) + self.xlim[0]
        _y = (y - self.S[1][0]) / (self.S[1][1] - self.S[1][0]) * (self.ylim[1] - self.ylim[0]) + self.ylim[0]
        return np.array([_x, _y])
    
    def __str__(self):
        raw_data_str = "raw data example:\n%s\n" % self.data[:1]
        nor_data_str = "normalized data example:\n%s" % self.normalize(self.data[:1])
        return raw_data_str + nor_data_str
