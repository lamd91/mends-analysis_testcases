#!/usr/bin/python3

# Load modules
import sys
import mpstool
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from matplotlib import pylab as pl
from matplotlib import colors
import matplotlib.ticker as mtick
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from random import randint
import matplotlib.ticker as ticker


def dataMismatchWithIteration(ensembleSize):
    """
    Plot obj function value at each iteration for each ensemble member

    Argument:
    ensembleSize -- A scalar representing the size of the ensemble
    """
    ensembleMemberIndices = np.arange(ensembleSize)
    labelColors = ['b', 'y', 'g', 'm', 'c', 'r', 'k']
    markers = ['o', '^', 's', '>', '*', 'v', 'd', '<']
    fig = plt.figure(figsize=(7, 4))
    iter_max = 1  # Initialize variable

    for j in ensembleMemberIndices:
        objFunValuesOfMember = np.reshape(np.loadtxt(
                               f'objFunValues_{i}.txt'), (1, -1)).tolist()
        iterations = np.reshape(np.arange(len(objFunValuesOfMember[0])), 
                                (1, -1)).tolist()
        maxIterOfMember = max(iterations[0])
        if maxIterOfMember > iter_max:
            iter_max = maxIterOfMember
        line = plt.plot(iterations[0], np.log10(objFunValuesOfMember[0]), 
                        labelColors[np.remainder(i, 6)], 
                        label=f'member i out of {ensembleSize}', 
                        linestyle=':', linewidth=0.5, 
                        marker=markers[np.remainder(i, 7)],
                        markersize=4, markeredgecolor='none')[0]
        line.set_clip_on(False)  # Set markers on top of axis
        plt.xticks(np.arange(1, iter_max+1, 1))
        axes = plt.gca()
        axes.set_xlim([0, iter_max+1])

    handles, labels = axes.get_legend_handles_labels()
    plt.legend([handles[0]], [labels[0]], loc='upper right', fontsize=8)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    plt.xlabel('Iteration', fontsize=8)
    plt.ylabel('Data mismatch', fontsize=8)
    ttl = axes.title
    ttl.set_position([.5, 1.02])
    plt.savefig(f'dataMismatch_N{ensembleSize}.png', 
                bbox_inches="tight", dpi=300)
    plt.show()

    return


def dataMismatchWithIteration_p5p50p95(ensembleSize, rankMember, nbIter):
    """ 
    Plot evolution of 5th, 50th (median) and 95th percentiles 
    of distribution of obj function values calculated from ensemble

    Argument:
    ensembleSize -- A scalar representing the size of the ensemble
    rankMember -- A scalar representing the index of the ensemble member
    nbIter -- A scalar representing the max number of iterations to show
    """
    iterations = np.arange(nbIter+1)
    objFunEns = np.empty((ensembleSize, nbIter+1))
    objFunEns[:,:] = np.nan

    fig = plt.figure(figsize=(7,4))

    for i in np.arange(ensembleSize):
        objFunValuesOfMember = np.reshape(
                        np.log10(np.loadtxt(f'objFunValues_{i}.txt')), (1,-1))
        length = objFunValuesOfMember.shape[1]
        objFunEns[i, 0:length] = objFunValuesOfMember

    q = [5,50,95]   
    objFunEns_q = np.nanpercentile(objFunEns, q=q, axis=0)
    plt.fill_between(iterations, objFunEns_q[0], objFunEns_q[2], 
                     facecolor='blue', alpha=0.5) 
    plt.plot(iterations, objFunEns_q[1], c='blue', label='median')
    plt.plot(iterations, objFunEns[rankMember], c='k', 
             label=f'member {rankMember}', linestyle='--')
    plt.xlim([0, nbIter+0.25])
    plt.xticks(np.arange(1, nbIter+1))
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.xlabel('Iteration', fontsize=10)
    plt.ylabel('Data mismatch (log10)', fontsize=10)
    plt.legend(fontsize=10)
    plt.savefig(f'dataMismatch_N{ensembleSize}_betweenQ5Q95.png',
                bbox_inches="tight", dpi=300)
    plt.show()

    return


def showObs():
    """
    Plot the synthetic hydraulic head and flow rate observations used for
    model calibration and/or prediction at all observation points
    """
    # Names of observation locations
    headObsLocationNames = ['Obs. point #1\nx=0, z=50m', 
                              'Obs. point #2\nx=0, z=150m',
                              'Obs. point #3\nx=0, z=250m',
                              'Obs. point #4\nx=0, z=350m',
                              'Obs. point #5\nx=0, z=450m',
                              'Obs. point #6\nx=1km, z=50m',
                              'Obs. point #7\nx=1km, z=150m',
                              'Obs. point #8\nx=1km, z=250m',
                              'Obs. point #9\nx=1km, z=350m',
                              'Obs. point #10\nx=1km, z=450m']

    # Load array of observation by location
    timeObs = np.arange(0, 43200+1200, 1200)  # includes time zero which corresponds to the initial head computed from the steady-state simulation
    headObsByLoc = np.loadtxt('hObs_byLoc.txt') # includes initial steady-state head
    
    nrows = 3
    ncols = 5
    fig, axarr = plt.subplots(nrows, ncols, figsize=(8, 9))
    fig.subplots_adjust(hspace=0.3)
    fig.subplots_adjust(wspace=0.6)

    k = 0   
    axarr[0, 0].set_ylabel('Hydraulic head (m)', fontsize=9)    
    axarr[1, 0].set_ylabel('Hydraulic head (m)', fontsize=9)
    for i in range(nrows-1):
        for j in range(ncols):
            # Plot observed hydraulic head (H) in red
            axarr[i, j].scatter(timeObs, headObsByLoc[:, k], s=4, c='r', 
                                edgecolors='none', label='obs', zorder=3) 
            axarr[i, j].set_title(headObsLocationNames[k], fontsize=9)
            axarr[i, j].set_xlim(0, 43200)
            axarr[i, j].set_ylim(np.min(headObsByLoc[1:, k])-30, 
                                 np.max(headObsByLoc[1:, k])+10)
            axarr[i, j].xaxis.set_ticks(np.arange(0, 43200, 12000))
            axarr[i, j].set_xticklabels(['0', '12', '24', '36'])
            axarr[i, j].tick_params(labelsize=8)
            handles, labels = axarr[i, j].get_legend_handles_labels()
            k += 1
    axarr[0, 0].legend([handles[0]], [labels[0]], loc="lower right", 
                       scatterpoints=1, fontsize=8)                   

    # Names of observation locations
    flowrObsLocationNames = ['Zone 1', 'Zone 2', 'Zone 3', 'Zone 4', 
                                  'Zone 5']
    # Load array of observation by location
    time = np.arange(300, 6300, 300)
    qObsByLoc = np.loadtxt('qObs_byLoc.txt')
    
    axarr[2, 0].set_ylabel('Flowrates (m' + r'$^3$' + '/s)', fontsize=9) 
    axarr[2, 0].yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.0e'))
    axarr[2, 1].yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.0e'))
    axarr[2, 2].yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.0e'))
    axarr[2, 3].yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.0e'))
    axarr[2, 4].yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.0e'))

    i = 2
    k = 0
    for j in range(ncols):
        # Observed flow rates (Q) in red
        axarr[i, j].scatter(time, qObsByLoc[:, k], s=4, c='r', 
                           edgecolors='none', label='obs', zorder=3) 
        axarr[i, j].set_title(flowrObsLocationNames[k], fontsize=9)
        axarr[i, j].set_ylim(0, np.max(qObsByLoc[:, k])*1.1)
        axarr[i, j].set_xlabel('Time (' + r'$10^3$' + ' s)', fontsize=9)
        axarr[i, j].xaxis.set_ticks(np.arange(0, 12000, 2000))
        axarr[i, j].set_xticklabels(['0', '2', '4', '6'])
        axarr[i, j].set_xlim(0, 6000)
        axarr[i, j].tick_params(labelsize=8)
        handles, labels = axarr[i, j].get_legend_handles_labels()
        k += 1

    axarr[2, 3].set_ylim(-0.001, 0.001)
    axarr[2, 3].yaxis.set_ticks(np.array([0]))   
    axarr[2, 3].yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.0e'))

#   figureName = f'HQ_priorPost_{iteration)_{ensembleSize}.png'
#   plt.savefig(figureName, bbox_inches="tight", dpi=300)
#   plt.show()

    return


def simDataEns(iteration, ensembleSize):
    """ 
    Plot ensemble of simulated data before and after data assimilation
    at every observation location

    Arguments:
    iteration -- A scalar indicating the iteration until the resulting
        ensemble of simulated data is shown (e.g. the total number of 
        iterations performed with ES-MDA)
    ensembleSize -- A scalar denoting the size of the ensemble
    """
    # Names of observation locations
    headObsLocationNames = ['Obs. point #1\nx=0, z=50m', 
                              'Obs. point #2\nx=0, z=150m',
                              'Obs. point #3\nx=0, z=250m',
                              'Obs. point #4\nx=0, z=350m',
                              'Obs. point #5\nx=0, z=450m',
                              'Obs. point #6\nx=1km, z=50m',
                              'Obs. point #7\nx=1km, z=150m',
                              'Obs. point #8\nx=1km, z=250m',
                              'Obs. point #9\nx=1km, z=350m',
                              'Obs. point #10\nx=1km, z=450m']

    # Load array of observation by location
   
    # Time vector includes time zero which corresponds to the initial head 
    # computed from the steady-state simulation
    timeObs = np.arange(0, 43200+1200, 1200)  
    # Head series includes initial steady-state head
    headObsByLoc = np.loadtxt('hObs_byLoc.txt')
 
    # Load prior ensemble of simulated data
    headSimEns_ini = np.loadtxt(f'hSim_ens_0_{ensembleSize}.txt')
    # Load last updated ensemble of simulated data
    headSimEns_last = np.loadtxt(f'hSim_ens_{iteration}_{ensembleSize}.txt')

    obsIndexIntervalByLoc = np.arange(0, 
                headSimEns_ini.shape[0]+timeObs.shape[0], timeObs.shape[0])
    
    nrows = 3
    ncols = 5
    fig, axarr = plt.subplots(nrows, ncols, figsize=(8, 9))
    fig.subplots_adjust(hspace=0.3)
    fig.subplots_adjust(wspace=0.6)

    k = 0   
    axarr[0, 0].set_ylabel('Hydraulic head (m)', fontsize=9)    
    axarr[1, 0].set_ylabel('Hydraulic head (m)', fontsize=9)
    for i in range(nrows-1):
        for j in range(ncols):
            # Observed hydraulic head (H) in red
            axarr[i, j].scatter(timeObs, headObsByLoc[:, k], s=4, c='r', 
                                edgecolors='none', label='obs', zorder=3)
            # Simulated ensemble before data assimilation
            axarr[i, j].plot(timeObs, headSimEns_ini[
                 obsIndexIntervalByLoc[k]:obsIndexIntervalByLoc[k+1], :], 
                 c=(0.7, 0.7, 0.7), linewidth=0.4, alpha=0.4, zorder=1, 
                 label='prior')
            # Simulated ensemble after data assimilation
            axarr[i, j].plot(timeObs, headSimEns_last[
                 obsIndexIntervalByLoc[k]:obsIndexIntervalByLoc[k+1], :], 
                 c='b', linewidth=0.4, alpha=0.8, zorder=2, 
                 label=f'it={iteration}')
            axarr[i, j].set_title(headObsLocationNames[k], fontsize=9)
            axarr[i, j].set_xlim(0, 43200)
            axarr[i, j].set_ylim(np.min(headObsByLoc[1:, k])-30, 
                                 np.max(headObsByLoc[1:, k])+10)
            axarr[i, j].xaxis.set_ticks(np.arange(0, 43200, 12000))
            axarr[i, j].set_xticklabels(['0', '12', '24', '36'])
            axarr[i, j].tick_params(labelsize=8)
            handles, labels = axarr[i, j].get_legend_handles_labels()
            k += 1
    axarr[0, 0].legend([handles[0], handles[ensembleSize], handles[-1]],
                       [labels[0], labels[ensembleSize], labels[-1]],
                       loc="lower right", scatterpoints=1, fontsize=8)

    # Names of observation locations
    flowrObsLocationNames = ['Zone 1', 'Zone 2', 'Zone 3', 'Zone 4', 'Zone 5']
    # Load array of observation by location
    time = np.arange(300, 6300, 300)
    qObsByLoc = np.loadtxt('qObs_byLoc.txt')

    # Load prior ensemble of simulated data
    qSimEns_ini = np.loadtxt(f'qSim_ens_0_{ensembleSize}.txt') 
    qSimEns_last = np.loadtxt(f'qSim_ens_{iteration}_{ensembleSize}.txt') 

    obsIndexIntervalByLoc = np.arange(0, qSimEns_ini.shape[0]+time.shape[0], 
                                      time.shape[0])
    
    axarr[2, 0].set_ylabel('Flowrates (m' + r'$^3$' + '/s)', fontsize=9) 
    axarr[2, 0].yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.0e'))
    axarr[2, 1].yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.0e'))
    axarr[2, 2].yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.0e'))
    axarr[2, 3].yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.0e'))
    axarr[2, 4].yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.0e'))

    i = 2
    k = 0
    for j in range(ncols):
        # Observed flow rates (Q) in red
        axarr[i, j].scatter(time, qObsByLoc[:, k], s=4, c='r', 
                           edgecolors='none', label='obs', zorder=3)
        # Simulated ensemble before data assimilation
        axarr[i, j].plot(time, qSimEns_ini[
                   obsIndexIntervalByLoc[k]:obsIndexIntervalByLoc[k+1], :], 
                   c=(0.7, 0.7, 0.7), linewidth=0.4, alpha=0.8, zorder=1, 
                   label='prior')
        # Simulated ensemble after data assimilation
        axarr[i, j].plot(time, qSimEns_last[
                   obsIndexIntervalByLoc[k]:obsIndexIntervalByLoc[k+1], :], 
                   c='b', linewidth=0.4, alpha=0.8, zorder=2, 
                   label=f'iter {iteration}') 
        axarr[i, j].set_title(flowrObsLocationNames[k], fontsize=9)
        axarr[i, j].set_ylim(0, np.max(qObsByLoc[:, k])*1.1)
        axarr[i, j].set_xlabel('Time (' + r'$10^3$' + ' s)', fontsize=9)
        axarr[i, j].xaxis.set_ticks(np.arange(0, 12000, 2000))
        axarr[i, j].set_xticklabels(['0', '2', '4', '6'])
        axarr[i, j].set_xlim(0, 6000)
        axarr[i, j].tick_params(labelsize=8)
        handles, labels = axarr[i, j].get_legend_handles_labels()
        k += 1

    axarr[2, 3].set_ylim(-0.001, 0.001)
    axarr[2, 3].yaxis.set_ticks(np.array([0]))   
    axarr[2, 3].yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.0e'))

#   figureName = f'HQ_priorPost_{iteration}_{ensembleSize}.png'
#   plt.savefig(figureName, bbox_inches="tight", dpi=300)
#   plt.show()

    return


def simHead_rejectionSamplingPosterior():
    """
    Plot the ensemble of simulated data resulting from the posterior ensemble
    of log K obtained by rejection sampling
    """
    # Names of observation locations
    headObsLocationNames = ['Obs. point #1\nx=0, z=50m', 
                              'Obs. point #2\nx=0, z=150m',
                              'Obs. point #3\nx=0, z=250m',
                              'Obs. point #4\nx=0, z=350m',
                              'Obs. point #5\nx=0, z=450m',
                              'Obs. point #6\nx=1km, z=50m',
                              'Obs. point #7\nx=1km, z=150m',
                              'Obs. point #8\nx=1km, z=250m',
                              'Obs. point #9\nx=1km, z=350m',
                              'Obs. point #10\nx=1km, z=450m']

    # Load array of observation by location
    # Time vector includes time zero which corresponds to the initial head 
    # computed from the steady-state simulation
    timeObs = np.arange(0, 43200+1200, 1200) 
    # Head series includes initial steady-state head
    headObsByLoc = np.loadtxt('hObs_byLoc.txt')

    # Load ensemble of simulated data corresponding to log K ensemble obtained
    # by rejection sampling 
    headSimEns_last = np.loadtxt('hSim_ens_post.txt') 

    obsIndexIntervalByLoc = np.arange(
            0, headSimEns_last.shape[0]+timeObs.shape[0], timeObs.shape[0])
    
    nrows = 2
    ncols = 5
#   plt.close('all')
    fig, axarr = plt.subplots(nrows, ncols, figsize=(8, 6))
    fig.subplots_adjust(hspace=0.3)
    fig.subplots_adjust(wspace=0.6)

    k = 0
    axarr[0, 0].set_ylabel('Hydraulic head (m)', fontsize=9)    
    axarr[1, 0].set_ylabel('Hydraulic head (m)', fontsize=9)
    for i in range(nrows):
        for j in range(ncols):
            # Observed hydraulic head in red
            axarr[i, j].scatter(timeObs, headObsByLoc[:, k], s=4, c='r', 
                                edgecolors='none', label='obs', zorder=3)
            # Simulated ensemble 
            axarr[i, j].plot(timeObs, headSimEns_last[
                 obsIndexIntervalByLoc[k]:obsIndexIntervalByLoc[k+1], :], 
                 c='b', linewidth=0.4, alpha=0.8, zorder=2, label='post') 
    #       axarr[i, j].set_xlabel('Time (' + r'$10^3$' + ' s)', fontsize=8)
            axarr[i, j].set_title(headObsLocationNames[k], fontsize=9)
    #       if i == 0:
    #           axarr[i, j].set_xlabel('')      
            axarr[i, j].set_xlim(0, 43200)
            axarr[i, j].set_ylim(np.min(headObsByLoc[1:, k])-30, 
                                 np.max(headObsByLoc[1:, k])+10)
            axarr[i, j].xaxis.set_ticks(np.arange(0, 43200, 12000))
            axarr[i, j].set_xticklabels(['0', '12', '24', '36'])
            axarr[i, j].tick_params(labelsize=8)
            handles, labels = axarr[i, j].get_legend_handles_labels()
            k += 1
    axarr[0, 0].legend([handles[0], handles[-1]], [labels[0], labels[-1]], loc="lower right", scatterpoints=1, fontsize=8)                  

    figureName = "H_exactPost.pdf" 
    plt.savefig(figureName, bbox_inches="tight")
#   plt.show()

    return


def conditionedCategoricalFields_4members(
        rank_member1, rank_member2, rank_member3, rank_member4, gridDims):
    """
    Plot categorical fields generated throughout the data assimilation 
    procedure (see the specified iterations in for loop) 
    for 4 different ensemble members

    Arguments:
    rank_member[1-4] -- Each argument is a scalar denoting the index of an 
                        ensemble member
    gridDims -- A tuple denoting the dimensions of the 2D grid
    """
    ref = np.flipud(np.reshape(np.loadtxt('ref.txt'), gridDims))

    ncols = 4
    nrows = 17
    grid = GridSpec(nrows, ncols, wspace=0.3, hspace=0.1)
    fig = plt.figure(figsize=(7, 8))
    fig.clf()

    k = 0
    for rank in [rank_member1, rank_member2]:    

        # For each iteration
        j=0
        cmap = plt.cm.rainbow
        norm = matplotlib.colors.BoundaryNorm(np.arange(-0.5, 2, 1), cmap.N)

        start_index = 0
        end_index = 1

        # Show generated categorical fields at iteration 1, 2, 3, 4, 8, 12, 16
        for i in [1, 2, 3, 4, 8, 12, 16]:
            ax = fig.add_subplot(grid[start_index:end_index, k:k+2])    

            # Load data
            if i == 0:
                mpSim = np.flipud(np.reshape(np.loadtxt('iniMPSimEns.txt')
                                             [:, rank], gridDims))
            else:
                mpSim = np.flipud(np.reshape(np.loadtxt(
                      f'ens_of_MPSim_{i}.txt')[:, rank], gridDims))

            im = ax.imshow(mpSim, cmap=cmap, norm=norm, vmin=0, vmax=1, 
                           aspect='auto')
        
            ax.get_xaxis().set_visible(False)  # Hide axis
            ax.axes.get_yaxis().set_visible(False)
            
            start_index = start_index+1
            end_index = end_index+1

        k = k+2

    k = 0
    for rank in [rank_member3, rank_member4]:    

        # For each iteration
        j=0
        cmap = plt.cm.rainbow
        norm = matplotlib.colors.BoundaryNorm(np.arange(-0.5, 2, 1), cmap.N)

        start_index = 8
        end_index = start_index + 1

        # Show generated categorical fields at iterations 1, 2, 3, 4, 8, 12, 16
        for i in [1, 2, 3, 4, 8, 12, 16]:
            ax = fig.add_subplot(grid[start_index:end_index, k:k+2])    

            # Load data
            if i == 0:
                mpSim = np.flipud(np.reshape(np.loadtxt('iniMPSimEns.txt')
                                             [:, rank], gridDims))
            else:
                mpSim = np.flipud(np.reshape(np.loadtxt(
                      f'ens_of_MPSim_{i}.txt')[:, rank], gridDims))

            im = ax.imshow(mpSim, cmap=cmap, norm=norm, vmin=0, vmax=1,
                           aspect='auto')
        
            ax.get_xaxis().set_visible(False)  # Hide axis
            ax.axes.get_yaxis().set_visible(False)
            
            start_index = start_index+1
            end_index = end_index+1

        k = k+2

    ax = fig.add_subplot(grid[16, 1:3])
    im = ax.imshow(ref, cmap=cmap, norm=norm, vmin=0, vmax=1, aspect='auto')

    ax.get_xaxis().set_visible(False)  # Hide axis
    ax.axes.get_yaxis().set_visible(False)

    return


def updatedCoarseVar_multiresoMPS(
                iteration, memberRank, coarseGridDims, fineGridDims):
    """
    Plot the different stages of the update process for one given iteration and
    one given ensemble member

    Arguments:
    iteration -- An integer representing the iteration number 
    memberRank -- An integer denoting the index of the member in the
                  ensemble
    coarseGridDims -- A tuple of integers composed of the width and height of
            the MPS field simulated at the coarsest scale in pixels
    fineGridDims -- A tuple of integers composed of the width and height of
            the MPS field simulated at the finest scale in pixels   
    """
    beforeIteration = iteration - 1
    par = np.reshape(
        np.loadtxt('ens_of_parameters_beforeUpdate_0.txt')[:, memberRank],
        coarseGridDims)  # Not shown in figure
    parIni_min = np.min(par)
    parIni_max = np.max(par)
    parUpdated = np.reshape(
        np.loadtxt(f'ens_of_parameters_{beforeIteration}.txt')[:, memberRank],
        coarseGridDims)
    pyrUpdated_beforeDS = np.reshape(
            np.loadtxt(f'ens_of_updatedPyr_afterKalman_{beforeIteration}.txt')
                   [:, memberRank],
            coarseGridDims)
    pyrUpdated_afterDS = np.reshape(
            np.loadtxt(f'ens_of_updatedPyr_afterKalman-DS_{iteration}.txt')
                    [:, memberRank],
            coarseGridDims)
    condFaciesSim = np.reshape(
            np.loadtxt(f'ens_of_MPSim_{iteration}.txt')[:, memberRank], 
            fineGridDims)

    i_HD = np.loadtxt('sampledCells_xCoord.txt').astype(int)
    j_HD = np.loadtxt('sampledCells_yCoord.txt').astype(int)

    maskMatrix = np.ones(coarseGridDims)
    maskMatrix[:, :] = np.nan
    maskMatrix[i_HD, j_HD] = pyrUpdated_beforeDS[i_HD, j_HD]

    HD_pyrUpdated_beforeDS = np.ma.array (maskMatrix, mask=np.isnan(maskMatrix))
    cmap = plt.cm.rainbow
    cmap.set_bad('white', 1.)

    fig, axs = plt.subplots(5, 1, figsize=(7, 5.5))
    fig.subplots_adjust(hspace=0.6)

    im1_r = axs[0].imshow(np.flipud(parUpdated), cmap='rainbow', aspect='auto', 
                          vmin=parIni_min, vmax=parIni_max)
    im2_r = axs[1].imshow(np.flipud(pyrUpdated_beforeDS), cmap='rainbow', 
                          aspect='auto', vmin=0, vmax=1)
    im1 = axs[0].imshow(np.flipud(parUpdated), cmap='rainbow_r', aspect='auto', 
                        vmin=parIni_min, vmax=parIni_max)
    im2 = axs[1].imshow(np.flipud(pyrUpdated_beforeDS), cmap='rainbow_r', 
                        aspect='auto', vmin=0, vmax=1)
    im3 = axs[2].imshow(np.flipud(HD_pyrUpdated_beforeDS), 
                        interpolation='nearest', cmap='rainbow_r', 
                        aspect='auto', vmin=0, vmax=1)
    im4 = axs[3].imshow(np.flipud(pyrUpdated_afterDS), cmap='rainbow_r', 
                        aspect='auto', vmin=0, vmax=1)
    # Get discrete colormap
    cmap5 = plt.get_cmap('rainbow', 
                         np.max(condFaciesSim)-np.min(condFaciesSim)+1) 
    im5 = axs[4].imshow(np.flipud(condFaciesSim), cmap=cmap5, aspect='auto', 
                        vmin=np.min(condFaciesSim)-0.5, 
                        vmax=np.max(condFaciesSim)+0.5)
    
    axs[0].set_title(
        'Updated field - coarse scale  [size: '+r'$13\times125$'+' pixels]', 
        loc='left', fontsize=9)
    axs[1].set_title(
        'Back-transformed field - coarse scale  [size: '
        + r'$13\times125$' + ' pixels]', loc='left', fontsize=9)
    axs[2].set_title(
        'Samples of hard data - coarse scale  [size: '
        + r'$13\times125$' + ' pixels]', loc='left', fontsize=9)
    axs[3].set_title(
        'MPS simulation - coarse scale  [size: ' 
        + r'$13\times125$' + ' pixels]', loc='left', fontsize=9)
    axs[4].set_title(
        'MPS simulation - original scale  [size: '
        + r'$50\times500$' + ' pixels]', loc='left', fontsize=9)
    divider1 = make_axes_locatable(axs[0])
    divider2 = make_axes_locatable(axs[1])
    divider3 = make_axes_locatable(axs[2])
    divider4 = make_axes_locatable(axs[3])
    divider5 = make_axes_locatable(axs[4])
    cax1 = divider1.append_axes("right", size="3%", pad=0.2)
    cax2 = divider2.append_axes("right", size="3%", pad=0.2)
    cax3 = divider3.append_axes("right", size="3%", pad=0.2)
    cax4 = divider4.append_axes("right", size="3%", pad=0.2)
    cax5 = divider5.append_axes("right", size="3%", pad=0.2)
    
    # Match colorbar with grid size
    cbar = plt.colorbar(im1_r, cax=cax1) 
    cbar2 = plt.colorbar(im2_r, cax=cax2)
    cbar3 = plt.colorbar(im2_r, cax=cax3) 
    cbar4 = plt.colorbar(im2_r, cax=cax4) 
    cbar5 = plt.colorbar(im5, cax=cax5, ticks=np.arange(0, 2, 1))

    # Set x and y ticks in meters
    fig.canvas.draw()

    for i in np.arange(4):
        axs[i].set_yticks([0, 6, 12])
        ylabels = [item.get_text() for item in axs[i].get_yticklabels()]
        ylabels[0] = '0'
        ylabels[1] = '250'
        ylabels[2] = '500'
        axs[i].set_yticklabels(ylabels)
    axs[4].set_xticks([0, 99, 199, 299, 399, 499]) 
    axs[4].set_yticks([0, 24, 49])
    xlabels = [item.get_text() for item in axs[4].get_xticklabels()]
    ylabels = [item.get_text() for item in axs[4].get_yticklabels()]
    xlabels[0] = '0'
    xlabels[1] = '1000'
    xlabels[2] = '2000'
    xlabels[3] = '3000'
    xlabels[4] = '4000'
    xlabels[5] = '5000'
    ylabels[0] = '0'
    ylabels[1] = '250'
    ylabels[2] = '500'
    axs[4].set_xticklabels(xlabels)
    axs[4].set_yticklabels(ylabels)
    for i in np.arange(4):
        axs[i].set_xticks([], [])

    fig.text(0.04, 0.5, 'Depth (m)', va='center', rotation='vertical', 
             fontsize=9)
    fig.text(0.5, 0.03, 'Distance to western boundary (m)', ha='center',
             fontsize=9)

    return


def rmse(ensembleSize, criticalLength, lastIt, gridDims):
    """
    Compute at each iteration the RMSE between the ensemble mean estimate and the reference value averaged over all pixels of the field.

    Arguments:
    ensembleSize -- A scalar denoting the size of the ensemble
    criticalLength -- A scalar denoting the length in gridblocks used to localize the update during the assimilation
    lastIt -- A scalar indicating the last update iteration for which to compute the RMSE
    gridDims -- A tuple denoting the dimensions of the 2D grid
    
    Return:
    rmseWithIterations -- A list of floats corresponding to the RMSE value from iteration 0 (before any update) up to lastIt
    """
    ref = np.reshape(np.loadtxt('ref.txt'), gridDims)[:, 0:criticalLength]           

    nbOfElements = gridDims[0]*gridDims[1] 
    parEns_ini = np.reshape(np.loadtxt('iniMPSimEns.txt')[0:nbOfElements, :], (nbOfElements, ensembleSize))
    ensMean_ini = np.reshape(np.dot(parEns_ini, np.ones((ensembleSize, 1))/ensembleSize), gridDims)[:,0:criticalLength]
    rmse_ini = np.sqrt(np.sum((ref-ensMean_ini)**2)/(gridDims[0]*criticalLength))

    rmseWithIterations = []
    rmseWithIterations.append(rmse_ini)
    for i in np.arange(1, lastIt+1):
        parEns_calib = np.reshape(np.loadtxt(f'ens_of_MPSim_{i}.txt')[0:nbOfElements, :], (nbOfElements, ensembleSize))
        ensMean_calib = np.reshape(np.dot(parEns_calib, np.ones((ensembleSize, 1))/ensembleSize), gridDims)[:, 0:criticalLength]
        rmse_fin = np.sqrt(np.sum((ref-ensMean_calib)**2)/(gridDims[0]*criticalLength))
        rmseWithIterations.append(rmse_fin)

    return rmseWithIterations


def meanMaps_vs_ref(gridDims, iterations, figsize=(7,7)):
    """
    Plot probability maps of the channel facies before and after the data
    assimilation at specified iterations and show the reference field
    at the bottom

    Arguments:
    gridDims -- A tuple of integers for the dimensions of the finest 
                simulation grid
    iterations -- A list of integers denoting the iterations for which
                  to compute the updated probability maps
    figize -- A tuple of values to specify the size of the figure
              depending on the number of maps shown
    """
    # Make figure
    fig, axs = plt.subplots(8, 1, figsize=figsize, sharex=True)
    axs.ravel()

    # For each iteration
    j=0
    for i in [0] + iterations:
    #for i in np.arange(0, iteration+1, 2):
        
        # Load data
        if i == 0:
            mpSimEnsMean = np.flipud(
                    np.reshape(np.mean(np.loadtxt('iniMPSimEns.txt'), axis=1),
                               gridDims))
        else:
            mpSimEnsMean = np.flipud(
                    np.reshape(np.mean(np.loadtxt(f'ens_of_MPSim_{i}.txt'), axis=1),
                               gridDims))

        axs[j].tick_params(axis='x', labelsize=10)
        axs[j].tick_params(axis='y', labelsize=10)
        im = axs[j].imshow(mpSimEnsMean, cmap='rainbow', vmin=0, vmax=1,
                           aspect='auto')
        axs[j].set_yticks([0, 24, 49])
        ylabels = [item.get_text() for item in axs[j].get_yticklabels()]
        ylabels[0] = '0'
        ylabels[1] = '250'
        ylabels[2] = '500'
        axs[j].set_yticklabels(ylabels)
        j=j+1   

    ref = np.flipud(np.reshape(np.loadtxt('ref.txt'), gridDims))
    axs[-1].imshow(ref, cmap='rainbow', vmin=0, vmax=1, aspect='auto')
    axs[-1].set_xticks([0, 99, 199, 299, 399, 499])
    axs[-1].set_yticks([0, 24, 49])
    xlabels = [item.get_text() for item in axs[-1].get_xticklabels()]
    ylabels = [item.get_text() for item in axs[-1].get_yticklabels()]
    xlabels[0] = '0'
    xlabels[1] = '1000'
    xlabels[2] = '2000'
    xlabels[3] = '3000'
    xlabels[4] = '4000'
    xlabels[5] = '5000'
    ylabels[0] = '0'
    ylabels[1] = '250'
    ylabels[2] = '500'
    axs[-1].set_xticklabels(xlabels)
    axs[-1].set_yticklabels(ylabels)

    cbar_ax = fig.add_axes([0.33, 0.92, 0.33, 0.02])
    cb = fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
    cbar_ax.xaxis.set_ticks_position('top')
    cbar_ax.tick_params(labelsize=10)
    #plt.title('Probability of channels')

    fig.text(0.04, 0.5, 'Depth (m)', va='center', rotation='vertical', fontsize=10)
    fig.text(0.5, 0.04, 'Distance to western boundary (m)', ha='center', fontsize=10)

    #plt.savefig('meanMapB_paramNearObs.png', bbox_inches="tight", dpi=300)

    return


def meanVarianceReduction(ensembleSize, criticalLength, lastIt, gridDims):

    nbOfElements = gridDims[0]*gridDims[1]
    parEns_ini = np.reshape(np.loadtxt('iniMPSimEns.txt')[0:nbOfElements, :],
                            (nbOfElements, ensembleSize))
    ensVar_ini = np.reshape(
          np.sum((parEns_ini - 
                 np.reshape(np.dot(parEns_ini, 
                                   np.ones((ensembleSize, ensembleSize))/ensembleSize),
                            (nbOfElements, ensembleSize))
                  )**2, axis=1)/ensembleSize, gridDims)
    meanEnsVar_ini = np.mean(ensVar_ini[:, 0:criticalLength])
    
    meanEnsVarWithIterations = []
    meanEnsVarWithIterations.append(meanEnsVar_ini)
    for i in np.arange(1, lastIt+1):
        ens_K_calib = np.reshape(
                np.loadtxt(f'ens_of_MPSim_{i}.txt')[0:nbOfElements, :],
                (nbOfElements, ensembleSize))
        ensVar_fin = np.reshape(
              np.sum((ens_K_calib - np.reshape(np.dot(ens_K_calib, 
                     np.ones((ensembleSize, ensembleSize))/ensembleSize), 
                     (nbOfElements, ensembleSize)))**2, axis=1)
              /ensembleSize, gridDims)
        meanEnsVar_fin = np.mean(ensVar_fin[:, 0:criticalLength])
        meanEnsVarWithIterations.append(meanEnsVar_fin)
    
    return meanEnsVarWithIterations


def KLdiv(ensembleSize, criticalLength, lastIt, gridDims):
    """
    Calculate the Kullback-Leibler divergence (or the "relative entropy")
    of the reference distribution obtained by rejection sampling and the 
    ensemble updated by data assimilation (DA) at each iteration

    Arguments:
    ensembleSize -- An integer denoting the ensemble size
    criticalLength -- An integer corresponding to the critical length in 
            pixels used to localize the update
    lastIt -- An integer representing the last iteration for which the 
            entropy is calculated
    griidDims -- A tuple of integers corresponding to the width and height of
            the categorical fields in pixels

    Return:
    KLdivs -- A list of KL divergence values
    """
    # Initialize list of KL divergence values (average over updated region)
    KLdivs = []

    # Load reference ensemble of facies fields obtained by rejection sampling
    # This ensemble may contain more members than the ensemble updated by DA
    refDist = np.reshape(np.loadtxt('rejectionSampling_posteriorDist.txt'),
                        (gridDims[0]*gridDims[1], -1)) 
    # Extract the facies values 
    allFacies = np.unique(refDist)
 
    # For each iteration 
    for i in range(lastIt+1):
        # Load corresponding updated ensemble of facies fields 
        if i == 0:
            estDist = np.reshape(
                    np.loadtxt('iniMPSimEns.txt'), (-1, ensembleSize))
        else:
            estDist = np.reshape(
                    np.loadtxt(f'ens_of_MPSim_{i}.txt'), (-1, ensembleSize))

        # Initialize array representing a map of KL divergence values
        KLdivMap = np.zeros(gridDims)

        # For each facies        
        for facies in allFacies:
            # Create ensemble of indicator values associated to the facies
            # for the ensemble updated by data assimilation
            faciesIndic_est = np.zeros(estDist.shape)
            faciesIndic_est[np.where(estDist == facies)[0],
                            np.where(estDist == facies)[1]] = 1
            # for the reference ensemble 
            faciesIndic_ref = np.zeros(refDist.shape)
            faciesIndic_ref[np.where(refDist == facies)[0],
                            np.where(refDist == facies)[1]] = 1
        
            # Calculate facies probability (proportion) map 
            # from the estimated and the reference distributions
            probaMap_est = np.reshape(np.mean(faciesIndic_est, axis=1), gridDims)
            probaMap_ref = np.reshape(np.mean(faciesIndic_ref, axis=1), gridDims)
            
            # Add contribution of facies to the overall relative entropy for
            # every pixel of the entropy map
            whereNotNull_i, whereNotNull_j = [
                        np.where((probaMap_est > 0) & (probaMap_ref > 0))[0], 
                        np.where((probaMap_est > 0) & (probaMap_ref > 0))[1]]
            KLdivMap[whereNotNull_i, whereNotNull_j] += np.multiply(
                        probaMap_ref[whereNotNull_i, whereNotNull_j], 
                        np.log2(np.divide(
                             probaMap_ref[whereNotNull_i, whereNotNull_j],
                             probaMap_est[whereNotNull_i, whereNotNull_j])))
                    
        # Calculate average KL divergence value over the updated region
        avgKLdiv = np.mean(KLdivMap[:, 0:criticalLength])
        KLdivs.append(avgKLdiv)
        
    return KLdivs


def shannonEntropy(ensembleSize, criticalLength, lastIt, gridDims):
    """
    Calculate the Shannon entropy of the estimated categorical ensemble at
    each iteration

    Arguments:
    ensembleSize -- An integer denoting the ensemble size
    criticalLength -- An integer corresponding to the critical length in 
            pixels used to localize the update
    lastIt -- An integer representing the last iteration for which the 
            entropy is calculated
    grimDims -- A tuple of integers corresponding to the width and height of
            the categorical fields in pixels

    Return:
    entropies -- A list of Shannon entropy values
    """
    # Initialize list of entropy values (average over updated region)
    entropies = []

    # For each iteration
    for i in range(lastIt+1):

        # Initialize array representing a map of entropy values
        entropyMap = np.zeros(gridDims)

        # Load corresponding file with ensemble of fields
        if i == 0:
            faciesEns_ini = np.reshape(
                    np.loadtxt('iniMPSimEns.txt'), (-1, ensembleSize))
            # Extract the facies values 
            allFacies = np.unique(faciesEns_ini)
        else:
            faciesEns_ini = np.reshape(
                    np.loadtxt(f'ens_of_MPSim_{i}.txt'), (-1, ensembleSize))
        
        # For each facies 
        for facies in allFacies:
            # Create ensemble of indicator values associated to the facies
            faciesIndicEns = np.zeros(faciesEns_ini.shape)
            faciesIndicEns[np.where(faciesEns_ini == facies)[0],
                           np.where(faciesEns_ini == facies)[1]] = 1
            # Calculate facies probability (proportion) map
            probaMap = np.reshape(np.mean(faciesIndicEns, axis=1), gridDims)
            
            # Add contribution of facies to the overall entropy for every 
            # pixel of the entropy map
            whereNotNull_i, whereNotNull_j = [
                        np.where(probaMap > 0)[0], np.where(probaMap > 0)[1]] 
            entropyMap[whereNotNull_i, whereNotNull_j] -= np.multiply(
                        probaMap[whereNotNull_i, whereNotNull_j], 
                        np.log2(probaMap[whereNotNull_i, whereNotNull_j]))
                    
        # Calculate average entropy value over pixels of the updated region
        avgEntropy = np.mean(entropyMap[:, 0:criticalLength])
        entropies.append(avgEntropy)
          
    return entropies


def ensembleOfConnectFunc(ensSize, it, fineGridDim_x, fineGridDim_y, axis, category):
    """
    Evaluate a connectivity function in the horizontal direction for each
    member of the categorical field ensemble

    Arguments:
    ensSize -- An integer indicating the ensemble size
    it -- An integer indicating an update iteration
    fineGridDim_x -- An integer denoting the number of grid cells 
                     along the horizontal axis
    fineGridDim_y -- An integer denoting the number of grid cells
                     along the vertical axis
    axis -- An integer indicating the axis along which to compute the 
            connectivity (0 means vertical, 1 horizontal)
    category -- An integer corresponding to the value of the facies

    Return:
    connectFunEns_ini, connectFunEns_fin -- Each variable corresponds to
        an array of values of shape (ensSize, it) calculated by the 
        connectivity function for each categorical field of ensemble
        before (connectFunEns_ini) and after the data assimilation
       (connectFunEns_fin)
    """
    
    # If connectivity is calculated in the horizontal direction
    if axis == 1:
        connectFunEns_ini=np.zeros((fineGridDim_x-1, ensSize))
        connectFunEns_fin=np.zeros((fineGridDim_x-1, ensSize))
    # Or in the vertical direction
    else:
        connectFunEns_ini=np.zeros((fineGridDim_y-1, ensSize))
        connectFunEns_fin=np.zeros((fineGridDim_y-1, ensSize))

    for i in zip(np.arange(2), np.array([0, it])):
        for j in np.arange(0, ensSize):
            if i[1] == 0:
                mpSim = np.flipud(np.reshape(
                        np.loadtxt('iniMPSimEns_init.txt')[:, j],
                        (fineGridDim_y, fineGridDim_x)))
                connectFun = mpstool.connectivity.get_function(mpSim, axis)
                connectFunEns_ini[:, j] = connectFun[category]
            else:
                mpSim = np.flipud(np.reshape(
                        np.loadtxt(f'ens_of_MPSim_{i[1]}.txt')[:, j],
                        (fineGridDim_y, fineGridDim_x)))
                connectFun = mpstool.connectivity.get_function(mpSim, axis)
                connectFunEns_fin[:, j] = connectFun[category]

    return connectFunEns_ini, connectFunEns_fin

