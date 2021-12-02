"""Regression with abstention plotting functions."""

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors
import seaborn as sns
import palettable
import numpy as np
from collections import Counter
import cartopy as ct
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy.ma as ma
import warnings
import metrics

__author__ = "Elizabeth A. Barnes and Randal J. Barnes"
__date__ = "March 5, 2021"

#----------------------------------------------------------------  

mpl.rcParams['figure.facecolor'] = 'white'
mpl.rcParams['figure.dpi'] = 150
dpiFig = 300.

# =============================================================================
def plot_diagnostics(EXP_NAME, history, exp_info, saveplot=True, showplot=True):

    loss_type, n_epochs, abst_setpoint, spinup_epochs, hiddens, lr_init, lr_epoch_bound, batch_size, network_seed, best_epoch = exp_info    
    trainColor = (117/255., 112/255., 179/255., 1.)
    valColor = (231/255., 41/255., 138/255., 1.)
    FS = 7
    plt.figure(figsize=(15, 7+3.5))

    #---------- plot loss -------------------
    plt.subplot(3,3,1)
    plt.plot(history.history['loss'], 'o', color=trainColor, label='training loss')
    plt.plot(history.history['val_loss'], 'o', color=valColor, label='validation loss')
    if 'prediction_loss' in history.history:    
        plt.plot(history.history['prediction_loss'], 'x', color=trainColor, label='training nonabstaining loss')
        plt.plot(history.history['val_prediction_loss'], 'x', color=valColor, label='validation nonabstaining loss')
    plt.axvline(x=best_epoch,linestyle = '--', color='tab:gray')
    plt.title('Loss')
    plt.xlabel('epoch')
    plt.grid(True)
    plt.legend(frameon=True, fontsize=FS)
    plt.xlim(-2, n_epochs+2)
    plt.ylim(0,2.8)

    # ---------- plot abstention fraction -------------------
    if 'abstention_fraction' in history.history:    
        plt.subplot(3, 3, 2)        
        plt.axhline(abst_setpoint, linestyle='-', color='gray', linewidth=3.0)
        plt.plot(history.history['abstention_fraction'], 'o', color=trainColor, label='training')
        plt.plot(history.history['val_abstention_fraction'], 'o', color=valColor, label='validation')
        plt.legend(frameon=True, fontsize=FS)        
        plt.axvline(x=best_epoch,linestyle = '--', color='tab:gray')        
        plt.title('Abstention Fraction')
        plt.xlabel('epoch')
        plt.ylim(0, 1.02)
        plt.grid(True)    
        plt.xlim(-2, n_epochs+2)
    
    # ---------- plot MAE -------------------
    plt.subplot(3, 3, 3)
    if 'val_mae_covered' in history.history:
        plt.plot(history.history['mae_covered'], 'o', color=trainColor, label='training')
        plt.plot(history.history['val_mae_covered'], 'o', color=valColor, label='validation')
        plt.title('Mean Absolute Error (covered only)')        
    elif 'val_mae' in history.history:    
        plt.plot(history.history['mae'], 'o', color=trainColor, label='training')
        plt.plot(history.history['val_mae'], 'o', color=valColor, label='validation')
        plt.title('Mean Absolute Error')        

    plt.axvline(x=best_epoch,linestyle = '--', color='tab:gray')        
    plt.xlabel('epoch')
    plt.legend(frameon=True, fontsize=FS)
    plt.grid(True)        
    plt.xlim(-2, n_epochs+2)
    
    # ---------- plot alpha -------------------
    if 'alpha_value' in history.history:
        plt.subplot(3, 3, 5)
        plt.plot(history.history['alpha_value'], 'o', color='k')
        plt.title('Alpha')
        plt.xlabel('epoch')
        plt.xlim(-2, n_epochs+2)
        plt.axvline(x=best_epoch,linestyle = '--', color='tab:gray')        
    
    
    # ---------- plot Mean -LogLikelihood -------------------
    plt.subplot(3, 3, 6)
    if 'val_log_likelihood_covered' in history.history:
        plt.plot(history.history['log_likelihood_covered'], 'o', color=trainColor, label='training')
        plt.plot(history.history['val_log_likelihood_covered'], 'o', color=valColor, label='validation')
        plt.title('Mean -LogLikelihood (covered only)')        
    elif 'val_log_likelihood' in history.history:    
        plt.plot(history.history['log_likelihood'], 'o', color=trainColor, label='training')
        plt.plot(history.history['val_log_likelihood'], 'o', color=valColor, label='validation')
        plt.title('Mean -LogLikelihood')        

    plt.axvline(x=best_epoch,linestyle = '--', color='tab:gray')        
    plt.xlabel('epoch')
    plt.legend(frameon=True, fontsize=FS)
#     plt.ylim(0, 3.02)
    plt.grid(True)        
    plt.xlim(-2, n_epochs+2)
    
    # ---------- plot Mean Sigma -------------------
    plt.subplot(3, 3, 8)
    if 'val_sigma_covered' in history.history:
        plt.plot(history.history['sigma_covered'], 'o', color=trainColor, label='training')
        plt.plot(history.history['val_sigma_covered'], 'o', color=valColor, label='validation')
        plt.title('Mean Predicted Sigma (covered only)')        
    elif 'val_sigma' in history.history:    
        plt.plot(history.history['sigma'], 'o', color=trainColor, label='training')
        plt.plot(history.history['val_sigma'], 'o', color=valColor, label='validation')
        plt.title('Mean Predicted Sigma')        

    plt.axvline(x=best_epoch,linestyle = '--', color='tab:gray')        
    plt.xlabel('epoch')
    plt.legend(frameon=True, fontsize=FS)
#     plt.ylim(0, 3.02)
    plt.grid(True)        
    plt.xlim(-2, n_epochs+2)    
    
    # ---------- plot Mean Likelihood -------------------
    plt.subplot(3, 3, 9)
    if 'val_likelihood_covered' in history.history:
        plt.plot(history.history['likelihood_covered'], 'o', color=trainColor, label='training')
        plt.plot(history.history['val_likelihood_covered'], 'o', color=valColor, label='validation')
        plt.title('Mean Likelihood (covered only)')        
    elif 'val_likelihood' in history.history:    
        plt.plot(history.history['likelihood'], 'o', color=trainColor, label='training')
        plt.plot(history.history['val_likelihood'], 'o', color=valColor, label='validation')
        plt.title('Mean Likelihood')        

    plt.axvline(x=best_epoch,linestyle = '--', color='tab:gray')        
    plt.xlabel('epoch')
    plt.legend(frameon=True, fontsize=FS)
#     plt.ylim(0, 3.02)
    plt.grid(True)        
    plt.xlim(-2, n_epochs+2)
    

    
    # ---------- report parameters -------------------
    plt.subplot(3, 3, 7)
    plt.ylim(0, 1)

    text = (
            "\n"
            + f"NETWORK PARAMETERS\n"
            + f"  hiddens        = {hiddens}\n"
            + f"  lr_init        = {lr_init}\n"
            + f"  lr_epoch_bound = {lr_epoch_bound}\n"
            + f"  n_epochs       = {n_epochs}\n"
            + f"  batch_size     = {batch_size}\n"
            + f"  network_seed   = {network_seed}\n"
    )

    text += ('\n' 
             + loss_type + '\n'
             + 'best epoch = ' + str(best_epoch) 
             + '\n'
            )
    
    if loss_type == 'AbstentionLogLossPID':
        text += (
            "\n"
            + f"PID ABSTENTION PARAMETERS\n"
            + f"  spinup_epochs  = {spinup_epochs}\n"
            + f"  abst_setpoint  = {abst_setpoint}\n"
        )

    plt.text(0.01, 0.95, text, fontfamily='monospace', fontsize='small', va='top')

    plt.axis('off')

    # ---------- Make and save the plot -------------------
    plt.tight_layout()
    if saveplot==True:
        plt.savefig('figures/model_diagnostics/' + EXP_NAME + '.png', dpi=dpiFig)
    if showplot==False:
        plt.close('all')
    else:
        plt.show()
        
        
# =============================================================================
def plot_predictionscatter(x, y_pred, y_true, tr, long_name, showplot=False):

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mae = np.round(np.nanmean(np.abs(y_pred[:,0]-y_true)),3)
        mae_tr = np.round(np.nanmean(np.abs(y_pred[tr==1,0]-y_true[tr==1])),3)    
        med_sigma = np.round(np.nanmedian(y_pred[:,1]),3)
        med_sigma_tr = np.round(np.nanmedian(y_pred[tr==1,1]),3)
        
    bounds = 4.
    #-----------
    plt.figure(figsize=(13.33*1.5,3*1.5))
    #--------------------------
    if np.shape(x)[1]==1:    
        plt.subplot(1,4,1)    
        plt.title('data with uncertainty')        
        sns.scatterplot(x=x[:,0], 
                        y=y_true, 
                        hue=y_pred[:,1],
                        hue_norm=(0,3),
                        s=8, 
                        legend=False,
                        palette='Spectral_r')
        plt.xlabel('x')
        plt.ylabel('y (truth)')
        plt.legend()
        plt.xlim(-bounds,bounds)
        plt.ylim(-bounds,bounds)

    #--------------------------
    ax2 = plt.subplot(1,4,2)
    plt.plot((-6,6),(-6,6),'--',color='tab:gray',linewidth=1.,alpha=1.)
    sns.scatterplot(x=y_true, 
                    y=y_pred[:,0], 
                    hue=y_pred[:,1],
                    hue_norm=(0,3),
                    hue_order=[0,.25,.5,.75,1.0,2.0,3.0],
                    s=8, 
                    legend=False,
                    palette='Spectral_r')
    text = (
            f"mae_all = {str(mae)}\n"
            + f"mae_tr  = {str(mae_tr)}\n"
    )
    plt.text(.6,.1,text, fontfamily='monospace', fontsize='small', va='top',transform=ax2.transAxes)    
    plt.ylabel('predicted y')
    plt.xlabel('y (truth)')
    plt.xlim(-bounds,bounds)
    plt.ylim(-bounds,bounds)
    plt.title('prediction vs truth')    

    #--------------------------
    plt.subplot(1,4,3)
    error = np.abs(y_true-y_pred[:,0])
    plt.plot(error,y_pred[:,1],'o',color='tab:gray',markersize=2.,label='tranquil', markerfacecolor='None')
    if(tr is not None):
        plt.plot(error[tr==1],y_pred[tr==1,1],'o',color='tab:purple',markersize=2.,label='tranquil', markerfacecolor='None')
    plt.ylabel('predicted sigma')
    plt.xlabel('absolute error')
    plt.title('uncertainty vs error')
    plt.legend()
    # plt.xlim(0,1)
    # plt.ylim(0,1)

    #--------------------------
    ax4 = plt.subplot(1,4,4)
    std = y_pred[:,1]
    sns.histplot(std, element='step', kde=True)
    plt.xlim(0,5)
    plt.xlabel('predicted sigma')
    plt.title('uncertainty')
    text = (
            f"med_sigma_all = {str(med_sigma)}\n"
            + f"med_sigma_tr  = {str(med_sigma_tr)}\n"
    )
    plt.text(.1,.8,text, fontfamily='monospace', fontsize='small', va='top',transform=ax4.transAxes)        
    
    #--------------------------    
    plt.tight_layout()
    plt.savefig('figures/prediction_plots/scatter_diagnostics_' + long_name + '.png', dpi=dpiFig)
    if(showplot==True):
        plt.show()
    else:
        plt.close()
        
    #--------------------------                    
#     fig = plt.figure()
#     ax = fig.add_axes([0.05, 0.80, 0.9, 0.1])
#     cmap = sns.color_palette("Spectral_r", as_cmap=True)
#     cb = mpl.colorbar.ColorbarBase(ax, 
#                                    orientation='horizontal',
#                                    cmap = cmap,
#                                    norm=mpl.colors.Normalize(0, 3),  # vmax and vmin                               
#                                    extend='max',
#                                    label='sigma'
#                                   )
#     plt.savefig('figures/prediction_plots/colorbarSigma' + '.png', dpi=dpiFig)
#     plt.close()

def drawOnGlobe(ax, map_proj, data, lats, lons, cmap='coolwarm', vmin=None, vmax=None, inc=None, cbarBool=True, contourMap=[], contourVals = [], fastBool=False, extent='both'):

    data_crs = ct.crs.PlateCarree()
    data_cyc, lons_cyc = add_cyclic_point(data, coord=lons) #fixes white line by adding point#data,lons#ct.util.add_cyclic_point(data, coord=lons) #fixes white line by adding point

    ax.set_global()
#     ax.coastlines(linewidth = 1.2, color='black')
#     ax.add_feature(cartopy.feature.LAND, zorder=0, scale = '50m', edgecolor='black', facecolor='black')    
    land_feature = cfeature.NaturalEarthFeature(
        category='physical',
        name='land',
        scale='50m',
        facecolor='black',
        edgecolor = None
    )
    ax.add_feature(land_feature)
#     ax.GeoAxes.patch.set_facecolor('black')
    
    if(fastBool):
        image = ax.pcolormesh(lons_cyc, lats, data_cyc, transform=data_crs, cmap=cmap)
#         image = ax.contourf(lons_cyc, lats, data_cyc, np.linspace(0,vmax,20),transform=data_crs, cmap=cmap)
    else:
        image = ax.pcolor(lons_cyc, lats, data_cyc, transform=data_crs, cmap=cmap,shading='auto')
    
    if(np.size(contourMap) !=0 ):
        contourMap_cyc, __ = add_cyclic_point(contourMap, coord=lons) #fixes white line by adding point
        ax.contour(lons_cyc,lats,contourMap_cyc,contourVals, transform=data_crs, colors='fuchsia')
    
    if(cbarBool):
        cb = plt.colorbar(image, shrink=.5, orientation="horizontal", pad=.02, extend=extent)
        cb.ax.tick_params(labelsize=6) 
    else:
        cb = None

    image.set_clim(vmin,vmax)
    
    return cb, image   

def add_cyclic_point(data, coord=None, axis=-1):

    # had issues with cartopy finding utils so copied for myself
    
    if coord is not None:
        if coord.ndim != 1:
            raise ValueError('The coordinate must be 1-dimensional.')
        if len(coord) != data.shape[axis]:
            raise ValueError('The length of the coordinate does not match '
                             'the size of the corresponding dimension of '
                             'the data array: len(coord) = {}, '
                             'data.shape[{}] = {}.'.format(
                                 len(coord), axis, data.shape[axis]))
        delta_coord = np.diff(coord)
        if not np.allclose(delta_coord, delta_coord[0]):
            raise ValueError('The coordinate must be equally spaced.')
        new_coord = ma.concatenate((coord, coord[-1:] + delta_coord[0]))
    slicer = [slice(None)] * data.ndim
    try:
        slicer[axis] = slice(0, 1)
    except IndexError:
        raise ValueError('The specified axis does not correspond to an '
                         'array dimension.')
    new_data = ma.concatenate((data, data[tuple(slicer)]), axis=axis)
    if coord is None:
        return_value = new_data
    else:
        return_value = new_data, new_coord
    return return_value

