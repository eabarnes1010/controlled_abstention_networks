"""Regression with abstention plotting function."""

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors
import seaborn as sns
import numpy as np
import warnings

__author__ = "Elizabeth A. Barnes and Randal J. Barnes"
__date__ = "April 16, 2021"

#----------------------------------------------------------------  

mpl.rcParams['figure.facecolor'] = 'white'
mpl.rcParams['figure.dpi'] = 150
dpiFig = 300.

# =============================================================================

        
        
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
#         plt.legend()
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
#     plt.legend()
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
    plt.savefig('figures/scatter_diagnostics_' + long_name + '.png', dpi=dpiFig)
    if(showplot==True):
        plt.show()
    else:
        plt.close()
