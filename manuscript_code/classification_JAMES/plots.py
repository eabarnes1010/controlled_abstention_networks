"""Classification with abstention plotting functions."""

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


import metrics

__author__ = "Elizabeth A. Barnes and Randal J. Barnes"
__date__ = "January 11, 2021"

mpl.rcParams['figure.facecolor'] = 'white'
mpl.rcParams['figure.dpi'] = 150
dpiFig = 300.


# ----------------------------------------------------
def plot_results(EXP_NAME, history, exp_info, saveplot=True, showplot=True):

    loss_type, n_epochs, abst_setpoint, spinup_epochs, hiddens, lr_init, lr_epoch_bound, batch_size, network_seed = exp_info    
    
#     if exp_info[0].find('PID')>=0:
#         loss_type, n_epochs, abst_setpoint, spinup_epochs, hiddens, lr_init, lr_epoch_bound, batch_size, network_seed = exp_info
#     else:
#         raise ValueError('no such loss_type')

    trainColor = (117/255., 112/255., 179/255., 1.)
    valColor = (231/255., 41/255., 138/255., 1.)
    FS = 7
    plt.figure(figsize=(15, 7))

    #---------- plot loss -------------------
    plt.subplot(2,3,1)
    plt.plot(history.history['loss'], 'o', color=trainColor, label='training loss')
    plt.plot(history.history['val_loss'], 'o', color=valColor, label='validation loss')
    if 'prediction_loss' in history.history:    
        plt.plot(history.history['prediction_loss'], 'x', color=trainColor, label='training nonabstaining loss')
        plt.plot(history.history['val_prediction_loss'], 'x', color=valColor, label='validation nonabstaining loss')
    plt.title('Loss')
    plt.xlabel('epoch')
    plt.grid(True)
    plt.legend(frameon=True, fontsize=FS)
    plt.xlim(-2, n_epochs+2)
#     plt.ylim(0,1.5)

    # ---------- plot abstention -------------------
    plt.subplot(2, 3, 2)
    if loss_type!='DNN':
        plt.axhline(abst_setpoint, linestyle='-', color='gray', linewidth=3.0)

    plt.plot(history.history['abstention_fraction'], 'o', color=trainColor, label='training')
    plt.plot(history.history['val_abstention_fraction'], 'o', color=valColor, label='validation')
    plt.title('Abstention Fraction')
    plt.xlabel('epoch')
    plt.legend(frameon=True, fontsize=FS)
    plt.ylim(0, 1.02)
    plt.grid(True)    
    plt.xlim(-2, n_epochs+2)

    # ---------- plot accuracy -------------------
    plt.subplot(2, 3, 3)

    plt.plot(history.history['prediction_accuracy'], 'o', color=trainColor, label='training')
    plt.plot(history.history['val_prediction_accuracy'], 'o', color=valColor, label='validation')
    plt.title('Prediction Accuracy (nonabstention)')
    plt.xlabel('epoch')
    plt.legend(frameon=True, fontsize=FS)
    plt.ylim(0, 1.02)
    plt.grid(True)        
    plt.xlim(-2, n_epochs+2)

    # ---------- report parameters -------------------
    plt.subplot(2, 3, 4)
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

    if loss_type != 'DNN':
        text += (
            "\n"
            + f"PID ABSTENTION PARAMETERS\n"
            + f"  spinup_epochs  = {spinup_epochs}\n"
            + f"  abst_setpoint  = {abst_setpoint}\n"
        )
    else:
        text += "\nUNKNOWN LOSS TYPE\n"

    plt.text(0.01, 0.95, text, fontfamily='monospace', fontsize='small', va='top')

    plt.axis('off')

    # ---------- plot alpha -------------------
    if 'alpha_value' in history.history:
        plt.subplot(2, 3, 5)
        plt.plot(history.history['alpha_value'], 'o', color='k')
        plt.title('Alpha')
        plt.xlabel('epoch')
        plt.xlim(-2, n_epochs+2)

    # ---------- Make and save the plot -------------------
    plt.tight_layout()
    if saveplot==True:
        plt.savefig('figures/model_diagnostics/' + EXP_NAME + '.png', dpi=dpiFig)
    if showplot==False:
        plt.close('all')
    else:
        plt.show()

# ----------------------------------------------------
def plot_data(X, z, gameboard, title_str, marker_size=4, plot_type='pred'):

    # get colors and legends
    cmap = getattr(palettable.cartocolors.qualitative, 'Bold_' + str(np.max([gameboard.nlabel, 5])) + '_r')
    if plot_type == 'pred':
        colors = cmap.mpl_colors
        pal = {}
        leg_labels = {}
        keys = range(gameboard.nlabel+1)
        for i in keys:
            pal[i] = colors[i]
            leg_labels[str(i)] = 'class ' + str(i)
            if i == gameboard.nlabel:
                pal[i] = 'silver'
                leg_labels[str(i)] = 'abstain'
    elif plot_type == 'acc':
        colors = palettable.cartocolors.qualitative.Prism_7.mpl_colors
        pal = {-1: 'silver', 0: colors[-1], 1: 'black'}
        leg_labels = {'-1': 'abstain', '0': 'wrong', '1': 'correct'}
    else:
        raise ValueError('no such plot_type')

    # make figure
    plt.axis("equal")

    # plot background shading
    v_edge = np.linspace(0, gameboard.nrow, gameboard.nrank+1) - 0.5
    h_edge = np.linspace(0, gameboard.ncol, gameboard.nfile+1) - 0.5

    plt.pcolormesh(
        v_edge,
        h_edge,
        gameboard.board,
        alpha=0.20,
        edgecolors=None,
        cmap=matplotlib.colors.ListedColormap(cmap.mpl_colors[:gameboard.nlabel])
    )
    plt.clim(0, gameboard.nlabel-1)

    for i in range(gameboard.nrank):
        for j in range(gameboard.nfile):
            if gameboard.tranquil[i, j]:
                x = [h_edge[j], h_edge[j+1], h_edge[j+1], h_edge[j], h_edge[j]]
                y = [v_edge[i], v_edge[i], v_edge[i+1], v_edge[i+1], v_edge[i]]
                plt.plot(x, y, '-k', linewidth=0.25)

    # plot dots
    ax = sns.scatterplot(
        x=[xy[0] for xy in X],
        y=[xy[1] for xy in X],
        hue=z,
        palette=pal,
        zorder=10,
        s=marker_size
    )

    legend_labelhandles, labels = ax.get_legend_handles_labels()
    new_labels = [leg_labels.get(key) for key in labels]
    plt.legend(legend_labelhandles, new_labels, bbox_to_anchor=(1.0, 0.90), loc=2, borderaxespad=0)

    plt.title(title_str, fontsize='large')
    plt.axis('off')



# ----------------------------------------------------
# Make the prediction plots.

# TODO: Currently, many of these plots and statistics are legit ONLY for the "PID" controller.

def plot_predictions(exp_name, data, gameboard, saveplot=True, showplot=True):

    X_all, y_all, X_train, y_train, X_val, y_val, y_train_pred, y_val_pred, onehotlabels_val, abstain = data

    cat_train_pred = np.argmax(y_train_pred, axis=1)
    cat_val_pred = np.argmax(y_val_pred, axis=1)

    # make final predictions plot
    FS = 15
    plt.figure(figsize=(16, 24))

    # ------------------------------------------------
    # Plot the real, raw data
    plt.subplot(3, 2, 1)
    title_str = (
        f"{gameboard.nlabel}-label data with {gameboard.ntranquil} quiet cells "
        + f"and {gameboard.nnoisy} noisy cells\n"
        + f"({100*gameboard.pr_mislabel}% mislabeled in noisy cells)"
    )
    plot_data(X_all, y_all, gameboard, title_str)

    # ------------------------------------------------
    # Plot validation statistics
    num_validation = len(X_val)
    acc = metrics.compute_dac_accuracy(onehotlabels_val, y_val_pred, abstain)

    tr = gameboard.aretranquil(X_val)
    num_tranquil = np.sum(tr)
    acc_tr = metrics.compute_dac_accuracy(onehotlabels_val[tr, :], y_val_pred[tr, :], abstain)

    plt.subplot(3, 2, 2)
    plt.ylim(0, 1)
    text = (
        f"VALIDATION PREDICTIONS\n"
        + f"number = {num_validation}\n"
        + f"accuracy = {int(np.round(acc*100))}%\n"
        + f"\n"
        + f"TRANQUIL VALIDATION PREDICTIONS\n"
        + f"number = {num_tranquil}\n"
        + f"accuracy = {int(np.round(acc_tr*100))}%\n"
    )
    plt.text(.2, .5, text, fontsize=FS*1., va='top')
    plt.axis('off')

    # ------------------------------------------------
    # Plot the validation subset predictions
    plt.subplot(3, 2, 3)

    title_str = "Validation Subset Predictions"
    plot_data(X_val, cat_val_pred, gameboard, title_str, plot_type='pred')

    # ------------------------------------------------
    # Plot the validation subset accuracy
    accuracy_val = []
    for i, cat in enumerate(cat_val_pred):
        if cat == abstain:
            accuracy_val.append(-1)
        elif cat == y_val[i]:
            accuracy_val.append(1)
        else:
            accuracy_val.append(0)

    plt.subplot(3, 2, 4)
    title_str = "Validation Subset Accuracy"
    plot_data(X_val, accuracy_val, gameboard, title_str, plot_type='acc')

    # ------------------------------------------------
    # Plot the training subset predictions
    plt.subplot(3, 2, 5)
    title_str = "Training Subset Predictions"
    plot_data(X_train, cat_train_pred, gameboard, title_str, plot_type='pred')

    # ------------------------------------------------
    # Plot the training subset accuracy
    accuracy_train = []
    for i, y in enumerate(cat_train_pred):
        if y == abstain:
            accuracy_train.append(-1)
        elif y == y_train[i]:
            accuracy_train.append(1)
        else:
            accuracy_train.append(0)

    plt.subplot(3, 2, 6)
    title_str = "Training Subset Accuracy"
    plot_data(X_train, accuracy_train, gameboard, title_str, plot_type='acc')

    # ---------- Make and save the plot -------------------
    plt.tight_layout()
    if saveplot:
        plt.savefig('figures/model_predictions/' + exp_name + '.png', dpi=dpiFig)
    if not showplot:
        plt.close('all')
    else:
        plt.show()

    print(f"Validation = {Counter(accuracy_val)}")
    print(f"Training = {Counter(accuracy_train)}")

    

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




def plot_stats_comparisons(df, savename, lines=False, shades=True):
    
    ABSTENTION_VAR = 'coverage'
    plt.figure(figsize=(6*2,4*3))
    plt.rcParams["axes.prop_cycle"] = plt.cycler("color", palettable.cartocolors.qualitative.Vivid_6.mpl_colors)    

    #-------------------
    s0 = plt.subplot(3,2,1)
    plot_stats_panel(df, var='acc', abstention_var=ABSTENTION_VAR, legend=True, lines=lines, shades=shades)

    #-------------------
    s1 = plt.subplot(3,2,2)
    plot_stats_panel(df, var='acc_portion_tr', abstention_var=ABSTENTION_VAR, legend=False, lines=lines, shades=shades)   
    plt.title('Contributions of Tranquil to Total Accuracy', fontsize=8)           
    
    #-------------------
    s5 = plt.subplot(3,2,5)
    plot_stats_panel(df, var='frac_corr_tr', abstention_var=ABSTENTION_VAR, legend=False, lines=lines, shades=shades)
    plt.title('Fraction of those Correct that are Tranquil', fontsize=8)           

    #-------------------
    s3 = plt.subplot(3,2,3)
    plot_stats_panel(df, var='acc_tr', abstention_var=ABSTENTION_VAR, legend=False, lines=lines, shades=shades)
    plt.title('Tranquil Accuracy', fontsize=8)        

    #-------------------
    s4 = plt.subplot(3,2,4)
    plot_stats_panel(df, var='acc_ntr', abstention_var=ABSTENTION_VAR, legend=False, lines=lines, shades=shades)
    plt.title('Not-Tranquil Accuracy', fontsize=8)           

    #-------------------
    s6 = plt.subplot(3,2,6)
    plot_stats_panel(df, var='frac_tr', abstention_var=ABSTENTION_VAR, legend=False, lines=lines, shades=shades)
    plt.title('Fraction Covered that are Tranquil', fontsize=8)           


    #----------------------------------------------------------
    s0.set_title('Total Accuracy\n' + savename, fontsize=8)        
    plt.tight_layout()
    
    return savename



def plot_stats_panel(df, var, abstention_var, legend=False, lines=False, shades=True):
    alpha = .2
    
    colors = palettable.cartocolors.qualitative.Bold_9_r.mpl_colors
    
    APP_LIST = np.unique(df['app_type'].values)
    APP_LIST = sorted(APP_LIST,reverse=False)
    for k,app_type in enumerate(APP_LIST):

        edgeclr='white'
        label=app_type
        if(app_type=='ORACLE'):
            clr = 'lightgray'
        elif((app_type=='DNN' or app_type=='ANN') and var=='acc'):
            clr = colors[0]
        elif((app_type=='DNN' or app_type=='ANN') and var=='acc_tr'):
            clr = colors[1]    
            label='ANN: tranquil'
        elif((app_type=='DNN' or app_type=='ANN') and var=='acc_ntr'):
            clr = (.25,.25,.25,1.)
            label='ANN: not tranquil'
        elif((app_type=='CAN' or app_type=='DAC') and var=='acc_tr'):    
            clr = colors[1]                
            label='CAN: tranquil'
        elif((app_type=='CAN' or app_type=='DAC') and var=='acc_ntr'):
            clr=(.25,.25,.25,1.)
            label='CAN: not tranquil'            
            edgeclr=(.25,.25,.25,1.)
        else:
            clr = colors[k]
        
        df_plot = df[df['app_type']==app_type]
        
        if(lines==True and shades==True):
            lines_label = None
            args = {'color':clr,}
        else:
            lines_label = app_type
            args={}
        
        if((app_type=='DNN' or app_type=='ANN') and var=='acc'):
            lines_label=None#'DNN median'
            
        if(app_type=='CAN' or app_type=='DAC'):
            ax1 = sns.scatterplot(data=df_plot, x=abstention_var, y=var,
                                 hue="setpoint",
                                 palette = palettable.cartocolors.qualitative.Vivid_5.mpl_colormap,
                                 label = label, 
                                 legend = False,
                                 ci = None, 
                                 markers=True, 
                                 s = 15,
                                 style="app_type",
                                 alpha = .6,
                                 zorder = 100,
                                 edgecolor=edgeclr, 
                                )          
        else:
            if(lines==True):
                sns.lineplot(data=df_plot, x=abstention_var, y=var, 
                                  label = lines_label, 
                                  legend = False,
                                  ci = None, 
                                  estimator = np.median, 
                                  markers=False, 
                                  style="app_type",
                                  linewidth=1.5,
                                  **args,
                                 )
            if(shades==True):
                df_stats = df_plot.groupby([abstention_var]).describe()
                plt.fill_between(x=df_stats.index, y1=df_stats[(var, 'min')], y2=df_stats[(var, 'max')], alpha = alpha, label=label, color=clr)

    #---------------------------------------

    plt.legend(loc=2,)
    if(legend is False):
        try:
            ax1.get_legend().remove()                       
        except:
            print('no such ax1')
    
    if(var=='frac_tr'):
        ax9 = sns.lineplot(data=df[df['app_type']=='ANN'], x=abstention_var, y='perf_frac_tr', 
                          label = None, 
                          legend = None,
                          ci = None, 
                          estimator = np.median, 
                          markers=False,
                          color = 'black',
                          dashes=[(2,2)],
                          style="app_type",
                          alpha = .2,
                         )
    xticks = np.arange(0,110,10)    
    plt.xticks(xticks,map(str,xticks))
    plt.xlabel('coverage (\%)', fontsize=16)
    plt.ylim(-.02,1.01)
    plt.xlim(-1,100)    
    plt.gca().invert_xaxis()
    ylabel = plt.gca().yaxis.get_label().get_text()
    if(ylabel.find('\_')==-1):
        plt.ylabel(ylabel.replace('_','\_'), fontsize=16)
    
    

    try:
        return ax1
    except:
        return None
  