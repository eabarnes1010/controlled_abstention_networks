# Elizabeth A. Barnes and Randal J. Barnes
# January 18, 2021
# v2.08

import numpy as np
import numpy.ma as ma
import random
import xarray as xr
import scipy.stats as stats
import random
import sys
import h5py
from sklearn import preprocessing
import matplotlib.pyplot as plt
import scipy.io as sio

#----------------------------------------------------------------    
#----------------------------------------------------------------    

def is_tranquil(N,inoise):
    tr = np.ones((N,))
    i = np.intersect1d(np.arange(0,N),inoise)
    tr[i] = 0
    return tr


#----------------------------------------------------------------    
def make_categories(y,n,y_perc=np.nan):

    p_vec = np.linspace(0,1,n+1)
    ycat = np.empty(np.shape(y))*np.nan
    y_p = np.empty((len(p_vec)-1,2))*np.nan    

    if(y_perc is not np.nan):
        assert len(p_vec)==len(y_perc)+1
    
    for i in np.arange(0,n):
        if(y_perc is np.nan):
            low = np.percentile(y,p_vec[i]*100)
            high = np.percentile(y,p_vec[i+1]*100)
        else:
            low,high = y_perc[i]

        y_p[i,:] = low, high            
        k = np.where(np.logical_and(y>=low, y<=high))[0]
        ycat[k] = i
    
    return ycat, y_p


#----------------------------------------------------------------    
def undersample(x,y,tr):
    
    cats = np.unique(y)
    mincount = len(y)
    for c in cats:
        num = np.count_nonzero(y==c)
        if(num < mincount):
            mincount = num
    # print results
    for c in cats:
        print('number originally in each class ' 
              + str(c) 
              + ' = ' 
              + str(np.count_nonzero(y==c)))
    
    yout = y[:1]
    xout = x[:1]
    trout = tr[:1]
    for c in cats:
        i = np.where(y==c)[0]
        i = i[:mincount]
        xout = np.append(xout,x[i],axis=0)
        yout = np.append(yout,y[i],axis=0)
        trout = np.append(trout,tr[i],axis=0)
   
    yout = np.asarray(yout[1:])
    xout = np.asarray(xout[1:])  
    trout = np.asarray(trout[1:])      

    # print results
    for c in cats:
        print('number in undersampled class ' 
              + str(c) 
              + ' = ' 
              + str(np.count_nonzero(yout==c)))
    
    return xout, yout, trout


#----------------------------------------------------------------
def split_data(X, y_cat, tranquil, corrupt):
    
    NSAMPLES = np.shape(X)[0]
    n_test = 5000
    n_val = 5000
    n_train = NSAMPLES - n_val - n_test

    
    # make training data
    X_train = X[:n_train,:,:]
    y_train_cat = y_cat[:n_train]
    tr_train = tranquil[:n_train]
    corrupt_train = corrupt[:n_train]
    print('----Training----')
    print(np.shape(X_train))
    print(np.shape(y_train_cat))
    print(np.shape(tr_train))    

    # make validation data
    X_val = X[n_train:n_train+n_val,:,:]
    y_val_cat = y_cat[n_train:n_train+n_val]
    tr_val = tranquil[n_train:n_train+n_val]    
    corrupt_val = corrupt[n_train:n_train+n_val]    
    print('----Validation----')
    print(np.shape(X_val))
    print(np.shape(y_val_cat))
    print(np.shape(tr_val))    

    # make testing data
    X_test = X[n_train+n_val:n_train+n_val*2,:,:]
    y_test_cat = y_cat[n_train+n_val:n_train+n_val*2]
    tr_test = tranquil[n_train+n_val:n_train+n_val*2]    
    corrupt_test = corrupt[n_train+n_val:n_train+n_val*2]    
    print('----Testing----')
    print(np.shape(X_test))
    print(np.shape(y_test_cat))
    print(np.shape(tr_test))    
    
    return (X_train, y_train_cat, tr_train, corrupt_train), (X_val, y_val_cat, tr_val, corrupt_val), (X_test, y_test_cat, tr_test, corrupt_test)


#----------------------------------------------------------------    
def preprocess_data(X_train, y_train, X_val, y_val, ABSTAIN):

    # standardize data
    xmean = np.nanmean(X_train[:]) #standardize here
    xstd = np.nanstd(X_train[:])
    X_train_std = (X_train-xmean)/xstd
    X_val_std = (X_val-xmean)/xstd
    
    # reshape the data
    X_train_std = np.reshape(X_train_std, (np.shape(X_train_std)[0],np.shape(X_train_std)[1]*np.shape(X_train_std)[2]))
    X_val_std = np.reshape(X_val_std, (np.shape(X_val_std)[0],np.shape(X_val_std)[1]*np.shape(X_val_std)[2]))

    # turn nan values (which correspond to land) to zero so they do not influence the training
    X_train_std = np.nan_to_num(X_train_std, copy=False, nan=0.0)
    X_val_std = np.nan_to_num(X_val_std, copy=False, nan=0.0)
    
    # one-hot encoding
    enc = preprocessing.OneHotEncoder()
    enc.fit(np.append(y_train, ABSTAIN)[:,np.newaxis])
    onehotlabels = enc.transform(np.array(y_train).reshape(-1, 1)).toarray()
    onehotlabels_val = enc.transform(np.array(y_val).reshape(-1, 1)).toarray()   
    
    return X_train_std, onehotlabels, X_val_std, onehotlabels_val, xmean, xstd 

#----------------------------------------------------------------    
def preprocess_reg_data(X_train, y_train, X_val, y_val, ABSTAIN):

    # standardize data
    xmean = np.nanmean(X_train[:]) #standardize here
    xstd = np.nanstd(X_train[:])
    X_train_std = (X_train-xmean)/xstd
    X_val_std = (X_val-xmean)/xstd
    
    # reshape the data
    X_train_std = np.reshape(X_train_std, (np.shape(X_train_std)[0],np.shape(X_train_std)[1]*np.shape(X_train_std)[2]))
    X_val_std = np.reshape(X_val_std, (np.shape(X_val_std)[0],np.shape(X_val_std)[1]*np.shape(X_val_std)[2]))

    # turn nan values (which correspond to land) to zero so they do not influence the training
    X_train_std = np.nan_to_num(X_train_std, copy=False, nan=0.0)
    X_val_std = np.nan_to_num(X_val_std, copy=False, nan=0.0)
    
    # one-hot encoding for regression
    onehotlabels = np.append(y_train,np.zeros(shape=np.shape(y_train)),axis=1)
    onehotlabels_val = np.append(y_val,np.zeros(shape=np.shape(y_val)),axis=1)    
    
    return X_train_std, onehotlabels, X_val_std, onehotlabels_val, xmean, xstd 



#----------------------------------------------------------------    
def load_data():

    data_filename = '../data/synthClimate/synth_exm_Libby.mat'

    with h5py.File(data_filename, 'r') as file:
        print(list(file.keys()))

    with h5py.File(data_filename, 'r') as file:
        X = np.array(file['SSTrand'])    
        lat = np.array(file['lat'])    
        lon = np.array(file['lon'])
    #     Cnt = list(file['Cnt'])
        y = np.array(file['y'])

    X = np.swapaxes(X,1,2)    

    return X, y, lat, lon

#----------------------------------------------------------------    
def load_simpledata(size=None):
    if(size is None):
        data_filename = '../data/synthClimate/simple_climatedata.mat'    
    else:
        data_filename = '../data/synthClimate/simple_climatedata_' + size + '.mat'    
    mat_contents = sio.loadmat(data_filename)
    
    X = np.array(mat_contents['X'])    
    lat = np.array(mat_contents['lat'])    
    lon = np.array(mat_contents['lon'])
#     C = np.array(mat_contents['Cnt'])
    y = np.array(mat_contents['y'])[0,:][:,np.newaxis]

    return X, y, lat, lon
#----------------------------------------------------------------    
def add_noise(data_name, X, y, lat, lon, pr_noise, nlabel, y_perc = np.nan, cutoff = 1., region_name='ENSO'):

    lat = lat.flatten()
    lon = lon.flatten()
    
    y_cat_orig, y_perc_out = make_categories(y,nlabel,y_perc)
    NSAMPLES = np.shape(X)[0]
    
    #-----------------------------------------------       
    # define the tranquil and noisy regions
    if data_name.find('badClasses')==0:
        inoise = np.where(np.isin(y_cat_orig, [cutoff]))[0]
        tranquil = is_tranquil(NSAMPLES,inoise)
        
    elif data_name.find('mixedLabels')==0:
        inoise = np.arange(0,len(y_cat_orig))
        tranquil = is_tranquil(NSAMPLES,inoise)

    elif data_name.find('tranquilFOO') == 0:      
        reg_lats, reg_lons = get_region(region_name = region_name)
        ilat = np.where(np.logical_and(lat>=reg_lats[0],lat<=reg_lats[1]))[0]
        ilon = np.where(np.logical_and(lon>=reg_lons[0],lon<=reg_lons[1]))[0]
        reg_avg = np.nansum(np.nansum(X[:,:,ilon],axis=-1)[:,ilat],axis=-1)/(len(ilat)*len(ilon))
        print('region shape = ' + str(len(ilat)) + ' x ' + str(len(ilon)))
        
        inoise = np.where(reg_avg<=cutoff)[0]
        tranquil = is_tranquil(NSAMPLES,inoise)
        
    elif data_name.find('noNoise') == 0:
        inoise = []
        tranquil = np.ones((NSAMPLES,))
        
    else:
        raise ValueError('no such data noise type.')
    
    #-----------------------------------------------       
    #change the labels on the noisy data
    y_cat = np.copy(y_cat_orig)   
    corrupt = np.zeros(np.shape(y_cat_orig))
    
    for i in inoise:
        if np.random.random() < pr_noise:
            label = y_cat_orig[i]                
            label = (label + np.random.randint(low=1, high=nlabel)) % nlabel
            corrupt_value = 1
        else:
            label = y_cat_orig[i]
            corrupt_value = 0
        y_cat[i] = label
        corrupt[i] = corrupt_value
         
    itranquil = np.where(tranquil==1)[0]
    i = np.where(y_cat_orig!=y_cat)[0]
    k = np.where(y_cat_orig[itranquil]!=y_cat[itranquil])[0]
    print('\n----Mislabeled----')
    print('# tranquil = ' + str(len(itranquil)) + ' out of ' + str(NSAMPLES) + ' samples')
    print('percent tranquil = ' + str(np.round(100.*len(itranquil)/len(tranquil))) + '%')
    try:
        print('tranquil mislabeled = ' + str(np.round(100.*len(k)/len(itranquil))) + '%')        
        print('non-tranquil mislabeled = ' + str(np.round(100.*len(i)/len(inoise))) + '%')
    except:
        pass
    print('total mislabeled = ' + str(np.round(100.*len(i)/len(y_cat))) + '%')
    #-----------------------------------------------  
    
    return X, y_cat, tranquil, corrupt, y_perc_out
    
    #----------------------------------------------------------------    
def add_reg_noise(data_name, X, y, lat, lon, pr_noise, nlabel, y_perc = np.nan, cutoff = 1., region_name='ENSO'):

    lat = lat.flatten()
    lon = lon.flatten()
    
    NSAMPLES = np.shape(X)[0]
    
    #-----------------------------------------------       
    # define the tranquil and noisy regions
    if data_name.find('mixedLabels')==0:
        inoise = np.arange(0,len(y))
        tranquil = is_tranquil(NSAMPLES,inoise)

    elif data_name.find('tranquilFOO') == 0:      
        reg_lats, reg_lons = get_region(region_name = region_name)
        ilat = np.where(np.logical_and(lat>=reg_lats[0],lat<=reg_lats[1]))[0]
        ilon = np.where(np.logical_and(lon>=reg_lons[0],lon<=reg_lons[1]))[0]
        reg_avg = np.nansum(np.nansum(X[:,:,ilon],axis=-1)[:,ilat],axis=-1)/(len(ilat)*len(ilon))
        print('region shape = ' + str(len(ilat)) + ' x ' + str(len(ilon)))
        
        inoise = np.where(reg_avg<=cutoff)[0]
        tranquil = is_tranquil(NSAMPLES,inoise)
        
    elif data_name.find('noNoise') == 0:
        inoise = []
        tranquil = np.ones((NSAMPLES,))
        
    else:
        raise ValueError('no such data noise type.')
    
    #-----------------------------------------------       
    #change the labels on the noisy data
    y_new = np.copy(y)   
    corrupt = np.zeros(np.shape(tranquil))
    
    for i in inoise:
        if np.random.random() < pr_noise:
            i_rand_sample = np.random.randint(0,NSAMPLES)
            value = y[i_rand_sample]
            corrupt_value = 1
        else:
            value = y[i]
            corrupt_value = 0
        y_new[i] = value
        corrupt[i] = corrupt_value
         
    itranquil = np.where(tranquil==1)[0]
    i = np.where(corrupt==1)[0]
    k = np.where(np.logical_and(tranquil==1, corrupt==1))[0]
    print('\n----Mislabeled----')
    print('# tranquil = ' + str(len(itranquil)) + ' out of ' + str(NSAMPLES) + ' samples')
    print('percent tranquil = ' + str(np.round(100.*len(itranquil)/len(tranquil))) + '%')
    try:
        print('tranquil mislabeled = ' + str(np.round(100.*len(k)/len(itranquil))) + '%')        
        print('non-tranquil mislabeled = ' + str(np.round(100.*len(i)/len(inoise))) + '%')
    except:
        pass
    print('total mislabeled = ' + str(np.round(100.*len(i)/len(y_new))) + '%')
    #-----------------------------------------------  
    
    return X, y_new, tranquil, corrupt

def get_region(region_name):
    
    if(region_name=='ENSO'):
        reg_lats = (-10.,10.)
        reg_lons = (360-170,360-82.)
    elif(region_name=='equatorialENSO'):
        reg_lats = (-5.,5.)
        reg_lons = (360-170,360-82.)
    elif(region_name=='shENSO'):
        reg_lats = (-10.,5.)
        reg_lons = (360-170,360-82.)
    elif(region_name=='nhENSO'):
        reg_lats = (5.,10.)
        reg_lons = (360-170,360-82.)
    elif(region_name=='tropicsENSO'):
        reg_lats = (-15.,15.)
        reg_lons = (360-170,360-82.)
    elif(region_name=='smallENSO'):
        reg_lats = (-5.,5.)
        reg_lons = (360-120,360-82.)
    elif(region_name=='NAO'):
        reg_lats = (40.,60.)
        reg_lons = (360-60.,360-20.)
    else:
        raise ValueError('no such region')
    
#     print(region_name)
    return reg_lats, reg_lons
