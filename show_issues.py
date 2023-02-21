#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 13:52:46 2022

@author: kangming

The following files under the "data" folder are required to run the code:
    megnet_273feat.csv: 
        featurized MP18 dataset
    MP_alloys_2021.11.10_273feat.csv: 
        featurized MP21 alloy dataset
    mp_e_form_alignnn_MP21_alloys.csv: 
        formation energy prediction on the MP21 alloy dataset, using 
        the MP18-pretrained ALIGNN model
    megnet_updated_id_MP21.csv:
        material_id in MP21 for each MP18 entry

"""

import time
import os

from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import r_regression
from lineartree import LinearForestRegressor
import xgboost as xgb
from jarvis.core.atoms import Atoms
import ast
import scipy
import matplotlib.colors as colors

from myfunc import custom_matplotlib, count_sg_num
custom_matplotlib()
ccycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

# Create folder to store figures
if not os.path.exists('figs'):
   os.makedirs('figs')

#%%
'''
Define the classical ML models
'''
random_state = 1
n_estimators = 100
max_features = 0.3

pipe={}
pipe['XGB'] = Pipeline([
    ('imputer', SimpleImputer()), 
    ('scaler', StandardScaler()),
    ('model', xgb.XGBRegressor(
                    n_estimators=2000,
                    learning_rate=0.1,
                    reg_lambda=0, # L2 regularization
                    reg_alpha=0.1,# L1 regularization
                    # max_depth=6,
                    tree_method='gpu_hist', gpu_id=0))
])

pipe['RF'] = Pipeline([
    ('imputer', SimpleImputer()), 
    ('scaler', StandardScaler()),
    ('model', RandomForestRegressor(n_estimators=n_estimators, 
                                    max_features=max_features, 
                                    n_jobs=-1, random_state=random_state))
])
pipe['LF'] = Pipeline([
    ('imputer', SimpleImputer()), 
    ('scaler', StandardScaler()),
    ('model', LinearForestRegressor(base_estimator=Ridge(),
                                    n_estimators=n_estimators, 
                                    max_features=max_features,
                                    n_jobs=-1, random_state=1))
])

# Features to be used in these models
important_features = ['mean CN_VoronoiNN', 'mean ordering parameter shell 1', 'mean neighbor distance variation', 'avg_dev CN_VoronoiNN', 'mean local difference in NValence', 'MagpieData mean NpUnfilled', 'MagpieData mean NsUnfilled', 'minimum local difference in Number', 'MagpieData mode GSmagmom', 'minimum local difference in Column', 'MagpieData mode NfUnfilled', 'MagpieData mode GSbandgap', 'MagpieData maximum MeltingT', 'avg_dev local difference in NdValence', 'minimum local difference in NpUnfilled', 'MagpieData maximum CovalentRadius', 'MagpieData mode NValence', 'MagpieData range NdUnfilled', 'range local difference in NfValence', 'avg_dev local difference in CovalentRadius', 'minimum local difference in NdValence', 'MagpieData mean NUnfilled', 'MagpieData minimum AtomicWeight', 'MagpieData mode NdUnfilled', 'minimum local difference in NdUnfilled', 'MagpieData mean MeltingT', 'avg_dev local difference in NValence', 'minimum local difference in MeltingT', 'range local difference in NUnfilled', 'MagpieData minimum NValence', 'MagpieData minimum NsUnfilled', 'minimum local difference in NpValence', 'mean ordering parameter shell 3', 'MagpieData minimum GSvolume_pa', 'minimum local difference in GSvolume_pa', 'MagpieData maximum Column', 'frac d valence electrons', 'MagpieData mode NpUnfilled', 'avg_dev local difference in GSbandgap', 'MagpieData minimum NdValence', 'minimum local difference in CovalentRadius', 'MagpieData avg_dev Row', 'MagpieData minimum Electronegativity', '0-norm', 'MagpieData maximum SpaceGroupNumber', 'MagpieData range Electronegativity', 'compound possible', 'range local difference in Column', 'MagpieData mode NsValence', 'MagpieData mode NfValence', 'minimum local difference in NsUnfilled', 'MagpieData mode NUnfilled', 'minimum neighbor distance variation', 'MagpieData mean MendeleevNumber', 'MagpieData avg_dev GSvolume_pa', 'minimum local difference in GSmagmom', 'minimum local difference in GSbandgap', 'frac s valence electrons', 'MagpieData minimum NfValence', 'MagpieData maximum Row', 'MagpieData minimum GSmagmom', 'MagpieData range NpUnfilled', 'range local difference in Row', 'avg_dev local difference in NsValence', 'MagpieData minimum GSbandgap', 'mean local difference in SpaceGroupNumber', 'MagpieData minimum NdUnfilled', 'MagpieData minimum NUnfilled', 'minimum local difference in NfUnfilled', 'minimum local difference in NfValence', 'MagpieData minimum NpUnfilled', 'MagpieData mode NsUnfilled', 'avg_dev local difference in MendeleevNumber', 'max relative bond length', 'avg_dev local difference in AtomicWeight', '10-norm', 'avg_dev neighbor distance variation', 'minimum local difference in NUnfilled', 'MagpieData minimum NfUnfilled', 'MagpieData mode Column', 'MagpieData avg_dev MendeleevNumber', 'MagpieData mode SpaceGroupNumber', 'range local difference in NfUnfilled', 'MagpieData mode GSvolume_pa', 'min relative bond length', 'MagpieData maximum NdValence', 'maximum CN_VoronoiNN', 'avg_dev local difference in NpValence', 'MagpieData avg_dev GSmagmom', 'avg_dev local difference in NpUnfilled']

#%%
''' 
Load data 
'''

# Load the whole MP18 data
dat_MP18 = pd.read_csv(
    'data/megnet_273feat.csv',index_col='id'
    ).drop(columns=['Unnamed: 0']).sample(frac=1,random_state=random_state)

# Change the index column name
dat_MP18.index.names = ['material_id']

# Load the alloys of interest (AoI) in MP21
dat_MP21 = pd.read_csv(
    'data/MP_alloys_2021.11.10_273feat.csv',index_col='material_id'
    ).sample(frac=1,random_state=random_state)

# Input features
X_MP18 = dat_MP18[important_features]
X_MP21 = dat_MP21[important_features]

# Labels
y_MP18 = dat_MP18.rename(
    columns={'e_form':'formation_energy_per_atom'}
    )['formation_energy_per_atom']

# y_MP21 will be a df containing columns of DFT ground truth, and ALIGNN, RF, XGB, LF predictions
y_MP21 = dat_MP21['formation_energy_per_atom'].to_frame()
y_MP21.columns=['DFT']

# MP18-pretrained ALIGNN prediction on the MP21 
y_MP21['ALIGNN'] = pd.read_csv(
    'data/mp_e_form_alignnn_MP21_alloys.csv',index_col='material_id'
    )

# The material_id of some entries in MP18 have changed. Their updated id are obtained
# using update_material_id.py.
megnet_updated_id_MP21 = pd.read_csv('data/megnet_updated_id_MP21.csv',index_col='id_MP21')

# Separate the AoI existing in MP18 from the new AoI existing only in MP21
# Note that these ids are the ids in MP21
old = list(set(dat_MP21.index) & set(megnet_updated_id_MP21.index))
new = list(set(dat_MP21.index) - set(old))

#%%
'''
Get the MP18 performance metrics for classical ML models
'''
n_train, n_val, n_test = 60000, 5000, 4239
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X_MP18, y_MP18, test_size=n_test, random_state=random_state+1
    )

for ml in ['LF', 'RF', 'XGB']:
    start = time.time()
    pipe[ml].fit(X_train_val, y_train_val)
    end = time.time()
    train_time = end-start
    y_pred = pipe[ml].predict(X_test)
    
    maes = metrics.mean_absolute_error(y_test,y_pred)
    rmse = metrics.mean_squared_error(y_test,y_pred,squared=False)
    r2 = metrics.r2_score(y_test,y_pred)
    print(f'Time for training {ml}: {train_time:.1f} seconds')
    print(f'Test scores ({ml}): MAE={maes:.3f}, RMSE={rmse:.3f}, r2={r2:.3f}')

#%%
'''
Get the statistics
'''
print('Results in Table II:')


print('Get the statistics of MP18')
# Min, Max, std, mean
print('Min,Max,STD,Mean:')
print(y_MP18.describe())
# MAD
print('MAD: ',(y_MP18 - y_MP18.mean()).abs().mean())


print('Get the statistics of MP21 AoI')
print('Min,Max,STD,Mean:')
print(y_MP21['DFT'].describe())
print('MAD: ',(y_MP21['DFT'] - y_MP21['DFT'].mean()).abs().mean())


print('Get the statistics of MP18 AoI')
print('Min,Max,STD,Mean:')
print(y_MP21.loc[old,'DFT'].describe())
print('MAD: ',(y_MP21.loc[old,'DFT'] - y_MP21.loc[old,'DFT'].mean()).abs().mean())

#%%
'''
Train classical models on the whole MP18 and predict on the MP21 alloys 

'''

# Train classical models on the whole MP18
X = X_MP18
y = y_MP18
for ml in ['LF', 'RF', 'XGB']:
    start = time.time()
    pipe[ml].fit(X,y)
    end = time.time()
    train_time = end-start
    print(f'Time for training {ml} on MP18: {train_time:.1f} seconds')
    y_MP21[ml] = pd.DataFrame(
        pipe[ml].predict(X_MP21), index = X_MP21.index, columns=[ml]
        )
    
# Get metrics on the MP21 test AoI
y_err = pd.DataFrame()
maes = {}
rmse = {}
r2 = {}
for ml in ['ALIGNN','XGB','RF','LF']:
    y_err[ml] = y_MP21.loc[new,ml] - y_MP21.loc[new,'DFT']
    maes[ml] =  metrics.mean_absolute_error(y_MP21.loc[new,'DFT'],y_MP21.loc[new,ml])
    rmse[ml] = metrics.mean_squared_error(y_MP21.loc[new,'DFT'],y_MP21.loc[new,ml],squared=False)
    r2[ml] = metrics.r2_score(y_MP21.loc[new,'DFT'],y_MP21.loc[new,ml])
    maes[ml] = round(maes[ml],3)
    rmse[ml] = round(rmse[ml],3)
    r2[ml] = round(r2[ml],3)
df_metrics=pd.DataFrame([maes,rmse,r2],index=['maes','rmse','r2']).transpose()

print('Results in Table III:')
print(df_metrics)

#%%
'''
Calculate correlation coefficients between predictions by the ALIGNN and 
one of classical models

'''
r_wrt_alignn={'r_p':{},'r_k':{},'r_s':{}}
for ml in ['LF', 'RF', 'XGB']:
    r_wrt_alignn['r_p'][ml] = r_regression(
        y_MP21.loc[new,'ALIGNN'].to_frame(),y_MP21.loc[new,ml]
        )[0]
    r_wrt_alignn['r_k'][ml] = scipy.stats.kendalltau(
        y_MP21.loc[new,'ALIGNN'].to_frame(),y_MP21.loc[new,ml]
        )[0]
    r_wrt_alignn['r_s'][ml] = scipy.stats.spearmanr(
        y_MP21.loc[new,'ALIGNN'].to_frame(),y_MP21.loc[new,ml]
        )[0]

r_wrt_alignn=pd.DataFrame(r_wrt_alignn)
print('Correlation between predictions by the ALIGNN and one of classical models:')
print(r_wrt_alignn)

#%%
'''
Plot the ALIGNN vs. DFT comparison 
'''

''' Scatter plot '''

linewidth = 0.6
scale = [-1.,4.5]
fig, ax1 = plt.subplots(nrows=2,figsize=(3.6,3.4*2))
ax = ax1[0]
ax.plot(scale,scale,'k',linewidth=linewidth)
ax.scatter(y_MP21['DFT'].loc[new], y_MP21['ALIGNN'].loc[new],
            marker='o', facecolors='none', edgecolors=ccycle[0],
            linewidth=linewidth,
            label='AoI in test set')
ax.scatter(y_MP21['DFT'].loc[old], y_MP21['ALIGNN'].loc[old],
            marker='s', facecolors=ccycle[1], edgecolors='k',
            linewidth=linewidth-0.25,
            label='AoI in train set')
ax.set_xticks(np.arange(-2,4.7,1))
ax.set_yticks(np.arange(-2,4.7,1))
ax.set_xlim(scale)
ax.set_ylim(scale)
ax.grid()
# ax.set_xlabel('$E_f^{DFT}$ (eV/atom)')
ax.set_ylabel('$E_f^{ALIGNN}$ (eV/atom)')
ax.legend(loc='upper left')

'''
Plot the ALIGNN errors
'''
ax = ax1[1]
ax.scatter(y_MP21['DFT'].loc[new], -y_err['ALIGNN'].loc[new],
            marker='o', facecolors='none', edgecolors=ccycle[0],
            linewidth=linewidth,label='AoI in test set')
ax.scatter(y_MP21['DFT'].loc[old], y_MP21.loc[old,'DFT'] - y_MP21.loc[old,'ALIGNN'],
            marker='s', facecolors=ccycle[1], edgecolors='k',
            linewidth=linewidth-0.25,
            label='AoI in train set')
ax.set_xticks(np.arange(-2,5,1))
ax.set_yticks(np.arange(-5,5,1))
ax.tick_params(axis='x', which='minor', bottom=True)
ax.grid()
ax.legend(loc='upper left')
ax.set_xlim([-1,4.5])
ax.set_ylim([-1,4.5])
ax.set_xlabel('$E_f^{DFT}$ (eV/atom)')
ax.set_ylabel('$E_f^{DFT}$ - $E_f^{ALIGNN}$ (eV/atom)')
# ax.legend(loc='upper right')
fig.savefig('figs/raise_issue_plot_alignn_scatter.png')

#%%
'''
hexbin plot
'''

linewidth = 0.36
scale = [-1.,4.5]

fig, ax1 = plt.subplots(nrows=2,figsize=(4.6,3.5*2))

ax = ax1[0]
ax.plot(scale,scale,'k',linewidth=linewidth)

d_grid = 0.2
px = y_MP21['DFT'].loc[new]
py = y_MP21['ALIGNN'].loc[new]
gridsize_x = int((max(px)-min(px))/d_grid)
gridsize_y = int((max(py)-min(py))/d_grid)
gridsize = (gridsize_x,gridsize_y)
hb = ax.hexbin(px, py,label='AoI in test set',
          gridsize=gridsize,cmap ='Blues',norm=colors.LogNorm(vmin=1,vmax=50))
cb = fig.colorbar(hb, ax=ax)
cb.set_label('Counts',fontsize=10)


ax.scatter(y_MP21['DFT'].loc[old], y_MP21['ALIGNN'].loc[old],
            marker='o',facecolors='none',edgecolors=ccycle[1],linewidth=1,
            s=15, label='AoI in train set')

ax.set_xticks(np.arange(-2,4.7,1))
ax.set_yticks(np.arange(-2,4.7,1))
ax.set_xlim(scale)
ax.set_ylim(scale)
ax.grid()
# ax.set_xlabel('$E_f^{DFT}$ (eV/atom)')
ax.set_ylabel('$E_f^{ALIGNN}$ (eV/atom)')

lgnd = ax.legend(loc=(0.01,0.75))
lgnd.legendHandles[1]._sizes = [60]

'''
Plot the ALIGNN errors
'''
ax = ax1[1]

d_grid = 0.2
px = y_MP21['DFT'].loc[new]
py = -y_err['ALIGNN'].loc[new]
gridsize_x = int((max(px)-min(px))/d_grid)
gridsize_y = int((max(py)-min(py))/d_grid)
gridsize = (gridsize_x,gridsize_y)
hb = ax.hexbin(px, py,label='AoI in test set',
          gridsize=gridsize,cmap ='Blues',norm=colors.LogNorm(vmin=1,vmax=50))
cb = fig.colorbar(hb, ax=ax)
cb.set_label('Counts',fontsize=10)

ax.scatter(y_MP21['DFT'].loc[old], y_MP21.loc[old,'DFT'] - y_MP21.loc[old,'ALIGNN'],
            # marker='o',facecolors=ccycle[1], edgecolors='k',linewidth=0.5,
            marker='o',facecolors='none',edgecolors=ccycle[1],linewidth=1,
            s=15,label='AoI in train set')
ax.set_xticks(np.arange(-2,5,1))
ax.set_yticks(np.arange(-5,5,1))
ax.tick_params(axis='x', which='minor', bottom=True)
ax.grid()
ax.set_xlim([-1,4.5])
ax.set_ylim([-1,4.5])
ax.set_xlabel('$E_f^{DFT}$ (eV/atom)')
ax.set_ylabel('$E_f^{DFT}$ - $E_f^{ALIGNN}$ (eV/atom)')
lgnd = ax.legend(loc=(0.01,0.78))
lgnd.legendHandles[1]._sizes = [60]


fig.savefig('figs/raise_issue_plot_alignn.png')






#%%
'''
Plot the classical ML predictions with ALIGNN
'''

''' Scatter plot '''
s = 30
linewidth = 0.75
scale = [-1,4.5]
fig, ax1 = plt.subplots(nrows=1,ncols=3,figsize=(10.,3.6))
marker = ['o','^','>','<']
ml = ['ALIGNN','XGB','RF','LF']
for i in range(0,3):
    ax = ax1[i]
    ax.scatter(y_MP21.loc[new,'DFT'], y_MP21.loc[new,ml[0]],
                edgecolors=ccycle[0],linewidth=linewidth,facecolors='none',
                marker=marker[0], label=ml[0],s=s)
    
    # ax.scatter(y_MP21.loc[new,'DFT'], y_MP21.loc[new,ml[i+1]],
    #             c=ccycle[i+1],edgecolors='k',linewidth=linewidth,
    #             marker=marker[i+1], label=ml[i+1],s=s)
    
    ax.scatter(y_MP21.loc[new,'DFT'], y_MP21.loc[new,ml[i+1]],
                edgecolors=ccycle[i+1],linewidth=linewidth,facecolors='none',
                marker=marker[i+1], label=ml[i+1],s=s)
    
    ax.plot(scale,scale,'k',linewidth=linewidth)
    ax.grid()
    ax.set_xticks(np.arange(-2,4.5,1))
    ax.set_yticks(np.arange(-2,4.5,1))
    ax.set_xlim(scale)
    ax.set_ylim(scale)
    ax.set_xlabel('$E_f^{DFT}$ (eV/atom)')
    if i==0:
        ax.set_ylabel('$E_f^{ML}$ (eV/atom)')
    
    ax.yaxis.labelpad = -10
    ax.xaxis.labelpad = 2
    ax.legend(loc='upper left')
fig.savefig('figs/raise_issue_compare_ml_scatter.png')

#%%
'''
hexbin plot
'''

s = 30
linewidth = 0.35
scale = [-1,4.5]
fig, ax1 = plt.subplots(nrows=1,ncols=3,figsize=(9.5,3.5))
marker = ['o','^','>','<']
ml = ['ALIGNN','XGB','RF','LF']
for i in range(0,3):
    ax = ax1[i]
    
    ax.scatter(y_MP21.loc[new,'DFT'], y_MP21.loc[new,ml[0]],
                edgecolors=ccycle[1],
                linewidth=1,facecolors='none',
                marker=marker[0], label=ml[0],s=15)
    
    d_grid = 0.2
    px = y_MP21.loc[new,'DFT']
    py = y_MP21.loc[new,ml[i+1]]
    gridsize_x = int((max(px)-min(px))/d_grid)
    gridsize_y = int((max(py)-min(py))/d_grid)
    gridsize = (gridsize_x,gridsize_y)
    hb = ax.hexbin(
        px, py,label=ml[i+1],alpha=1,
        gridsize=gridsize,cmap ='Blues',norm=colors.LogNorm(vmin=1,vmax=50)) 
    
    ax.plot(scale,scale,'k',linewidth=linewidth)
    ax.grid()
    ax.set_xticks(np.arange(-2,4.5,1))
    ax.set_yticks(np.arange(-2,4.5,1))
    ax.set_xlim(scale)
    ax.set_ylim(scale)    
    
    ax.set_xlabel('$E_f^{DFT}$ (eV/atom)')
    if i==0:
        ax.set_ylabel('$E_f^{ML}$ (eV/atom)')
        
    ax.yaxis.labelpad = -10
    ax.xaxis.labelpad = 2
    ax.legend(loc='upper left')
    
cbar_ax = fig.add_axes([0.99, 0.23, 0.015, 0.7])
cb = fig.colorbar(hb, cax=cbar_ax)
cb.set_label('Counts',fontsize=10)   
fig.savefig('figs/raise_issue_compare_ml.png')





#%%
'''
Compute false discovery rate (fdr). Not discussed in the paper
'''

def get_fdr(y_true,y_pred, sta_crite):
    sta_ture = (y_true <= sta_crite)
    sta_pred = (y_pred <= sta_crite) 
    confu_mat = metrics.confusion_matrix(sta_ture,sta_pred,labels=[True, False])
    print(confu_mat)
    tp, tn, fp, fn = confu_mat.ravel()    
    fdr = round(fp/(fp+tp)*100,1) # in percentage
    return fdr    

get_fdr(y_MP21.loc[new,'DFT'], y_MP21.loc[new,'ALIGNN'],0)


for sta_crite in [-0.25, 0, 0.25, 0.5,0.75,1]:
    def get_fdr(y_true,y_pred):
        sta_ture = (y_true <= sta_crite)
        sta_pred = (y_pred <= sta_crite) 
        confu_mat = metrics.confusion_matrix(sta_ture,sta_pred,labels=[True, False])
        tp, tn, fp, fn = confu_mat.ravel()    
        fdr = round(fp/(fp+tp)*100,1) # in percentage
        return fdr    
    
    fdr={'test':{},'train':{}}
    for ml in ['LF', 'RF', 'XGB', 'ALIGNN']:
        fdr['test'][ml] = get_fdr(y_MP21.loc[new,'DFT'], y_MP21.loc[new,ml])
        fdr['train'][ml] = get_fdr(y_MP21.loc[old,'DFT'], y_MP21.loc[old,ml])
    print(sta_crite, fdr['test'])


#%%

'''
In the following, we want to find out the origin of the issue. First, we look at
some simple indicators to better understand the distribution of the dataset
'''

# Define the alloys of interests (AoI)
elements = "Li Be Na Mg Al K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Cs Ba".split()
# Convert to jarvis atom object
dat_MP18['atoms'] = dat_MP18['atoms'].apply(lambda x: Atoms.from_dict(ast.literal_eval(x)))
# Get elements
dat_MP18.insert(1, column='elements', 
                value=dat_MP18['atoms'].apply(lambda x: x.uniq_species))


#%%

df_elements = dat_MP21['elements']
count_elements={'old':[],'new':[],'all':[]}
maes={ml:[] for ml in ['ALIGNN','RF','LF','XGB']}
for ele in elements:
    # alloys in train set
    idx = dat_MP21.loc[old,'elements'].apply(lambda x: ele in x)
    count_elements['old'].append(idx.sum())
    # alloys in test set
    idx = dat_MP21.loc[new,'elements'].apply(lambda x: ele in x)
    count_elements['new'].append(idx.sum())
    for ml in ['ALIGNN','RF','LF','XGB']:
        maes[ml].append(y_err[idx][ml].abs().mean())    
    # the whole train set
    idx = dat_MP18['elements'].apply(lambda x: ele in x)
    count_elements['all'].append(idx.sum())
# convert to df
count_elements = pd.DataFrame(count_elements,index=elements)
# non-AoI in the train set    
count_elements['non-alloys']=count_elements['all']-count_elements['old']
# Plot
fig, ax = plt.subplots(figsize=(12,4.5))
ax.bar(elements,count_elements['new'],bottom=count_elements['old'],
       fill=False, edgecolor=ccycle[0],linewidth=2,width=0.7,
       label='AoI in test set')
ax.bar(elements,count_elements['old'],label='AoI in train set')
ax.set_xlim([-0.5,33.5])
ax.set_ylim([0,2750])
ax.set_ylabel('Number of structures')
# ax.set_xlabel('Elements considerd in AoI')
ax.legend(loc=[0.2,0.8])
color='k'
ax2=ax.twinx()
ax2.plot(elements, maes['ALIGNN'],'s-',color=color)
ax2.spines['right'].set_color(color)
ax2.tick_params(axis='y', colors=color)
ax2.set_ylabel('MAE (eV/atom)', color=color)
ax2.set_ylim([0,1.75])
fig.savefig('figs/explain_issue_elements_dist.png')


#%%
'''
Check the correlation between MAE in the test set and the size of the training set
'''

# First check correlation between MAE and the size of the MP18 AoI
a=pd.DataFrame()
r={'r_p':{},'r_k':{}, 'r_s':{}}
a['size'] = np.log10(count_elements['old'])
for ml in ['ALIGNN','RF','LF','XGB']:
    a[ml] = maes[ml]
    r['r_p'][ml] = r_regression(a['size'].to_frame(),a[ml])[0]
    r['r_k'][ml] = scipy.stats.kendalltau(a['size'].to_frame(),a[ml])[0]
    r['r_s'][ml] = scipy.stats.spearmanr(a['size'].to_frame(),a[ml])[0]
r = pd.DataFrame(r)
print('Correlation between the number of X-containing AoI vs MAE for X-containing test samples')
print(r)

# Then check correlation between MAE and the size of the WHOLE MP18
a=pd.DataFrame()
r={'r_p':{},'r_k':{}, 'r_s':{}}
a['size'] = np.log10(count_elements['all'])
idx = a[a['size']> 1.].index
for ml in ['ALIGNN','RF','LF','XGB']:
    a[ml] = maes[ml]    
    r['r_p'][ml] = r_regression(a.loc[idx,'size'].to_frame(),a.loc[idx,ml])[0]
    r['r_k'][ml] = scipy.stats.kendalltau(a.loc[idx,'size'].to_frame(),a.loc[idx,ml])[0]
    r['r_s'][ml] = scipy.stats.spearmanr(a.loc[idx,'size'].to_frame(),a.loc[idx,ml])[0]
r = pd.DataFrame(r)
print('Correlation between the number of X-containing AoI vs MAE for X-containing test samples')
print(r)


fig, ax = plt.subplots(figsize=(12,5.5))
ax.bar(elements,count_elements['new'],bottom=count_elements['all'],
       fill=False, edgecolor=ccycle[0],linewidth=2,width=0.7,
       label='AoI in test set')
ax.bar(elements,count_elements['old'],bottom=count_elements['non-alloys'],label='AoI in train set')
ax.bar(elements,count_elements['non-alloys'],label='Non-AoI in train set')
ax.set_xlim([-0.5,33.5])
ax.set_ylim([0,14000])
ax.set_ylabel('Number of structures')
# ax.set_xlabel('Elements considerd in the alloy subset')
ax.legend(loc=[0.1,0.75])
# color='k'
# ax2=ax.twinx()
# ax2.plot(elements, maes,'s-',color=color)
# ax2.spines['right'].set_color(color)
# ax2.tick_params(axis='y', colors=color)
# ax2.set_ylabel('MAE (eV/atom)', color=color)
# ax2.set_ylim([0,1.75])

from mpl_toolkits.axes_grid1.inset_locator import inset_axes
ax = inset_axes(ax, width=2.45, height=2.45)
# fig, ax = plt.subplots(figsize=(5,5))
ax.scatter(count_elements['all'], maes['ALIGNN'], edgecolors='k',s=50)
ax.set_xscale('log')
ax.set_xlim([100,2*10**4])
ax.set_ylim([0,2])
ax.set_xlabel('# structures in MP18')
ax.set_ylabel('MAE (eV/atom)')
ax.text(1300,1.8,'$r_p$' + f' = {r["r_p"]["ALIGNN"]:.2f}')
ax.text(1300,1.6,'$r_k$' + f' = {r["r_k"]["ALIGNN"]:.2f}')
ax.text(1300,1.4,'$r_s$' + f' = {r["r_s"]["ALIGNN"]:.2f}')

fig.savefig('figs/SI_explain_issue_corr_MAE_dataset_size.png')



#%%
'''
Plot number of unary to quaternary alloys
'''
nelements={}
nelements['new'] = dat_MP21.loc[new].value_counts(['nelements'])
nelements['old'] = dat_MP21.loc[old].value_counts(['nelements'])

# Plot 1: # structures vs. # elements
fig, ax1 = plt.subplots(ncols=3,figsize=(12,4))
ax=ax1[0]
ax.bar(range(1,5),nelements['new'][range(1,5)],bottom=nelements['old'][range(1,5)],
       fill=False, edgecolor=ccycle[0],linewidth=2,width=0.7,
       label='AoI in test set')
ax.bar(range(1,5),nelements['old'][range(1,5)],label='AoI in train set')
ax.set_xlabel('Number of elements in the structures')
ax.set_ylabel('Number of structures')
ax.set_ylim([0,5000])
ax.legend(loc=(0.01,0.818))


idx={}
for nelements in [2,3]:
    idx[f'{nelements},all'] = dat_MP21[dat_MP21['nelements']==nelements].index.tolist()
    idx[f'{nelements},old'] = list(set(idx[f'{nelements},all']) & set(old))
    idx[f'{nelements},new'] = list(set(idx[f'{nelements},all']) & set(new))
scale = [-1.5,4.7]
# Plot 2: Binary
ax=ax1[1]
ax.plot(scale,scale,'k',linewidth=linewidth)
ax.scatter(y_MP21.loc[idx['2,new'], 'DFT'],y_MP21.loc[idx['2,new'], 'ALIGNN'],
           marker='o', facecolors='none', edgecolors=ccycle[0],linewidth=1,
           label='Binary in test set')
ax.scatter(y_MP21.loc[idx['2,old'], 'DFT'],y_MP21.loc[idx['2,old'], 'ALIGNN'],
           marker='s', facecolors=ccycle[1],edgecolors='k',linewidth=0.5,
           label='Binary in train set')
ax.set_xticks(np.arange(-2,4.7,1))
ax.set_yticks(np.arange(-2,4.7,1))
ax.set_xlim(scale)
ax.set_ylim(scale)
ax.grid()
ax.set_xlabel('$E_f^{DFT}$ (eV/atom)')
ax.set_ylabel('$E_f^{ALIGNN}$ (eV/atom)')
ax.legend(loc=(0.01,0.818))
# Plot 3: Ternary
ax=ax1[2]
ax.plot(scale,scale,'k',linewidth=linewidth)
ax.scatter(y_MP21.loc[idx['3,new'], 'DFT'],y_MP21.loc[idx['3,new'], 'ALIGNN'],
           marker='o', facecolors='none', edgecolors=ccycle[0],linewidth=1,
           label='Ternary in test set')
ax.scatter(y_MP21.loc[idx['3,old'], 'DFT'],y_MP21.loc[idx['3,old'], 'ALIGNN'],
           marker='s', facecolors=ccycle[1],edgecolors='k',linewidth=0.5,
           label='Ternary in train set')
ax.set_xticks(np.arange(-2,4.7,1))
ax.set_yticks(np.arange(-2,4.7,1))
ax.set_xlim(scale)
ax.set_ylim(scale)
ax.grid()
ax.set_xlabel('$E_f^{DFT}$ (eV/atom)')
ax.set_ylabel('$E_f^{ALIGNN}$ (eV/atom)')
ax.legend(loc=(0.01,0.818))

fig.savefig('figs/SI_explain_issue_nelements.png')

#%%
'''
Plot the space group number distribution
'''
# Get the space group number (can take a few minutes)
dat_MP18.insert(1, column='space_group_number', 
                value=dat_MP18['atoms'].apply(
                    lambda x: x.get_spacegroup[0]
                    )
                )
sgn_all = count_sg_num(dat_MP21,'space_group_number')
sgn_old_AoI = count_sg_num(dat_MP21.loc[old],'space_group_number')
sgn_old_all = count_sg_num(dat_MP18,'space_group_number')
sgn_new = count_sg_num(dat_MP21.loc[new],'space_group_number')
sgn_new.sort_values()
sgn_old = sgn_old_AoI

# Plot the figure
fontsize = 15
fig, ax = plt.subplots(figsize=(7,4.5))
width=1.5
ax.bar(sgn_new.index,sgn_new.values,bottom=sgn_old.values,width=width-0.3,
       fill=False, edgecolor=ccycle[0],linewidth=width,
       label='AoI in test set')
ax.bar(sgn_old.index, sgn_old.values, width=width,
       label='AoI in train set')
ax.set_xticks(range(10,240,30))
ax.set_yticks(range(0,1200,150))
ax.set_xlim([0,231])
ax.set_ylim([0,1100])
ax.legend(loc=(0.5,0.815))
ax.set_xlabel('Space group number',fontsize=fontsize)
ax.set_ylabel('Number of structures',fontsize=fontsize)
ax.text(40,1000,'SG-38',ha='left',size=fontsize)
ax.text(73,750,'SG-71',ha='left',size=fontsize)
ax.text(187,550,'SG-187',ha='right',size=fontsize)
ax.yaxis.labelpad=-5
fig.savefig('figs/explain_issue_space_group.png')


#%%
'''
Parity plot colored by space groups
'''

# identify entries that belong to different SG and in the training/test sets
idx={'new':{}, 'old':{}}
idx_tmp = dat_MP21[dat_MP21['space_group_number']==71].index.tolist()
idx['new']['71'] = list(set(idx_tmp) & set(new))
idx['old']['71'] = list(set(idx_tmp) & set(old))
idx_tmp = dat_MP21[dat_MP21['space_group_number']==38].index.tolist()
idx['new']['38'] = list(set(idx_tmp) & set(new))
idx['old']['38'] = list(set(idx_tmp) & set(old))
idx['new']['other'] = list(set(new)-set(idx['new']['38'])-set(idx['new']['71']))
idx['old']['other'] = list(set(old)-set(idx['old']['38'])-set(idx['old']['71']))

idx_tmp = dat_MP21[dat_MP21['space_group_number']==187].index.tolist()
idx['new']['187'] = list(set(idx_tmp) & set(new))
idx['old']['187'] = list(set(idx_tmp) & set(old))

mae_SG71 = (y_MP21.loc[idx['new']['71'],'DFT'] 
            - y_MP21.loc[idx['new']['71'],'ALIGNN']).abs().mean()

'''
Parity plot for SG-71 groups
'''

# fig, ax = plt.subplots(figsize=(4,4))
# ax.scatter(y_MP21.loc[idx['new']['71'],'DFT'],
#            y_MP21.loc[idx['new']['71'],'ALIGNN'],
#            marker='o',facecolors='none',edgecolors=ccycle[0],linewidth=1,
#            label='SG-71 in test set')
# ax.scatter(y_MP21.loc[idx['old']['71'],'DFT'],
#            y_MP21.loc[idx['old']['71'],'ALIGNN'],
#            marker='s',facecolors=ccycle[1],edgecolors='k',linewidth=0.5,
#            label='SG-71 in train set')
# ax.set_xlim(scale)
# ax.set_xticks(np.arange(-1,4.5,1))
# ax.set_yticks(np.arange(-1,4.5,1))
# ax.set_xlabel('$E_f^{DFT}$ (eV/atom)')
# ax.set_ylim(scale)
# ax.plot(scale,scale,'k',linewidth=0.7)
# ax.set_ylabel('$E_f^{ALIGNN}$ (eV/atom)')
# ax.xaxis.labelpad=3
# ax.yaxis.labelpad=-10
# ax.text(1.7,1.1,f'Test MAE in SG-71: \n{mae_SG71:.3f} eV/atom',fontsize=11)
# ax.legend(loc='upper left')


#%%
'''
Parity plot for different space groups
'''
linewidth=0.6
fig, ax = plt.subplots(ncols=3,figsize=(11,4))
plt.subplots_adjust(wspace=0, hspace=0)

ax[2].scatter(y_MP21.loc[idx['new']['71'],'DFT'],
              y_MP21.loc[idx['new']['71'],'ALIGNN'],
              marker='o',facecolors='none',edgecolors=ccycle[0],linewidth=linewidth,
              label='SG-71 in test set')
ax[2].scatter(y_MP21.loc[idx['old']['71'],'DFT'],
              y_MP21.loc[idx['old']['71'],'ALIGNN'],
              marker='s',facecolors=ccycle[1],edgecolors='k',linewidth=linewidth,
              label='SG-71 in train set')

ax[1].scatter(y_MP21.loc[idx['new']['38'],'DFT'],
              y_MP21.loc[idx['new']['38'],'ALIGNN'],
              marker='o',facecolors='none',edgecolors=ccycle[0],linewidth=linewidth,
              label='SG-38 in test set')
ax[1].scatter(y_MP21.loc[idx['old']['38'],'DFT'],
              y_MP21.loc[idx['old']['38'],'ALIGNN'],
              marker='s',facecolors=ccycle[1],edgecolors='k',linewidth=linewidth,
              label='SG-38 in train set')

ax[0].scatter(y_MP21.loc[idx['new']['187'],'DFT'],y_MP21.loc[idx['new']['187'],'ALIGNN'],
              marker='o',facecolors='none',edgecolors=ccycle[0],linewidth=linewidth,
              label='SG-187 in test set')
ax[0].scatter(y_MP21.loc[idx['old']['187'],'DFT'],y_MP21.loc[idx['old']['187'],'ALIGNN'],
              marker='s',facecolors=ccycle[1],edgecolors='k',linewidth=0.5,
              label='SG-187 in train set')

scale=[-1,4.5]
for i in range(len(ax)):
    ax[i].set_xlim(scale)
    ax[i].set_xticks(np.arange(-1,4.5,1))
    ax[i].set_yticks(np.arange(-1,4.5,1))
    ax[i].set_xlabel('$E_f^{DFT}$ (eV/atom)')
    ax[i].set_ylim(scale)
    ax[i].plot(scale,scale,'k',linewidth=0.7)
    ax[i].yaxis.labelpad=-10
    ax[i].legend(loc='upper left')

ax[0].set_ylabel('$E_f^{ALIGNN}$ (eV/atom)')

fig.savefig('figs/explain_issue_space_group_parity_plot_scatter.png')


''' hexbin '''
gridsize = (40,10)
norm = colors.LogNorm(vmin=1,vmax=50) #Normalize colors.LogNorm(vmin=1,vmax=100)
fig, ax = plt.subplots(ncols=3,figsize=(11,4))
plt.subplots_adjust(wspace=0, hspace=0)

hb = ax[2].hexbin(y_MP21.loc[idx['new']['71'],'DFT'],
                y_MP21.loc[idx['new']['71'],'ALIGNN'],
                label='SG-71 in test set',
                gridsize=gridsize,cmap ='Blues',
                norm=norm) 

ax[2].scatter(y_MP21.loc[idx['old']['71'],'DFT'],
              y_MP21.loc[idx['old']['71'],'ALIGNN'],
              marker='s',facecolors=ccycle[1],edgecolors='k',linewidth=0.5,
              label='SG-71 in train set')

hb = ax[1].hexbin(y_MP21.loc[idx['new']['38'],'DFT'],
                y_MP21.loc[idx['new']['38'],'ALIGNN'],
                label='SG-38 in test set',
                gridsize=gridsize,cmap ='Blues',
                norm=norm) 

ax[1].scatter(y_MP21.loc[idx['old']['38'],'DFT'],
              y_MP21.loc[idx['old']['38'],'ALIGNN'],
              marker='s',facecolors=ccycle[1],edgecolors='k',linewidth=0.5,
              label='SG-38 in train set')

hb = ax[0].hexbin(y_MP21.loc[idx['new']['187'],'DFT'],
                y_MP21.loc[idx['new']['187'],'ALIGNN'],
                label='SG-187 in test set',
                gridsize=gridsize,cmap ='Blues',
                norm=norm) 

ax[0].scatter(y_MP21.loc[idx['old']['187'],'DFT'],y_MP21.loc[idx['old']['187'],'ALIGNN'],
              marker='s',facecolors=ccycle[1],edgecolors='k',linewidth=0.5,
              label='SG-187 in train set')

scale=[-1,4.5]
for i in range(len(ax)):
    ax[i].set_xlim(scale)
    ax[i].set_xticks(np.arange(-1,4.5,1))
    ax[i].set_yticks(np.arange(-1,4.5,1))
    ax[i].set_xlabel('$E_f^{DFT}$ (eV/atom)')
    ax[i].set_ylim(scale)
    ax[i].plot(scale,scale,'k',linewidth=0.7)
    ax[i].yaxis.labelpad=-10
    ax[i].legend(loc='upper left')
ax[0].set_ylabel('$E_f^{ALIGNN}$ (eV/atom)')

cbar_ax = fig.add_axes([0.99, 0.23, 0.015, 0.7])
cb = fig.colorbar(hb, cax=cbar_ax)
cb.set_label('Counts',fontsize=10)   

fig.savefig('figs/explain_issue_space_group_parity_plot.png',dpi=200)



#%%
'''
Plot the distribution of different space groups for all materials in the whole MP18

'''


sgn_old = sgn_old_all
fig, ax1 = plt.subplots(nrows=2,figsize=(8,6))
ax=ax1[0]
width=1.5
ax.bar(sgn_new.index,sgn_new.values,bottom=sgn_old.values,
        fill=False, edgecolor=ccycle[0],linewidth=1,width=1.2,
        label='AoI test set')
ax.bar(sgn_old_AoI.index, sgn_old_AoI.values, color=ccycle[1],
        bottom=(sgn_old_all-sgn_old_AoI).values,width=width,label='AoI in train set')
ax.bar(sgn_old.index, (sgn_old_all-sgn_old_AoI).values, width=width,
       color=ccycle[2],label='Non-AoI in train set')
ax.set_xlim([0,231])
ax.set_xticks(range(0,240,20))
# ax.set_yscale('log')
ax.legend(loc='upper center')
ax.set_ylabel('Number of structures')
ax.text(38,1400,'SG-38',ha='left',size=12)
ax.text(71,1200,'SG-71',ha='left',size=12)

ax.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=True,      # ticks along the bottom edge are on
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off

'''
Plot the distribution of ALIGNN prediction errors for the test AoI

'''

ax=ax1[1]
ax.scatter(dat_MP21.loc[new,'space_group_number'],
           y_err.loc[new,'ALIGNN'].abs(),
           marker='_',
           s=15,
           #edgecolors='k',
           linewidth=1,
           label='Errors of test samples'
           )
ax.set_xlim([0,231])
ax.set_ylim([0,3.6])
ax.set_xticks(range(0,240,20))
ax.set_xlabel('Space group number')
ax.set_ylabel('|$E_f^{ALIGNN}$ - $E_f^{DFT}$| (eV/atom)')
ax.legend(loc='upper right')
# fig.subplots_adjust(top=0.9,wspace=0, hspace=0)
fig.savefig('figs/SI_explain_issue_space_group.png')

 

#%%
'''
Plot the model disagreement. Not discussed in manuscript
'''

y_diff={}
for ml in ['XGB','LF','RF']:
    y_diff[ml] = (y_MP21.loc[new,ml]-y_MP21.loc[new,'ALIGNN']).abs()
y_diff = pd.DataFrame(y_diff)
y_alldiff={}
ml = ['ALIGNN', 'XGB','LF','RF']
for i in range(4):
    for j in range(i+1,4):
        y_alldiff[f'{ml[i]}-{ml[j]}'] = (y_MP21.loc[new,ml[i]]-y_MP21.loc[new,ml[j]]).abs()
y_alldiff = pd.DataFrame(y_alldiff)
y_err = y_MP21.loc[new,['ALIGNN','DFT']].max(axis=1) - y_MP21.loc[new,['ALIGNN','DFT']].min(axis=1)

'''
2d plots
'''
fig, ax1 = plt.subplots(ncols=3, figsize=(12,3.8))
ax1[0].scatter(y_diff['XGB'], y_diff['LF'], edgecolor='k',linewidth=0.6,
               c=y_err,vmin=0.,vmax=3.)
ax1[0].set_xlabel('|$E_f^{XGB}-E_f^{ALIGNN}$|')
ax1[0].set_ylabel('|$E_f^{LF}-E_f^{ALIGNN}$|')

ax1[1].scatter(y_diff['XGB'], y_diff['RF'], edgecolor='k',linewidth=0.6,
               c=y_err,vmin=0.,vmax=3.)
ax1[1].set_xlabel('|$E_f^{XGB}-E_f^{ALIGNN}$|')
ax1[1].set_ylabel('|$E_f^{RF}-E_f^{ALIGNN}$|')

ax1[2].scatter(y_diff['RF'], y_diff['LF'], edgecolor='k',linewidth=0.6,
               c=y_err,vmin=0.,vmax=3.)
ax1[2].set_xlabel('|$E_f^{RF}-E_f^{ALIGNN}$|')
ax1[2].set_ylabel('|$E_f^{LF}-E_f^{ALIGNN}$|')
fig.savefig('figs/SI_spot_issue_UMAP_error_2d.png')

'''
3d plot
'''
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
mappable = ax.scatter(y_diff['XGB'], y_diff['RF'], y_diff['LF'], 
                      edgecolor='k',linewidth=0.5,
                      c=y_err,vmin=0.,vmax=3.                     
                      )

ax.set_xlabel('|$E_f^{XGB}-E_f^{ALIGNN}$|')
ax.set_ylabel('|$E_f^{RF}-E_f^{ALIGNN}$|')
ax.set_zlabel('|$E_f^{LF}-E_f^{ALIGNN}$|')
# # Adjust the interactive plot and get the angles
# print(ax.azim,ax.elev)
# ax.view_init(21, 120)
ax.azim = 126
ax.elev = 37
cax = fig.add_axes([0.12, 0.15, 0.02, 0.7]) 
cbar_ax = fig.colorbar(mappable, cax=cax,orientation='vertical')
cbar_ax.set_label('|$E_f^{DFT}-E_f^{ALIGNN}$| (eV/atom)',
                   labelpad=-50,
                   size=10
                  )
fig.savefig('figs/SI_spot_issue_UMAP_error_3d.png')


#%%
'''
UMAP showing the model disagreements(QBC)
'''
import umap
z_umap = {}
X = y_diff
# X = y_alldiff
n_neighbors_list = [5, 15, 75]
for n_neighbors in [15]: #n_neighbors_list: 
    reducer = umap.UMAP(
        random_state=0,
        low_memory = False,
        n_neighbors=n_neighbors)     
    z_umap[n_neighbors] = pd.DataFrame(reducer.fit_transform(X),
                                       index = X.index, columns=[0,1])
    
#%%
'''
UMAP with different n_neighbors
'''

fig, ax1 = plt.subplots(ncols=3, figsize=(10,3.8))
for i, n_neighbors in enumerate(n_neighbors_list):
    z = z_umap[n_neighbors]
    ax = ax1[i]
    
    ax.tick_params(
        axis='both',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are on
        left=False,         # ticks along the left edge are off
        labelbottom=False, # labels along the bottom edge are off
        labelleft=False) # labels along the bottom edge are off
    
    mappable = ax.scatter(z[0], z[1],marker='o',edgecolors='k',linewidth=0.2,
                          c=y_err,vmin=0.,vmax=3,
                          label='AoI in test set')
    ax.set_title(f'UMAP with n_neighbors={n_neighbors}')    
# fig.subplots_adjust(right=0.8) # https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.subplots_adjust.html
cbar_ax = fig.add_axes([0.995, 0.075, 0.01, 0.8])
cbar_ax = fig.colorbar(mappable, cax=cbar_ax,shrink=0.95)
cbar_ax.set_label('|$E_f^{DFT}-E_f^{ALIGNN}$| (eV/atom)',labelpad=7,size=10)

#%%
'''
Plot n_neighbors = 15
'''
z = z_umap[15]
fig, ax = plt.subplots(figsize=(4,2.5))
ax.set_title('UMAP with n_neighbors=15',fontsize=10)
mappable = ax.scatter(z[0], z[1],marker='o',edgecolors='k',linewidth=0.2,
                      c=y_err,vmin=0.,vmax=3,label='AoI in test set')
cbar_ax = fig.colorbar(mappable, ax=ax)
cbar_ax.set_label('|$E_f^{DFT}-E_f^{ALIGNN}$| (eV/atom)',
                   labelpad=7,
                   size=10
                  )
ax.tick_params(
    axis='both',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are on
    left=False,         # ticks along the left edge are off
    labelbottom=False, # labels along the bottom edge are off
    labelleft=False) # labels along the bottom edge are off
fig.savefig('figs/spot_issue_UMAP_error.png')



