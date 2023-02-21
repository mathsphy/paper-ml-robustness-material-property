#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This code produces the following csv outputs under the "data" folder:
    z_umap_*: UMAP embeddings for the whole MP18, and MP21 alloys.
    maes_RF_AL*, 10runs.maes_RF_AL0, 10runs.maes_RF_rand: MAE on the MP21 AoI 
        during the active learning process using different acquisition policies
"""

#%% Import tools
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.colors as colors
import umap

# Model training
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from sklearn.feature_selection import r_regression
from sklearn.linear_model import Ridge
import scipy
import xgboost as xgb
from lineartree import LinearForestRegressor
from myfunc import custom_matplotlib
import os 

custom_matplotlib()
ccycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

random_state = 1

model_al={}
model_al['XGB'] = xgb.XGBRegressor(n_estimators=2000,learning_rate=0.1,
                                   reg_lambda=0, # L2 regularization
                                   reg_alpha=0.1,# L1 regularization
                                   tree_method='gpu_hist', gpu_id=0)
max_features=0.3
n_estimators=100
model_al['RF'] = RandomForestRegressor(n_estimators=n_estimators, 
                                       max_features=max_features, 
                                       n_jobs=-1, random_state=random_state)
model_al['LF'] = LinearForestRegressor(base_estimator=Ridge(),
                                       n_estimators=n_estimators, 
                                       max_features=max_features,
                                       n_jobs=-1, random_state=random_state)

#%% Import data
'''
In this part:
    1. Load the dataset
    2. Consider only the important features
    3. Impute NaN with mean in the feature columns
    4. Mark the existing alloys and the new alloys
'''


'''
1. Load dataset which are already featurized
'''

# megnet_273feat is the dataset used for the MP18-pretrained model
dat_MP18 = pd.read_csv('data/megnet_273feat.csv')
dat_MP18 = dat_MP18.set_index('id')
dat_MP18.index.names = ['material_id'] # https://stackoverflow.com/questions/19851005/rename-pandas-dataframe-index
# drop unused columns
dat_MP18 = dat_MP18.drop(columns=['Unnamed: 0','atoms','structure'])
# MP.2021.11.10 is the latest version (Aug.28) available in the new API.
# There are 7800 entries.
dat_MP21 = pd.read_csv('data/MP_alloys_2021.11.10_273feat.csv').set_index('material_id')

''' 
2. Use only the important features:
Important features obtained from another script train_MP18.py:
90 features to keep: 
['mean CN_VoronoiNN', 'mean ordering parameter shell 1', 'mean neighbor distance variation', 'avg_dev CN_VoronoiNN', 'mean local difference in NValence', 'MagpieData mean NpUnfilled', 'MagpieData mean NsUnfilled', 'minimum local difference in Number', 'MagpieData mode GSmagmom', 'minimum local difference in Column', 'MagpieData mode NfUnfilled', 'MagpieData mode GSbandgap', 'MagpieData maximum MeltingT', 'avg_dev local difference in NdValence', 'minimum local difference in NpUnfilled', 'MagpieData maximum CovalentRadius', 'MagpieData mode NValence', 'MagpieData range NdUnfilled', 'range local difference in NfValence', 'avg_dev local difference in CovalentRadius', 'minimum local difference in NdValence', 'MagpieData mean NUnfilled', 'MagpieData minimum AtomicWeight', 'MagpieData mode NdUnfilled', 'minimum local difference in NdUnfilled', 'MagpieData mean MeltingT', 'avg_dev local difference in NValence', 'minimum local difference in MeltingT', 'range local difference in NUnfilled', 'MagpieData minimum NValence', 'MagpieData minimum NsUnfilled', 'minimum local difference in NpValence', 'mean ordering parameter shell 3', 'MagpieData minimum GSvolume_pa', 'minimum local difference in GSvolume_pa', 'MagpieData maximum Column', 'frac d valence electrons', 'MagpieData mode NpUnfilled', 'avg_dev local difference in GSbandgap', 'MagpieData minimum NdValence', 'minimum local difference in CovalentRadius', 'MagpieData avg_dev Row', 'MagpieData minimum Electronegativity', '0-norm', 'MagpieData maximum SpaceGroupNumber', 'MagpieData range Electronegativity', 'compound possible', 'range local difference in Column', 'MagpieData mode NsValence', 'MagpieData mode NfValence', 'minimum local difference in NsUnfilled', 'MagpieData mode NUnfilled', 'minimum neighbor distance variation', 'MagpieData mean MendeleevNumber', 'MagpieData avg_dev GSvolume_pa', 'minimum local difference in GSmagmom', 'minimum local difference in GSbandgap', 'frac s valence electrons', 'MagpieData minimum NfValence', 'MagpieData maximum Row', 'MagpieData minimum GSmagmom', 'MagpieData range NpUnfilled', 'range local difference in Row', 'avg_dev local difference in NsValence', 'MagpieData minimum GSbandgap', 'mean local difference in SpaceGroupNumber', 'MagpieData minimum NdUnfilled', 'MagpieData minimum NUnfilled', 'minimum local difference in NfUnfilled', 'minimum local difference in NfValence', 'MagpieData minimum NpUnfilled', 'MagpieData mode NsUnfilled', 'avg_dev local difference in MendeleevNumber', 'max relative bond length', 'avg_dev local difference in AtomicWeight', '10-norm', 'avg_dev neighbor distance variation', 'minimum local difference in NUnfilled', 'MagpieData minimum NfUnfilled', 'MagpieData mode Column', 'MagpieData avg_dev MendeleevNumber', 'MagpieData mode SpaceGroupNumber', 'range local difference in NfUnfilled', 'MagpieData mode GSvolume_pa', 'min relative bond length', 'MagpieData maximum NdValence', 'maximum CN_VoronoiNN', 'avg_dev local difference in NpValence', 'MagpieData avg_dev GSmagmom', 'avg_dev local difference in NpUnfilled']
'''
important_features = ['mean CN_VoronoiNN', 'mean ordering parameter shell 1', 'mean neighbor distance variation', 'avg_dev CN_VoronoiNN', 'mean local difference in NValence', 'MagpieData mean NpUnfilled', 'MagpieData mean NsUnfilled', 'minimum local difference in Number', 'MagpieData mode GSmagmom', 'minimum local difference in Column', 'MagpieData mode NfUnfilled', 'MagpieData mode GSbandgap', 'MagpieData maximum MeltingT', 'avg_dev local difference in NdValence', 'minimum local difference in NpUnfilled', 'MagpieData maximum CovalentRadius', 'MagpieData mode NValence', 'MagpieData range NdUnfilled', 'range local difference in NfValence', 'avg_dev local difference in CovalentRadius', 'minimum local difference in NdValence', 'MagpieData mean NUnfilled', 'MagpieData minimum AtomicWeight', 'MagpieData mode NdUnfilled', 'minimum local difference in NdUnfilled', 'MagpieData mean MeltingT', 'avg_dev local difference in NValence', 'minimum local difference in MeltingT', 'range local difference in NUnfilled', 'MagpieData minimum NValence', 'MagpieData minimum NsUnfilled', 'minimum local difference in NpValence', 'mean ordering parameter shell 3', 'MagpieData minimum GSvolume_pa', 'minimum local difference in GSvolume_pa', 'MagpieData maximum Column', 'frac d valence electrons', 'MagpieData mode NpUnfilled', 'avg_dev local difference in GSbandgap', 'MagpieData minimum NdValence', 'minimum local difference in CovalentRadius', 'MagpieData avg_dev Row', 'MagpieData minimum Electronegativity', '0-norm', 'MagpieData maximum SpaceGroupNumber', 'MagpieData range Electronegativity', 'compound possible', 'range local difference in Column', 'MagpieData mode NsValence', 'MagpieData mode NfValence', 'minimum local difference in NsUnfilled', 'MagpieData mode NUnfilled', 'minimum neighbor distance variation', 'MagpieData mean MendeleevNumber', 'MagpieData avg_dev GSvolume_pa', 'minimum local difference in GSmagmom', 'minimum local difference in GSbandgap', 'frac s valence electrons', 'MagpieData minimum NfValence', 'MagpieData maximum Row', 'MagpieData minimum GSmagmom', 'MagpieData range NpUnfilled', 'range local difference in Row', 'avg_dev local difference in NsValence', 'MagpieData minimum GSbandgap', 'mean local difference in SpaceGroupNumber', 'MagpieData minimum NdUnfilled', 'MagpieData minimum NUnfilled', 'minimum local difference in NfUnfilled', 'minimum local difference in NfValence', 'MagpieData minimum NpUnfilled', 'MagpieData mode NsUnfilled', 'avg_dev local difference in MendeleevNumber', 'max relative bond length', 'avg_dev local difference in AtomicWeight', '10-norm', 'avg_dev neighbor distance variation', 'minimum local difference in NUnfilled', 'MagpieData minimum NfUnfilled', 'MagpieData mode Column', 'MagpieData avg_dev MendeleevNumber', 'MagpieData mode SpaceGroupNumber', 'range local difference in NfUnfilled', 'MagpieData mode GSvolume_pa', 'min relative bond length', 'MagpieData maximum NdValence', 'maximum CN_VoronoiNN', 'avg_dev local difference in NpValence', 'MagpieData avg_dev GSmagmom', 'avg_dev local difference in NpUnfilled']

print(f'important_features: {len(important_features)}')

X_MP18 = dat_MP18[important_features]
y_MP18 = dat_MP18['e_form']
X_MP21 = dat_MP21[important_features]
y_MP21 = dat_MP21['formation_energy_per_atom']

'''
3. Impute NaN with mean in the feature columns
'''
imp_mean = SimpleImputer(strategy='mean')
imp_mean.fit(pd.concat([X_MP18,X_MP21]))

X_MP18 = pd.DataFrame(imp_mean.transform(X_MP18), index=X_MP18.index, 
                      columns=X_MP18.columns)

X_MP21 = pd.DataFrame(imp_mean.transform(X_MP21), index=X_MP21.index, 
                      columns=X_MP21.columns)

'''
4. Mark the existing alloys and the new alloys

'''
# The material_id of some entries in MP18 have changed. Their updated id are obtained
# using update_material_id.py.
megnet_updated_id_MP21 = pd.read_csv('data/megnet_updated_id_MP21.csv').set_index('id_MP21')
old = list(set(dat_MP21.index) & set(megnet_updated_id_MP21.index))
new = list(set(dat_MP21.index) - set(old))
# Separate the existing alloys and the new alloys
X_MP21_old = X_MP21.loc[old]
y_MP21_old = y_MP21.loc[old]
X_MP21_new = X_MP21.loc[new]
y_MP21_new = y_MP21.loc[new]
print(f'Entries existing in MP18: {X_MP21_old.shape[0]}')
print(f'Entries not in MP18: {y_MP21_new.shape[0]}')

#%% Define RF model
''' 
Define the RF model. 
'''

def train_model(X,y,random_state):
    model = model_al['RF']    
    model.fit(X, y)
    return model

def transf_df(model, df):
    return pd.DataFrame(model.transform(df),index=df.index,
                        columns=df.columns)   

def get_comp(pca,X):
    components = pca.transform(X)
    components = pd.DataFrame(components, 
                              index=X.index,
                              columns = [i for i in range(components.shape[1])])
    return components

def plot_pca(components_new, components_old, cmap, cbar_title):
    def my_subfigs(ax, pca_x, pca_y, components_new, components_old):
        ax.set_title(f'PC{pca_x} vs. PC{pca_y}')
        ax.set_xlabel(f'Principle component {pca_x}')
        ax.set_ylabel(f'Principle component {pca_y}')
        # plot the test data
        cax = ax.scatter(components_new[pca_x], components_new[pca_y], marker='o', 
                         edgecolors='k',
                         c=cmap.loc[components_new.index], 
                          # cmap='tab20b',
                         label="Test set", s=25)
        # plot the training data
        ax.scatter(components_old[pca_x], components_old[pca_y], marker='s', 
                    # facecolors='none',edgecolors='r', 
                    alpha=1, c='r',
                    label="Train set",
                    s=15)
        ax.legend()
        plt.tight_layout()
        return cax        
    fig, ax = plt.subplots(nrows=2, ncols=3,figsize=(18,12))    
    my_subfigs(ax[0,0], 0, 1, components_new, components_old)
    my_subfigs(ax[0,1], 0, 2, components_new, components_old)
    my_subfigs(ax[0,2], 0, 3, components_new, components_old)
    my_subfigs(ax[1,0], 1, 2, components_new, components_old)
    my_subfigs(ax[1,1], 1, 3, components_new, components_old)
    mappable = my_subfigs(ax[1,2], 2, 3, components_new, components_old)    
    fig.subplots_adjust(left=0.15) # https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.subplots_adjust.html
    cbar_ax = fig.add_axes([0.05, 0.1, 0.02, 0.8])
    cbar_ax = fig.colorbar(mappable, cax=cbar_ax,shrink=0.9)
    cbar_ax.set_label(cbar_title,labelpad=-70,size=18)
    return fig

def pred_ints(model_RF, X, percentile=95):
    """
    Defind the function to get the confidence interval, by interrogating the 
    individual trees in the RF, see:
        https://blog.datadive.net/prediction-intervals-for-random-forests/

    """
    y_preds = []
    for tree in model_RF.estimators_:
        y_pred = tree.predict(X)
        y_preds.append(y_pred)
    # Convert to np.array by stacking list of arrays along the column axis
    # with each column being the prediction from a different tree
    y_preds = np.stack(y_preds, axis=1)           
    #
    q_down = (100 - percentile) / 2.
    q_up = 100 - q_down
    y_lower = pd.Series(np.percentile(y_preds, q_down, axis=1),index=X.index)
    y_upper = pd.Series(np.percentile(y_preds, q_up, axis=1)  ,index=X.index)  
    y_mean = pd.Series(model_RF.predict(X) ,index=X.index)  
    y_uctt = pd.Series(y_upper - y_lower,index=X.index)  
    return y_mean, y_uctt, y_lower, y_upper


#%%
'''
Get ALIGNN prediction errors
'''
y_MP21_ALIGNN = pd.read_csv(
    'data/mp_e_form_alignnn_MP21_alloys.csv',index_col='material_id'
    )['Ef_alignn']

y_err = np.abs(y_MP21.loc[new]-y_MP21_ALIGNN.loc[new])


#%%
'''
Plot the range of features:
'''

selected_features = [i for i in X_MP21.columns.tolist() 
                      if X_MP21[i].min() < X_MP21[i].max()]
q = (1/X_MP21.shape[0])*2
print(f'Quantile = {q}')

all_min = X_MP21[selected_features].quantile(q=q)
old_min = X_MP21_old[selected_features].quantile(q=q)
all_max = X_MP21[selected_features].quantile(q=1-q)
old_max = X_MP21_old[selected_features].quantile(q=1-q)
X_MP21_old_mm_min = (old_min-all_min)/(all_max-all_min)
X_MP21_old_mm_max = (old_max-all_min)/(all_max-all_min)

'''
Don't plot the features whose ranges change little
'''
q=0.05
not_too_low = X_MP21_old_mm_min[X_MP21_old_mm_min>q]
not_too_high = X_MP21_old_mm_min[X_MP21_old_mm_max<1-q]
selected_features = list(set(not_too_low.index) | set(not_too_high.index))
X_MP21_old_mm_min = X_MP21_old_mm_min[selected_features]
X_MP21_old_mm_max = X_MP21_old_mm_max[selected_features]
# rearrange orders
selected_features = (X_MP21_old_mm_max-X_MP21_old_mm_min).sort_values().index.tolist()
fig, ax = plt.subplots(figsize=(6,4))
ax.plot(X_MP21_old_mm_min.loc[selected_features],selected_features,'o-',label='min$_{train}$')
ax.plot(X_MP21_old_mm_max.loc[selected_features],selected_features,'o-',label='max$_{train}$')
ax.set_xlabel('Feature value range (MinMax scaled)')
ax.legend(fontsize=11)
ax.grid(linewidth=1)
yticks = ax.get_yticks()
ax.set_ylim([yticks[0],yticks[-1]])
ax.set_xlim([0,1])
# plt.xticks(xticks,rotation = 65,ha='right',fontsize=7)
fig.savefig('figs/explain_issue_feature_range.png')

#%%
y_err = np.abs(y_MP21.loc[new]-y_MP21_ALIGNN.loc[new])

fig, ax = plt.subplots(figsize=(4.5,4))
mappable = ax.scatter(X_MP21.loc[new,'mean neighbor distance variation'],
                      X_MP21.loc[new,'mean CN_VoronoiNN'],
                      c=y_err,vmin=0,vmax=4,
                      label='AoI in test set')
ax.scatter(X_MP21.loc[old,'mean neighbor distance variation'],
           X_MP21.loc[old,'mean CN_VoronoiNN'],marker='+', 
           c='orangered',label='AoI in train set')
ax.set_xlabel('mean neighbor distance variation')
ax.set_ylabel('mean CN_VoronoiNN')
cbar_ax = fig.colorbar(mappable, ax=ax,shrink=0.7,anchor=(-1.5,0.7))
cbar_ax.set_label('|$E_f^{ALIGNN}-E_f^{DFT}$| (eV/atom)',
                   labelpad=-45,
                   size=8.5)
legend = ax.legend(loc='upper center')
# legend.legendHandles[1]._alpha=1
fig.savefig('figs/explain_issue_feature_vs_error.png')




#%%
''' Dimensionality reduction below '''

scaler = StandardScaler().fit(X_MP21[important_features])
X_MP21_std = transf_df(scaler, X_MP21[important_features])
y_err = np.abs(y_MP21.loc[new]-y_MP21_ALIGNN.loc[new])

#%%
def plot_dim_red(z,figname=None):
    fontsize=10

    fig, ax = plt.subplots(figsize=(4,3.3))
    ax.tick_params(
        axis='both',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are on
        left=False,         # ticks along the left edge are off
        labelbottom=False, # labels along the bottom edge are off
        labelleft=False) # labels along the bottom edge are off    
    mappable = ax.scatter(z.loc[new,0], z.loc[new,1],c=y_err,vmin=0,vmax=3.5,marker='o',
                          edgecolors='k',linewidth=0.1,
                          label='AoI in test set')
    ax.scatter(z.loc[old,0], z.loc[old,1], marker='+', s=20,
               alpha=0.45, c='orangered',label='AoI in train set')
    legend = ax.legend(fontsize=fontsize)
    legend.legendHandles[1]._alpha=1
            
    cbar_ax = fig.add_axes([0.99, 0.075, 0.03, 0.8])
    cbar_ax = fig.colorbar(mappable, cax=cbar_ax,shrink=0.95)
    cbar_ax.set_label('|$E_f^{ALIGNN}-E_f^{DFT}$| (eV/atom)',
                       labelpad=7,
                       size=fontsize
                      )
    fig.tight_layout()
    if figname is not None:
        fig.savefig(figname,bbox_inches='tight')

#%%
''' 
PCA and TSNE
''' 

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

pca = PCA(n_components=2)
tsne = TSNE(n_components=2,perplexity=75)

X_pca = pd.DataFrame(pca.fit_transform(X_MP21_std),X_MP21_std.index)
X_tsne = pd.DataFrame(tsne.fit_transform(X_MP21_std),X_MP21_std.index)

plot_dim_red(X_pca,'figs/explain_issue_PCA.png')
plot_dim_red(X_tsne,'figs/explain_issue_TSNE.png')


#%%

''' 
UMAP plot
'''

n_neighbors_list = [5,15,75]
y_err = np.abs(y_MP21.loc[new]-y_MP21_ALIGNN.loc[new])
z_umap = {}
X = X_MP21_std[selected_features]
for n_neighbors in n_neighbors_list: 
    csv = f'data/z_umap_{n_neighbors}.csv'
    if os.path.exists(csv):
        z_umap[n_neighbors] = pd.read_csv(csv,index_col=0)
    else:
        reducer = umap.UMAP(low_memory = False,n_neighbors=n_neighbors)     
        z_umap[n_neighbors] = pd.DataFrame(
            reducer.fit_transform(X),index = X.index, columns=[0,1]
            )
    z_umap[n_neighbors].columns=[0,1]


''' UMAP colored by ALIGNN prediction errors '''

fontsize=12

def plotUMAP(figname,cmap,show_train=True):
    fig, ax1 = plt.subplots(ncols=3, figsize=(10,4))
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
        mappable = ax.scatter(z.loc[new,0], z.loc[new,1],c=cmap,vmin=0,vmax=3.5,marker='o',
                              edgecolors='k',linewidth=0.1,
                              label='AoI in test set')
        ax.set_title(f'UMAP with n_neighbors={n_neighbors}',fontsize=fontsize)
        if show_train:
            ax.scatter(z.loc[old,0], z.loc[old,1], marker='+',
                        alpha=0.5, c='orangered',label='AoI in train set')
    legend = ax.legend(fontsize=fontsize)
    if show_train:
        legend.legendHandles[1]._alpha=1
    
    # https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.subplots_adjust.html
    # fig.subplots_adjust(right=0.5) 
    cbar_ax = fig.add_axes([0.99, 0.075, 0.01, 0.8])
    cbar_ax = fig.colorbar(mappable, cax=cbar_ax,shrink=0.95)
    cbar_ax.set_label('|$E_f^{ALIGNN}-E_f^{DFT}$| (eV/atom)',
                       labelpad=7,
                       size=10
                      )
    fig.savefig(figname,bbox_inches='tight')

plotUMAP('figs/SI_explain_issue_UMAP_show_train.png',cmap=y_err)
plotUMAP('figs/SI_explain_issue_UMAP_hide_train.png',cmap=y_err,show_train=(False))

plot_dim_red(z_umap[75],'figs/explain_issue_UMAP.png')


#%%
'''
UMAP colored by space group numbers
'''

fig, ax1 = plt.subplots(ncols=3, figsize=(10,3.8))
idx={}
for sgn in [38,71,187]:
    idx[sgn] = dat_MP21[dat_MP21['space_group_number']==sgn].index.tolist()
idx['other'] = list(set(dat_MP21.index.tolist())-set(idx[38])-set(idx[71])-set(idx[187]))

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
    for sgn in [38,71,187,'other']:
        ax.scatter(z.loc[idx[sgn],0], z.loc[idx[sgn],1],marker='o',s=20,
                   edgecolors='k',linewidth=0.1,label=f'SG-{sgn}')
    ax.set_title(f'UMAP with n_neighbors={n_neighbors}',fontsize=fontsize)
ax.legend(fontsize=fontsize)
fig.savefig('figs/SI_explain_issue_UMAP_space_group.png')

#%%
'''
RF prediction intervals
'''
X = X_MP18
y = y_MP18
model = train_model(X,y,random_state)
y_RF={}


#%%
'''
UMAP colored by RF uncertainty
'''

y_RF['mean'], y_RF['uctt'], y_RF['lo'], y_RF['hi'] = pred_ints(model, X_MP21)
# __, y_RF['uctt_train'], __, __ = pred_ints(model, X_MP21.loc[old])

fig, ax1 = plt.subplots(ncols=3, figsize=(10,4))
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
    mappable = ax.scatter(z.loc[new,0], z.loc[new,1],c=y_RF['uctt'].loc[new],
                            # vmin=0.5,vmax=2.5,
                          marker='o',
                          edgecolors='k',linewidth=0.1,
                          label='AoI in test set')
    ax.set_title(f'UMAP with n_neighbors={n_neighbors}')
legend = ax.legend()
cbar_ax = fig.add_axes([0.99, 0.075, 0.01, 0.8])
cbar_ax = fig.colorbar(mappable, cax=cbar_ax,shrink=0.95)
cbar_ax.set_label('RF uncertainty (eV/atom)',
                   labelpad=7,
                   size=10
                  )
fig.savefig('figs/SI_spot_issue_UMAP_RF_uctt.png',bbox_inches='tight')

#%%
'''
Plot the RF uncertainty vs. the RF prediction errors
'''
linewidth=0.7
y_err = np.abs(y_MP21 - y_RF['mean'])
fig, ax = plt.subplots(figsize=(3.7,3.7))

ax.scatter(y_RF['uctt'].loc[new],y_err.loc[new], marker='o',s=20,
            facecolors='none',
            edgecolors=ccycle[0],linewidth=linewidth,label='AoI in test set'
            )
ax.scatter(y_RF['uctt'].loc[old],y_err.loc[old], marker='s',s=20,
            # facecolors=ccycle[1],
            facecolors='none',
            edgecolors='k',
            linewidth=linewidth,label='AoI in train set')

ax.set_xticks(np.arange(0,4.5,1))
ax.set_yticks(np.arange(0,4.5,1))
ax.set_xlim([0,3.5])
ax.set_ylim([0,4.5])
ax.set_xlabel('RF uncertainty (eV/atom)')
ax.set_ylabel('|$E_f^{DFT}-E_f^{RF}$| (eV/atom)')
ax.legend(loc=(0.0,0.87),fontsize=8.1)

r_p = r_regression(y_RF['uctt'].loc[new].to_frame(),y_err.loc[new])[0]
r_s = scipy.stats.spearmanr(y_RF['uctt'].loc[new].to_frame(),y_err.loc[new])[0]
r_k = scipy.stats.kendalltau(y_RF['uctt'].loc[new].to_frame(),y_err.loc[new])[0]
fontsize=11
ax.text(0.01,1.75,f'$r_p$={r_p:.2f}',fontsize=fontsize)
ax.text(0.01,1.50,f'$r_s$={r_s:.2f}',fontsize=fontsize)
ax.text(0.01,1.25,f'$r_k$={r_k:.2f}',fontsize=fontsize)
fig.savefig('figs/spot_issue_RF_uctt_scatter.png',bbox_inches='tight')

#%%
''' hexbin '''
linewidth=0.7
y_err = np.abs(y_MP21 - y_RF['mean'])
fig, ax = plt.subplots(figsize=(5,3.7))


d_grid = 0.2
gridsize = (30,20)

hb = ax.hexbin(y_RF['uctt'].loc[new],y_err.loc[new],label='AoI in test set',
          gridsize=gridsize,cmap ='Blues',norm=colors.LogNorm(vmin=1,vmax=50))
cb = fig.colorbar(hb, ax=ax)
cb.set_label('Counts',fontsize=10)


ax.scatter(y_RF['uctt'].loc[old],y_err.loc[old], marker='s',s=20,
            facecolors='none',
            edgecolors='k',
            linewidth=linewidth,label='AoI in train set')

ax.set_xticks(np.arange(0,4.5,1))
ax.set_yticks(np.arange(0,4.5,1))
ax.set_xlim([0,3.5])
ax.set_ylim([0,4.5])
ax.set_xlabel('RF uncertainty (eV/atom)')
ax.set_ylabel('|$E_f^{DFT}-E_f^{RF}$| (eV/atom)')
ax.legend(loc=(0.01,0.85),fontsize=8.1)

r_p = r_regression(y_RF['uctt'].loc[new].to_frame(),y_err.loc[new])[0]
r_s = scipy.stats.spearmanr(y_RF['uctt'].loc[new].to_frame(),y_err.loc[new])[0]
r_k = scipy.stats.kendalltau(y_RF['uctt'].loc[new].to_frame(),y_err.loc[new])[0]
fontsize=11
ax.text(0.01,1.75,f'$r_p$={r_p:.2f}',fontsize=fontsize)
ax.text(0.01,1.50,f'$r_s$={r_s:.2f}',fontsize=fontsize)
ax.text(0.01,1.25,f'$r_k$={r_k:.2f}',fontsize=fontsize)
fig.savefig('figs/spot_issue_RF_uctt.png',bbox_inches='tight')


#%%
'''
If we consider the data whose uncertainty is larger than a given threshold as 
problematic, evaluate the average true error of those data

'''

y_mean_err=[]
y_mean_err_by_chunk=[]

y_std_err=[]
d = 0.2
y_uctt_threshold=np.arange(0,3.2,d)
pcts=[]
for y_tmp in y_uctt_threshold:
    y_uctt = y_RF['uctt'].loc[new]
    # get the material ids whose RF uncertainty is > y_tmp
    index = y_uctt[y_uctt>y_tmp].index
    y_err1 = y_err.loc[new].loc[index]
    pct = y_err1.shape[0]/y_uctt.shape[0]
    y_mean_err.append(y_err1.mean())
    # get the material ids whose RF uncertainty is > y_tmp and < y_tmp+d
    index = y_uctt[(y_uctt>y_tmp) & (y_uctt<y_tmp+d)].index
    y_err1 = y_err.loc[new].loc[index]
    y_mean_err_by_chunk.append(y_err1.mean())
    pcts.append(pct)
fig, ax = plt.subplots(figsize=(4.5,4))
ax.plot(y_uctt_threshold,y_mean_err,marker='o',label='Samples with $\Delta E > E_0$')
ax.plot(y_uctt_threshold,y_mean_err_by_chunk,marker='>',label=f'Samples with $\Delta E$ in [$E_0$,$E_0$+{d}]')
ax.set_xlabel('Selection threshold $E_0$ (eV/atom)')
ax.set_ylabel('MAE of selected samples (eV/atom)')
ax.set_xlim([0,3])
ax.set_ylim([0,2.5])
ax.grid()
ax2 = ax.twinx()
ax2.set_ylim([0,1])
ax2.plot(y_uctt_threshold,pcts,marker='^',c='g',label='Fraction of samples with $\Delta E > E_0$')
ax2.set_ylabel('Fraction of selected samples')
fig.legend(fontsize=8.5,loc=(0.25,0.85))
fig.tight_layout()

#%%
'''
Start active learning
'''

def train_al(model,selected_ids):
    '''
    Add selected materials to the training set
    
    Parameters
    ----------
    selected_ids : list
        List of material ids to be added .
    random_state : int
        DESCRIPTION.

    Returns
    -------
    maes_all : float
        MAE for all the MP21 AoI
    maes_not_in : float
        MAE for the MP21 AoI excluding the ones added to the training set
    '''
    X = pd.concat([X_MP18,X_MP21_old, X_MP21_new.loc[selected_ids]])
    y = pd.concat([y_MP18,y_MP21_old, y_MP21_new.loc[selected_ids]])
    model = model.fit(X,y)
    y_pred = pd.DataFrame(model.predict(X_MP21_new),index=X_MP21_new.index)
    maes_all = mean_absolute_error(y_pred,y_MP21_new)
    maes_not_in = mean_absolute_error(y_pred.drop(index=selected_ids),
                                      y_MP21_new.drop(index=selected_ids))
    return maes_all, maes_not_in

#%%
# by UMAP
'''
RF AL runs
'''
modelname='RF'
num_run = 10
z_threshold=11.1111

maes_all={}
maes_not_in={}
maes_all_rnd={}
maes_not_in_rnd={}
nsamples_list = [0, 2, 10, 20, 30, 40, 50, 75, 100, 125, 150, 175, 200, 250, 
                 300, 350, 400, 450, 500, 600, 700]
df_maes_RF_AL0={}
df_maes_RF_rand={}
for random_state in range(num_run):
    for nsamples in nsamples_list:
        # Randomly selection in the 
        selected_ids = z[z[0]>z_threshold].sample(n=nsamples,random_state=random_state).index
        maes_all[nsamples], __ = train_al(model_al[modelname], selected_ids)
        selected_ids = z.loc[y_MP21_new.index].sample(n=nsamples,random_state=random_state).index
        maes_all_rnd[nsamples], __ = train_al(model_al[modelname], selected_ids)
    df_maes_RF_AL0[random_state] = pd.Series(maes_all)
    df_maes_RF_rand[random_state]= pd.Series(maes_all_rnd)
    
maes_RF_AL0 = pd.DataFrame(df_maes_RF_AL0)
maes_RF_rand = pd.DataFrame(df_maes_RF_rand)
maes_RF_AL0['mean'] = maes_RF_AL0.mean(axis=1)
maes_RF_AL0['std'] = maes_RF_AL0.std(axis=1)
maes_RF_rand['mean'] = maes_RF_rand.mean(axis=1)
maes_RF_rand['std'] = maes_RF_rand.std(axis=1)
maes_RF_AL0.to_csv(f'data/{num_run}runs.maes_RF_AL0.csv')
maes_RF_rand.to_csv(f'data/{num_run}runs.maes_RF_rand.csv')

#%%
'''
Use the uncertainty as defined by the diagreement between different models

'''

def train_UQ_ML(idx_pool_ini,n_per_step,nstep):
    '''
    AL based on RF UQ

    Parameters
    ----------
    idx_pool_ini : list of the candidate pool.

    Returns
    -------
    maes_all : Series of maes for all the candidates
    maes_not_in : Series of maes for all the candidates except the ones added for training

    '''
    
    model={}
    maes_all={}
    maes_not_in={}
    selected_ids = []         
    # train initial model
    for modelname in ['XGB','RF','LF']:
        model[modelname] = model_al[modelname].fit(X_MP18,y_MP18)
        maes_all[modelname]={}
        maes_not_in[modelname]={}
        maes_all[modelname][0],maes_not_in[modelname][0] = train_al(
            model[modelname], selected_ids
            )
    for i in range(nstep):
        # Remove selected samples from the pool
        idx_pool = list(set(idx_pool_ini)-set(selected_ids))
        y_diff = pd.DataFrame()
        y_pred = pd.DataFrame() 
        for modelname in ['XGB','RF','LF']:
            y_pred[modelname] = pd.DataFrame(
                model_al[modelname].predict(X_MP21.loc[idx_pool]),
                index=idx_pool
                )
        y_diff['XGB-LF'] = (y_pred['XGB']-y_pred['LF']).abs()
        y_diff['XGB-RF'] = (y_pred['XGB']-y_pred['RF']).abs()
        y_diff['LF-RF'] = (y_pred['LF']-y_pred['RF']).abs()
        y_diff['dis'] = y_diff['XGB-LF']+y_diff['XGB-RF']+y_diff['LF-RF']
        selected_ids.extend(y_diff['dis'].sort_values(ascending=False).index[:n_per_step])
        nsamples = (i+1)*n_per_step
        for modelname in ['XGB','RF','LF']:
            maes_all[modelname][nsamples], maes_not_in[modelname][nsamples] = train_al(model[modelname], selected_ids)
    maes_all = pd.DataFrame(maes_all)
    maes_not_in = pd.DataFrame(maes_not_in)
    return maes_all, maes_not_in

n_per_step,nstep = 5, 140

idx_pool_ini = z[z[0]>z_threshold].index.tolist()  
df_maes_RF_AL3, __ = train_UQ_ML(idx_pool_ini,n_per_step,nstep)
idx_pool_ini = new
df_maes_RF_AL4, __ = train_UQ_ML(idx_pool_ini,n_per_step,nstep)
df_maes_RF_AL3.to_csv('data/maes_RF_AL3.csv')
df_maes_RF_AL4.to_csv('data/maes_RF_AL4.csv')

#%%
'''
As the ~optimal baseline model: assume that labels are known, pick the test 
samples with largest errors to train
'''

def train_err(idx_pool_ini,n_per_step,nstep):
    '''
    AL based on RF UQ

    Parameters
    ----------
    idx_pool_ini : list of the candidate pool.

    Returns
    -------
    maes_all : Series of maes for all the candidates
    maes_not_in : Series of maes for all the candidates except the ones added for training

    '''
    modelname='RF'
    maes_all={}
    maes_not_in={}
    selected_ids = []         
    # train initial model
    model_al[modelname].fit(X_MP18,y_MP18)
    maes_all={}
    maes_not_in={}
    maes_all[0],maes_not_in[0] = train_al(
        model_al[modelname], selected_ids
        )
    for i in range(nstep):
        # Remove selected samples from the pool
        idx_pool = list(set(idx_pool_ini)-set(selected_ids))
        y_pred = model_al[modelname].predict(X_MP21.loc[idx_pool])
        y_diff = (y_pred - y_MP21.loc[idx_pool]).abs()
        selected_ids.extend(y_diff.sort_values(ascending=False).index[:n_per_step])
        nsamples = (i+1)*n_per_step
        maes_all[nsamples], maes_not_in[nsamples] = train_al(model_al[modelname], selected_ids)
    maes_all = pd.Series(maes_all)
    maes_not_in = pd.Series(maes_not_in)
    return maes_all, maes_not_in

n_per_step,nstep = 5,140
idx_pool_ini = new
df_maes_RF_AL5, __ = train_err(idx_pool_ini,n_per_step,nstep)
df_maes_RF_AL5.to_csv('data/maes_RF_AL5.csv')


#%%
num_run=10
# Random baseline
maes_RF_rand = pd.read_csv(f'data/{num_run}runs.maes_RF_rand.csv',index_col=0)
# Random in the UMAP cluster
maes_RF_AL0 = pd.read_csv(f'data/{num_run}runs.maes_RF_AL0.csv',index_col=0)
# Query by committee 
df_maes_RF_AL4 = pd.read_csv('data/maes_RF_AL4.csv',index_col=0)
# For reference
df_maes_RF_AL5 = pd.read_csv('data/maes_RF_AL5.csv',index_col=0)

fig, ax = plt.subplots(figsize=(4.,3.5))
# ax.set_xscale('log')
ax.set_yticks(np.arange(0.05,0.45,0.05))
ax.set_xticks(np.arange(0,600,100))
ax.set_xlim([0,500])
ax.set_ylim([0.05,0.4])
ax.errorbar(maes_RF_rand.index,maes_RF_rand['mean'],maes_RF_rand['std'],marker='.',label='Random')
ax.errorbar(maes_RF_AL0.index,maes_RF_AL0['mean'],maes_RF_AL0['std'],marker='.',label='UMAP+random')
ax.errorbar(df_maes_RF_AL4['RF'].index, df_maes_RF_AL4['RF'],marker='.', label='QBC') # To add
ax.errorbar(df_maes_RF_AL5.index, df_maes_RF_AL5,marker='.',label='Optimal') # To add
ax.set_xlabel('Number of added data')
ax.set_ylabel('MAE in AoI test set (eV/atom)')
ax.minorticks_on()
ax.legend(fontsize=8,loc=(0.08,0.71))
ax.grid(which='both',linewidth=0.3)
# Upper x axis
ax_up = ax.twiny()
ax_up.set_xticks(ax.get_xticks())
tot_new = X_MP21_new.shape[0]
ax_up.set_xticklabels( [ round(i,1) for i in ax.get_xticks()/tot_new*100] )
ax_up.set_xlabel('Ratio of added data to test data (%)')
# Inset
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
ax = inset_axes(ax, width=1., height=1.1)
ax.set_xticks(np.arange(0,60,10))
ax.set_xlim([0,50])
ax.set_yticks(np.arange(0.1,0.4,0.05))
ax.set_ylim([0.1,0.4])
plt.xticks(fontsize=9)
plt.yticks(fontsize=9)
ax.errorbar(maes_RF_rand.index,maes_RF_rand['mean'],maes_RF_rand['std'],marker='.',label='Random')
ax.errorbar(maes_RF_AL0.index,maes_RF_AL0['mean'],maes_RF_AL0['std'],marker='.',label='UMAP+random')
ax.errorbar(df_maes_RF_AL4['RF'].index, df_maes_RF_AL4['RF'],marker='.', label='QBC') # To add
ax.errorbar(df_maes_RF_AL5.index, df_maes_RF_AL5,marker='.',label='Prediction error based') # To add
plt.tight_layout()
fig.savefig('figs/solve_issue_RF_AL0.png')


