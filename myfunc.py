#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 09:51:05 2022

@author: kangming
"""
import numpy as np
import pandas as pd

# For structure object conversion
import ast
from jarvis.core.atoms import Atoms
from pymatgen.core.structure import Structure



def strAtoms2objAtoms(str_atoms):
    """
    Convert back to a jarvis.core.atoms object from a string that was previously 
    converted from a jarvis.core.atoms object

    Parameters
    ----------
    str_atoms : str
        The string to which a jarvis.core.atoms object is converted.

    Returns
    -------
    obj_Atoms : jarvis.core.atoms

    """
    # convert the string of dict into dict: 
    # https://stackoverflow.com/questions/988228/convert-a-string-representation-of-a-dictionary-to-a-dictionary
    dict_atom = ast.literal_eval(str_atoms) # need module ast
    # need module jarvis
    obj_Atoms = Atoms(lattice_mat=dict_atom['lattice_mat'], 
                      coords=dict_atom['coords'], 
                      elements=dict_atom['elements'])
    
    # seems that we can simply do (https://jarvis-tools.readthedocs.io/en/master/databases.html)
    # obj_Atoms = Atoms.from_dict(str_atoms)
    
    return obj_Atoms






def strAtoms2objStructure(str_atom):
    """
    Convert strAtoms to objStructure

    Parameters
    ----------
    str_atoms : str
        The string to which a jarvis.core.atoms object is converted.

    Returns
    -------
    obj_structure : pymatgen.core.structure.Structure

    """
    
    strAtoms2objAtoms(str_atom).write_poscar() # Write POSCAR
    obj_structure = Structure.from_file('POSCAR')
    [site.to_unit_cell(in_place=True) for site in obj_structure.sites]
    return obj_structure

def dictAtoms2objStructure(dict_atom):
    """
    Convert strAtoms to objStructure

    Parameters
    ----------
    str_atoms : dict
        The dict to which a jarvis.core.atoms object is converted.

    Returns
    -------
    obj_structure : pymatgen.core.structure.Structure

    """
    obj_Atom = Atoms.from_dict(dict_atom)    
    obj_Atom.write_poscar() # Write POSCAR
    obj_structure = Structure.from_file('POSCAR')
    [site.to_unit_cell(in_place=True) for site in obj_structure.sites]
    return obj_structure


def count_sg_num(df, col='spacegroup.number'):
    '''
    Count space group occurrence in the column 'spacegroup.number' of a dataframe 

    Parameters
    ----------
    df : TYPE
        DESCRIPTION.
    col : TYPE, optional
        DESCRIPTION. The default is 'spacegroup.number'.

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    df_sg = df[col].astype(int).value_counts().sort_index() 
    tot_sg = 230 # total number of space group
    for i in range(1,tot_sg+1):
        if i not in df_sg.index:
            df_sg[i]=0
    return df_sg.sort_index()

def dict2struct(str_struct):
    dict_struct = ast.literal_eval(str_struct) # convert sting to dict
    structure = Structure.from_dict(dict_struct)
    [site.to_unit_cell(in_place=True) for site in structure.sites]
    return structure # convert dict to Structure object

def to_unitcell(structure):
    [site.to_unit_cell(in_place=True) for site in structure.sites]
    return structure

def StructureFeaturizer(
        df_in,
        col_id='structure',
        ignore_errors=True,
        chunksize=30
        ):
    """
    Featurize a dataframe using Matminter Structure featurizer

    Parameters
    ----------
    df : Pandas.DataFrame 
        DataFrame with a column named "structure"

    Returns
    -------
    A DataFrame containing 273 features (columns)

    """
    # For featurization
    from matminer.featurizers.base import MultipleFeaturizer
    from matminer.featurizers.composition import (ElementProperty, 
                                                  Stoichiometry, 
                                                  ValenceOrbital, 
                                                  IonProperty)
    from matminer.featurizers.structure import (SiteStatsFingerprint, 
                                                StructuralHeterogeneity,
                                                ChemicalOrdering, 
                                                StructureComposition, 
                                                MaximumPackingEfficiency)
    
    
    if isinstance(df_in, pd.Series):
        df = df_in.to_frame()
    else:
        df = df_in
    df[col_id] = df[col_id].apply(to_unitcell)
    
    # 128 structural feature
    struc_feat = [
        SiteStatsFingerprint.from_preset("CoordinationNumber_ward-prb-2017"), 
        SiteStatsFingerprint.from_preset("LocalPropertyDifference_ward-prb-2017"),
        StructuralHeterogeneity(),
        MaximumPackingEfficiency(),
        ChemicalOrdering()
        ]       
    # 145 compositional features
    compo_feat = [
        StructureComposition(Stoichiometry()),
        StructureComposition(ElementProperty.from_preset("magpie")),
        StructureComposition(ValenceOrbital(props=['frac'])),
        StructureComposition(IonProperty(fast=True))
        ]
    featurizer = MultipleFeaturizer(struc_feat+compo_feat)    
    # Set the chunksize used for Pool.map parallelisation
    featurizer.set_chunksize(chunksize=chunksize)
    featurizer.fit(df[col_id])
    X = featurizer.featurize_dataframe(df=df,col_id=col_id,ignore_errors=ignore_errors)  
    # check failed entries    
    print('Featurization completed.')
    failed = np.any(pd.isnull(X.iloc[:,df.shape[1]:]), axis=1)
    if np.sum(failed) > 0:
        print(f'Number failed: {np.sum(failed)}/{len(failed)}')
    return X



def custom_matplotlib():
    import matplotlib as mpl
    
    ''' Figure '''    
    mpl.rcParams['figure.figsize'] = [4.5,4.5]
    mpl.rcParams['figure.dpi'] = 150    
    mpl.rcParams['figure.autolayout'] = True
    mpl.rcParams['savefig.dpi'] = 300
    mpl.rcParams['savefig.transparent'] = False
    mpl.rcParams['legend.framealpha'] = 1
    '''
    Font: 
        https://www.statology.org/change-font-size-matplotlib/#:~:text=Note%3A%20The%20default%20font%20size%20for%20all%20elements%20is%2010.
        https://stackoverflow.com/questions/3899980/how-to-change-the-font-size-on-a-matplotlib-plot

    '''
    font = {
        #'family' : 'normal',
            # 'weight' : 'bold',
            'size'   : 14}
    mpl.rc('font', **font)
    








