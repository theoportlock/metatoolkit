#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import functions as f
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, mean_absolute_error, r2_score
import os
import shap
from imblearn.over_sampling import SMOTE
import sys
from itertools import permutations

def predict(df, analysis, shap_val=False, shap_interact=False, n_iter=30):
    outputs = []
    aucrocs = []
    maes = []
    r2s = []
    meanabsshaps = pd.DataFrame()
    shap_interacts = pd.DataFrame(index=pd.MultiIndex.from_tuples(permutations(df.columns, 2)))
#    
    random_state = 1
    for random_state in range(n_iter):
        if analysis.lower() == 'classifier':
            model = RandomForestClassifier(n_jobs=-1, random_state=random_state)
        elif analysis.lower() == 'regressor':
            model = RandomForestRegressor(n_jobs=-1, random_state=random_state)
        else:
            raise ValueError("Invalid analysis type. Choose 'classifier' or 'regressor'.")
# 
        X, y = df, df.index
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random_state, stratify=y)
        smoter = SMOTE(random_state=random_state)
        X_train_upsample, y_train_upsample = smoter.fit_resample(X_train, y_train)
#
        model.fit(X_train_upsample, y_train_upsample)
#        
        if analysis.lower() == 'classifier':
            y_prob = model.predict_proba(X_test)[:, 1]
            aucrocs.append(roc_auc_score(y_test, y_prob))
#        
        if analysis.lower() == 'regressor':
            y_pred = model.predict(X_test)
            maes.append(mean_absolute_error(y_test, y_pred))
            r2s.append(r2_score(y_test, y_pred))
#        
        if shap_val:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer(X)
            meanabsshaps[random_state] = pd.Series(
                    np.abs(shap_values.values[:,:,1]).mean(axis=0),
                    index=X.columns
            )
#        
        if shap_interact:
            explainer = shap.TreeExplainer(model)
            explainer = shap.Explainer(model)
            inter_shaps_values = explainer.shap_interaction_values(X)
            sum_shap_interacts = pd.DataFrame(
                    data=np.abs(inter_shaps_values[:,:,:,1]).sum(0),
                    columns=df.columns,
                    index=df.columns)
            shap_interacts[random_state] = sum_shap_interacts.stack()
#
    f.save(pd.Series(aucrocs).to_frame('aucroc'), f'{subject}aucrocs')
    f.save(meanabsshaps, f'{subject}meanabsshaps')
    f.save(shap_interacts,f'{subject}shap_interacts')

def parse_args(args):
    parser = argparse.ArgumentParser(
       prog='predict.py',
       description='Random Forest Classifier/Regressor with options'
    )
    parser.add_argument('analysis', type=str, help='Regressor or Classifier')
    parser.add_argument('subject', type=str, help='Data name or full filepath')
    parser.add_argument('-n','--n_iter', type=int, help='Number of iterations for bootstrapping')
    parser.add_argument('--shap_val', action='store_true', help='SHAP interpreted output')
    parser.add_argument('--shap_interact', action='store_true', help='SHAP interaction interpreted output')
    return parser.parse_args(args)

#arguments = ['classifier','speciesCondition.MAM','--shap_val','--shap_interact', '-n=20']
arguments = sys.argv[1:]
args = parse_args(arguments)

# Check if the provided subject is a valid file path
if os.path.isfile(args.subject):
    subject = Path(args.subject).stem
else:
    subject = args.subject

df = f.load(subject)
analysis = args.analysis
shap_val = args.shap_val
shap_interact = args.shap_interact
n_iter = args.n_iter

predict(df, args.analysis, shap_val=args.shap_val, shap_interact=args.shap_interact, n_iter=args.n_iter)
