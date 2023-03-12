#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from flask import Flask, request, render_template
import pandas as pd
import numpy as np
import scipy
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer
from sklearn.decomposition import PCA
import itertools
import pickle
import keras
from keras.models import load_model

app = Flask(__name__, template_folder='templates')


def my_data_transformer(X, scaler_minmax_to_use=np.nan, scaler_normalize_to_use=np.nan, scaler_minmax_for_cross_to_use=np.nan, scaler_standard_for_pca_to_use=np.nan, pca_to_use=np.nan, cols_to_drop=[]):
    X_copy = X.copy()
    
    # 1. Временно убираем категориальный признак 
    X_cover_angle = pd.DataFrame(X_copy.pop('cover_angle'))
    
    # 2. Приводим к единому масштабу данных - [0, 1] - через MinMaxScaler
    scaler_minmax = scaler_minmax_to_use if scaler_minmax_to_use is not np.nan else MinMaxScaler().fit(X_copy)
    X_scaled = pd.DataFrame(scaler_minmax.transform(X_copy), index=X_copy.index, columns=X_copy.columns.tolist())
    
    # 3. Создание новых признаков: Нормализуем по строкам (по наблюдениям = горизонтально), а не по признакам (вертикально), т.к. по признакам корреляция с целевой переменной незначительная
    scaler_normalize = scaler_normalize_to_use if scaler_normalize_to_use is not np.nan else Normalizer().fit(X_copy.drop(columns=cols_to_drop))
    X_norm = pd.DataFrame(scaler_normalize.transform(X_copy.drop(columns=cols_to_drop)), index=X_copy.index, columns=[f'{col}_norm' for col in X_copy.drop(columns=cols_to_drop).columns.tolist()])    
    
    # 4. Создание новых признаков: попарное перемножение 
    col_names = X_copy.columns.tolist()
    col_pair_combinations = list(itertools.combinations(col_names, 2))
    X_cross = []
    for col_pair in col_pair_combinations:
        X_cross.append(X_copy.loc[:, col_pair].apply(lambda x: x[0] * x[1], axis=1))
    X_cross = pd.DataFrame(X_cross).T
    X_cross.columns = [f'{col_name[0]}_cross_{col_name[1]}' for col_name in col_pair_combinations]
    # Нормализуем получившиеся значения - приводим к масштабу данных в диапазоне [0-1]
    scaler_minmax_for_cross = scaler_minmax_for_cross_to_use if scaler_minmax_for_cross_to_use is not np.nan else MinMaxScaler().fit(X_cross)
    X_cross_scaled = pd.DataFrame(scaler_minmax_for_cross.transform(X_cross), index=X_copy.index, columns=X_cross.columns.tolist())          
        
    # 5. Создание новых признаков: главные компоненты (PCA)
    scaler_standard_for_pca = scaler_standard_for_pca_to_use if scaler_standard_for_pca_to_use is not np.nan else StandardScaler().fit(X_copy)
    X_stand = scaler_standard_for_pca.transform(X_copy)
    pca = pca_to_use if pca_to_use is not np.nan else PCA(random_state=42).fit(X_stand)
    X_pca = pd.DataFrame(pca.transform(X_stand), index=X_copy.index, columns=[f'PC_{i + 1}' for i in np.arange(X_stand.shape[1])])
    
    # Собираем все признаки в один датасет, включая ранее временно исключенный категориальный
    X_to_use = pd.concat([X_scaled, X_norm, X_cross_scaled, X_pca, X_cover_angle], axis=1).reset_index(drop=True)
    
    return X_to_use

# Model 1 & Model 2
scaler_minmax_to_use = pickle.load(open(f'{app.root_path}/model_to_use/scaler_minmax_for_elasticity_model.pkl', 'rb'))
scaler_normalize_to_use = pickle.load(open(f'{app.root_path}/model_to_use/scaler_normalize_for_elasticity_model.pkl', 'rb'))
scaler_minmax_for_cross_to_use = pickle.load(open(f'{app.root_path}/model_to_use/scaler_minmax_for_cross_for_elasticity_model.pkl', 'rb'))
scaler_standard_for_pca_to_use = pickle.load(open(f'{app.root_path}/model_to_use/scaler_standard_for_pca_for_elasticity_model.pkl', 'rb'))
pca_to_use = pickle.load(open(f'{app.root_path}/model_to_use/pca_for_elasticity_model.pkl', 'rb'))

features_names = ['matrix_filler_ratio', 'density', 'hardener', 'epoxid', 'temperature', 'resin', 'cover_angle', 'cover_step', 'elasticity_boxcox', 'surface_density_boxcox', 'cover_strength_boxcox']

elasticity_on_stratching_aver_values = pd.read_csv(f'{app.root_path}/model_to_use/elasticity_on_stretching_aver_values.csv', index_col=[0])['elasticity_on_stretching'].to_dict()
solidity_on_stratching_aver_values = pd.read_csv(f'{app.root_path}/model_to_use/solidity_on_stretching_aver_values.csv', index_col=[0])['solidity_on_stretching'].to_dict()

# Model 3
scaler_minmax_to_use_matrix = pickle.load(open(f'{app.root_path}/model_to_use/scaler_minmax_for_matrix_model.pkl', 'rb'))
scaler_normalize_to_use_matrix = pickle.load(open(f'{app.root_path}/model_to_use/scaler_normalize_for_matrix_model.pkl', 'rb'))
scaler_minmax_for_cross_to_use_matrix = pickle.load(open(f'{app.root_path}/model_to_use/scaler_minmax_for_cross_for_matrix_model.pkl', 'rb'))
scaler_standard_for_pca_to_use_matrix = pickle.load(open(f'{app.root_path}/model_to_use/scaler_standard_for_pca_for_matrix_model.pkl', 'rb'))
pca_to_use_matrix = pickle.load(open(f'{app.root_path}/model_to_use/pca_for_matrix_model.pkl', 'rb'))

features_names_2 = ['density', 'hardener', 'epoxid', 'temperature', 'elasticity_on_stretching', 'solidity_on_stretching', 'resin', 'cover_angle', 'cover_step', 'elasticity_boxcox', 'surface_density_boxcox', 'cover_strength_boxcox']


@app.route('/', methods=['post', 'get'])
def compute():
    
    result = {'elasticity_on_stretching': 0, 
              'solidity_on_stretching': 0, 
              'matrix_filler_ratio': 0}
   
    if request.method == 'POST':
        density = float(request.form.get('density').replace(',', '.'))
        elasticity = float(request.form.get('elasticity').replace(',', '.'))
        hardener = float(request.form.get('hardener').replace(',', '.'))
        epoxid = float(request.form.get('epoxid').replace(',', '.'))
        temperature = float(request.form.get('temperature').replace(',', '.'))
        surface_density = float(request.form.get('surface_density').replace(',', '.'))
        resin = float(request.form.get('resin').replace(',', '.'))
        cover_angle = float(request.form.get('cover_angle').replace(',', '.'))
        cover_step = float(request.form.get('cover_step').replace(',', '.'))
        cover_strength = float(request.form.get('cover_strength').replace(',', '.'))
        matrix_filler_ratio = float(request.form.get('matrix_filler_ratio').replace(',', '.'))
        
        # box-cox transformation
        elasticity_boxcox = scipy.stats.boxcox((elasticity + 1), 0.8514864751506241)
        surface_density_boxcox = scipy.stats.boxcox((surface_density + 1), 0.6348829215273967)
        cover_strength_boxcox = scipy.stats.boxcox((cover_strength + 1), 1.2154612257440889)
        
        
        # Model 1 and Model 2

        features = pd.Series([matrix_filler_ratio, density, hardener, epoxid, temperature, resin, cover_angle, cover_step, elasticity_boxcox, surface_density_boxcox, cover_strength_boxcox], index=features_names, dtype='float64').to_frame().T
        X_to_use = my_data_transformer(features, scaler_minmax_to_use, scaler_normalize_to_use, scaler_minmax_for_cross_to_use, scaler_standard_for_pca_to_use, pca_to_use)
               
        elasticity_on_stretch_model = pickle.load(open(f'{app.root_path}/model_to_use/elasticity_stretch_model.pkl', 'rb'))
        solidity_on_stretch_model = pickle.load(open(f'{app.root_path}/model_to_use/solidity_stretch_model.pkl', 'rb'))
        
        elasticity_on_stretching_class_label = elasticity_on_stretch_model.predict(X_to_use)[0]
        solidity_on_stretching_class_label = solidity_on_stretch_model.predict(X_to_use)[0]
        
        result['elasticity_on_stretching'] = elasticity_on_stratching_aver_values[elasticity_on_stretching_class_label]
        result['solidity_on_stretching'] = solidity_on_stratching_aver_values[solidity_on_stretching_class_label]
        
        
        # Model 3
        
        features_2 = pd.Series([density, hardener, epoxid, temperature, result['elasticity_on_stretching'], result['solidity_on_stretching'], resin, cover_angle, cover_step, elasticity_boxcox, surface_density_boxcox, cover_strength_boxcox], index=features_names_2, dtype='float64').to_frame().T
        X_to_use_2 = my_data_transformer(features_2, scaler_minmax_to_use_matrix, scaler_normalize_to_use_matrix, scaler_minmax_for_cross_to_use_matrix, scaler_standard_for_pca_to_use_matrix, pca_to_use_matrix, ['elasticity_on_stretching', 'solidity_on_stretching'])
        matrix_filler_model = load_model(f'{app.root_path}/model_to_use/matrix_filler_best_model.h5')
        result['matrix_filler_ratio'] = matrix_filler_model.predict(X_to_use_2)[0][0]
    
    return render_template('index.html', result=result)


if __name__ == '__main__':
    app.run()