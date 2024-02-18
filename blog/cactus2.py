import pandas as pd
import numpy as np
import rasterio
from rasterio import *
from rasterio.plot import show
from pyspatialml import Raster
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.pipeline import Pipeline
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (10,6.5)
import geopandas as gpd

gdf = pd.read_csv("predicted_yield_data_extacted_2.csv")
yield1 = "Zim_final_image_100.tif"
rfReg = RandomForestRegressor(min_samples_leaf=40, oob_score=True)

print(gdf.tail())
df1 = pd.DataFrame(gdf)
df1.head()
predictors =  df1
bins = np.linspace(min(predictors['yield']),max(predictors['yield']),100)
gdf_2 = pd.read_csv("predicted_crop_type_data_extacted_2.csv")
gdf_2['yield'].unique()

df2 = gdf_2
df2.head()
predictors2 =  df2
bins = np.linspace(min(predictors2['yield']),max(predictors2['yield']),100)
plt.hist((predictors2['yield']),bins,alpha=0.8);
X = predictors2.iloc[:,[0,1,2,3,4]].values

Y = predictors2.iloc[:,5:6].values
feat = predictors2.iloc[:,[0,1,2,3,4]].columns.values
feat
Y
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.5, random_state=24)
y_train = np.ravel(Y_train)
y_test = np.ravel(Y_test)
y_test
cf = RandomForestClassifier(random_state = 42)
cf.get_params()
rfclas = RandomForestClassifier(min_samples_leaf=40, oob_score=True)
rfclas.fit(X_train, Y_train);
dic_pred = {}
dic_pred['train'] = rfReg.predict(X_train)
dic_pred['train']
dic_pred['test'] = rfReg.predict(X_test)
dic_pred['test']
pearsonr_all = [pearsonr(dic_pred['train'],Y_train)[1],pearsonr(dic_pred['test'],Y_test)[1]]
pearsonr_all
rfclas.oob_score_
pipeline = Pipeline([('rf',RandomForestClassifier())])

parameters = {
        'rf__max_features':(3,4,5),
        'rf__max_samples':(0.5,0.6,0.7),
        'rf__n_estimators':(500,1000),
        'rf__max_depth':(50,100,200,300)}

grid_search = GridSearchCV(pipeline,parameters,n_jobs=6,cv=5,scoring='r2',verbose=1)
grid_search.fit(X_train,y_train)

rfReg = RandomForestClassifier(n_estimators=500,max_features=0.33,max_depth=50,max_samples=0.5,n_jobs=-1,random_state=24 , oob_score = True)
rfReg
dic_pred = {}
dic_pred['train'] = rfReg.predict(X_train)
dic_pred['test'] = rfReg.predict(X_test)
pearsonr_all_tune = [pearsonr(dic_pred['train'],y_train)[0],pearsonr(dic_pred['test'],y_test)[0]]
pearsonr_all_tune
grid_search.best_score_
print ('Best Training score: %0.3f' % grid_search.best_score_)
print ('Optimal parameters:') 
best_par = grid_search.best_estimator_.get_params()
for par_name in sorted(parameters.keys()):
    print ('\t%s: %r' % (par_name, best_par[par_name]))
rfclas = RandomForestClassifier(n_estimators=500,max_features=0.33,max_depth=50,max_samples=0.5,n_jobs=-1,random_state=24 , oob_score = True)
rfclas.fit(X_train, y_train);
rfclas
impt = [rfclas.feature_importances_, np.std([tree.feature_importances_ for tree in rfclas.estimators_],axis=1)]
ind = np.argsort(impt[0])
ind
plt.rcParams["figure.figsize"] = (6,12)
plt.barh(range(len(feat)),impt[0][ind],color="b", xerr=impt[1][ind], align="center")
plt.yticks(range(len(feat)),feat[ind])

predictors_rasters = [yield1]
stack = Raster(predictors_rasters)
result2 = stack.predict(estimator=rfclas, dtype='int16', nodata=-1)

plt.rcParams["figure.figsize"] = (12,12)
result2.iloc[0].cmap = "plasma"
result2.plot()
plt.show()

type(result2)
result2.write("Predicted_Cactus.tif",nodatavals=-1)

