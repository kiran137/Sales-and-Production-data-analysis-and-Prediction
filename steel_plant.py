# -*- coding: utf-8 -*-
"""
Created on Sun May 27 11:45:21 2018

@author: kiran pulaparthi
"""
import numpy as np
import pandas as pd
import matplotlib.pylab as plt,mpld3
import seaborn as sns
from matplotlib.pylab import rcParams
import calendar

# importing data from local
df = pd.read_excel("sales-data-2016-17-test-data.xlsx")

# getting column names into lower case and cleaning
df.columns = df.columns.str.lower().str.replace(' ', '_')
df.columns = df.columns.str.lower().str.replace('.', '')

df['month']=df['date_of_sale'].dt.month
df['day']=df['date_of_sale'].dt.day

df['day_of_week']=df['date_of_sale'].dt.weekday
df['week_of_year']=df['date_of_sale'].dt.weekofyear


df[['month','daily_sale_steel_value_rs']].groupby(['month']).mean().plot.bar(title="2016-2017 Graph")
plt.savefig('steelVsMonth')
df[['day','daily_sale_steel_value_rs']].groupby(['day']).mean().plot.bar(title="2016-2017 Graph")
plt.savefig('steelvsday')
df[['day_of_week','daily_sale_steel_value_rs']].groupby(['day_of_week']).mean().plot.bar(title="2016-2017 Graph")
plt.savefig('steelvsweek')
df[['month','daily_sale_pig_iron_value']].groupby(['month']).mean().plot.bar(title="2016-2017 Graph")
plt.savefig('pigironVsMonth')
df[['day','daily_sale_pig_iron_value']].groupby(['day']).mean().plot.bar(title="2016-2017 Graph")
plt.savefig('pigironvsday')
df[['day_of_week','daily_sale_pig_iron_value']].groupby(['day_of_week']).mean().plot.bar(title="2016-2017 Graph")
plt.savefig('pigvsweek')
df[['month','daily_sale_byproduct_value_rs']].groupby(['month']).mean().plot.bar(title="2016-2017 Graph")
plt.savefig('byproductVsMonth')
df[['day','daily_sale_byproduct_value_rs']].groupby(['day']).mean().plot.bar(title="2016-2017 Graph")
plt.savefig('byvsday')
df[['day_of_week','daily_sale_byproduct_value_rs']].groupby(['day_of_week']).mean().plot.bar(title="2016-2017 Graph")
plt.savefig('byvsweek')
import datetime as dt
df['dates'] = pd.to_datetime(df['date_of_sale'])
df['dates']=df['dates'].map(dt.datetime.toordinal)
df=df.drop(['date_of_sale'],axis=1)
# fetching data only for steel
df_steel= pd.DataFrame(df.iloc[:,[0,1,25]])
df_pigiron=pd.DataFrame(df.iloc[:,[25,6,7]])
df_byproduct=pd.DataFrame(df.iloc[:,[25,12,13]])
df_pigiron=df_pigiron[df_pigiron['daily_sale_pig_iron_value']!=0]
df_byproduct=df_byproduct.drop_duplicates(subset=['dates'])
df_byproduct=df_byproduct[df_byproduct['daily_sale_byproduct_value_rs']!=0]

df_steel.info()

# find important features
# Feature Importance with Extra Trees Classifier
from sklearn.ensemble import ExtraTreesRegressor
array = df_steel.values
X = array[:,df_steel.columns!='daily_sale_steel_value_rs']
Y = array[:,1]
# feature extraction
model = ExtraTreesRegressor()
model.fit(X, Y)
print(model.feature_importances_)

#df_steel.describe()
from sklearn.ensemble import ExtraTreesRegressor
array = df_byproduct.values
S= array[:,df_byproduct.columns!='daily_sale_byproduct_value_rs']
T = array[:,2]
# feature extraction
model2 = ExtraTreesRegressor()
model2.fit(S, T)
print(model2.feature_importances_)

## dividing independent and dependent variables
X= df_steel.loc[:,df_steel.columns!='daily_sale_steel_value_rs'].values
y=df_steel.iloc[:,1].values

X_pig=df_pigiron.iloc[:,[0,1]].values
y_pig=df_pigiron.iloc[:,2].values

X_bypro=df_byproduct.iloc[:,[0,1]].values
y_bypro=df_byproduct.iloc[:,2].values
# splitting data into train and test
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size = 0.2, random_state = 0)

XP_train,XP_test,yp_train,yp_test=train_test_split(X_pig,y_pig,test_size=0.2,random_state=0)

XB_train,XB_test,yb_train,yb_test=train_test_split(X_bypro,y_bypro,test_size=0.2,random_state=0)


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
XB_train = sc.fit_transform(XB_train)
XB_test = sc.transform(XB_test)


from sklearn.linear_model import LinearRegression
mlr = LinearRegression()
mlr.fit(XP_train, yp_train)
yp_pred = mlr.predict(XP_test)
# bagging technique, considering this ensemble technique
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn import model_selection
cart=LinearRegression()
num_trees=100
seed=6
kfold= model_selection.KFold(n_splits=10,random_state=seed)
model=BaggingRegressor(base_estimator=cart,n_estimators=num_trees,random_state=seed)
resultssteel= model_selection.cross_val_score(model,X_train,y_train,cv=kfold)
resultssteel.mean()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)




#xg for bypro
from xgboost import XGBRegressor
xgb1=XGBRegressor(n_estimators=200, learning_rate=0.08, gamma=0, subsample=0.75,
                           colsample_bytree=1, max_depth=7,objective= "reg:linear")
xgb1.fit(XB_train,yb_train)
yb_pred=xgb1.predict(XB_test)



# Predicting a new result



from sklearn.metrics import r2_score,mean_squared_error
r2_score(y_test,y_pred)
from math import sqrt

sqrt(mean_squared_error(yp_test,yp_pred))
r2_score(yp_test,yp_pred)
r2_score(yb_test,yb_pred)
from sklearn.metrics import explained_variance_score
explained_variance_score(yb_pred,yb_test)
from sklearn.pipeline import make_pipeline
from sklearn.externals import joblib
joblib.dump(model,"regression_model.pkl")
joblib.dump(mlr,"regression_model1.pkl")
joblib.dump(xgb1,"regression_model2.pkl")

