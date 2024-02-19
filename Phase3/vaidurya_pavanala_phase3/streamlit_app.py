import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
#import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from dmba import regressionSummary, adjusted_r2_score
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score
from yellowbrick.regressor import ResidualsPlot
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.inspection import plot_partial_dependence
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
import io
import requests

import math
st.set_option('deprecation.showPyplotGlobalUse', False)


st.title('Analysis of property tax in West Roxbury')

DATE_COLUMN = 'date/time'
DATA_URL = "https://raw.githubusercontent.com/reisanar/datasets/master/WestRoxbury.csv"

@st.cache_data
def load_data(nrows):
    ds = pd.read_csv(io.StringIO(s.decode('utf-8')), nrows=nrows)
    ds = ds.rename(columns={'TOTAL VALUE ': 'TOTAL_VALUE', 'LOT SQFT ': 'LOT_SQFT', 'YR BUILT': 'YR_BUILT', 'GROSS AREA ': 'GROSS_AREA', 'LIVING AREA': 'LIVING_AREA', 'FLOORS ': 'FLOORS', 'BEDROOMS ': 'BEDROOMS', 'FULL BATH': 'FULL_BATH', 'HALF BATH': 'HALF_BATH'})
    return ds


tab0, tab1, tab2, tab3, tab4, tab5 = st.tabs(["Raw data","Preprocessing", "EDA", "ML Models","Report Findings","Run your own"])
model_accuracies = {}
if 'multiAcc' not in st.session_state:
    st.session_state.multiAcc = 0
if 'deciAcc' not in st.session_state:
    st.session_state.deciAcc = 0
if 'randAcc' not in st.session_state:
    st.session_state.randAcc = 0
if 'svrAcc' not in st.session_state:
    st.session_state.svrAcc = 0
if 'knnAcc' not in st.session_state:
    st.session_state.knnAcc = 0
if 'gradBAcc' not in st.session_state:
    st.session_state.gradBAcc = 0
if 'lassoAcc' not in st.session_state:
    st.session_state.lassoAcc = 0

if 'multirsme' not in st.session_state:
    st.session_state.multirsme = 0
if 'decirsme' not in st.session_state:
    st.session_state.decirsme = 0
if 'randrsme' not in st.session_state:
    st.session_state.randrsme = 0
if 'svrrsme' not in st.session_state:
    st.session_state.svrrsme = 0
if 'knnrsme' not in st.session_state:
    st.session_state.knnrsme = 0
if 'gradBrsme' not in st.session_state:
    st.session_state.gradBrsme = 0
if 'lassorsme' not in st.session_state:
    st.session_state.lassorsme = 0

model_rmse = {}

with tab0:
   st.subheader("Step 1: Raw Data Collection")
   data_load_state = st.text('Loading data...')
   data = load_data(5802)
   data_load_state.text("Data Loaded! Ready for next steps!")

   if st.checkbox('Show raw data'):
    st.subheader('Raw data')
    st.write(data)


with tab1:
   st.subheader("Step 2: Data Processing and Step 3: Data Cleaning")
   inconsis, dups, vari, drop, encodecat,stand, norm, bin = st.tabs(["Check for inconsistency","Remove duplicates", "Feature Variance","Drop irrelevant","Encode","zForm","Normalization","Binning"])

   
   with inconsis:
      if st.button("Check for inconsistency"):
         st.write("YR_BUILT has a value with 0; removing it")
         data['YR_BUILT'].value_counts()
         data = data[data.YR_BUILT != 0]
         st.write(data['YR_BUILT'].value_counts())

   with dups:
      if st.button("Identify and remove any duplicated rows in the dataset"):
         
         dup = data[data.duplicated()]
         removeDups = st.text('Processing data...')
         st.write("Duplicated Rows found:", dup)
         data.drop_duplicates(inplace=True)
         removeDups.text("Done! (using drop_duplicates)")
         #st.write(data)   
   
   with vari:
      if st.button("Feature Variance"):
         pd.set_option('display.max_columns', None)
         data.describe()
         data.round(2)
         st.write(data) 
   
   with drop:
      if st.button("Drop irrelevant columns"):
         st.write("Dropping 'GROSS_AREA' and 'TAX'")
         data = data.drop('GROSS_AREA', axis=1)
         data = data.drop('TAX', axis=1)
         st.write(data)

   with encodecat:
      if st.button("Encode categorical data"):
         dfdum = data
         dfdum = pd.get_dummies(dfdum,columns=['REMODEL'])
         dfdum = pd.get_dummies(dfdum, drop_first = True)
         st.write(dfdum)

   with stand:
      if st.button("Perform z-score standardization"):
         
         std_scaler = StandardScaler()
         #std_scaler
         # fit and transform the data
         dfdum = data
         dfdum = pd.get_dummies(dfdum,columns=['REMODEL'])
         dfdum = pd.get_dummies(dfdum, drop_first = True)
         dfT = dfdum
         df_std = pd.DataFrame(std_scaler.fit_transform(dfT), columns=dfT.columns)
         st.write(df_std)
   
   with norm:
      if st.button("Min-Max scaling Normalization"):
         scaler = MinMaxScaler()
         attributes = ['TOTAL_VALUE','LOT_SQFT']
         dfdum = data
         dfdum = pd.get_dummies(dfdum,columns=['REMODEL'])
         dfdum = pd.get_dummies(dfdum, drop_first = True)
         dfdum[attributes] = scaler.fit_transform(dfdum[attributes])
         st.write(dfdum.sample(10))

   with bin:
      if st.button("Get qcut binning"):
         st.write(data['YR_BUILT'].describe())


with tab2:
   st.subheader("Step 4: Exploratory Data Analysis")
   
   with st.container():
    col1, col2 = st.columns(2)
    with col1:
        st.write('Scatter plot between living area and total value')
    with col2:
        #st.line_chart((0,1), height=100)
        fig, ax = plt.subplots()
        ax.scatter(data['LIVING_AREA'], data['TOTAL_VALUE'])
        ax.set_xlabel('LIVING_AREA')
        ax.set_ylabel('TOTAL_VALUE')
        ax.set_title('Effect of Living Area on the Total value')
        # display the plot in the Streamlit app
        st.pyplot()

   with st.container():
    col1, col2 = st.columns(2)
    with col1:
        st.write('Correlation Heatmaps')
    with col2:
        Co = data.corr()
        # create a heatmap using Seaborn
        fig, ax = plt.subplots()
        sns.heatmap(Co, annot=True, cmap='PuBuGn')
        # display the heatmap in the Streamlit app
        st.pyplot(fig)

   with st.container():
    col1, col2 = st.columns(2)
    with col1:
        st.write('Distribution histogram')
    with col2:
        hist = data.hist(bins=15, figsize=(12, 8), grid=False, rwidth=0.8, align='right', histtype='barstacked', alpha=0.6)
        plt.title('Distribution of all the columns')
        plt.tight_layout()
        st.pyplot()

   with st.container():
      col1, col2 = st.columns(2)
      with col1:
         st.write("Barplot for Remodel")
      with col2:
         sns.barplot(x="REMODEL", y="TOTAL_VALUE", data=data)
         st.pyplot()
   
   with st.container():
      col1, col2 = st.columns(2)
      with col1:
         st.write("Voilin plot to study the distribution of the data")
      with col2:
         plt.figure(figsize=(12,8))
         sns.violinplot(x='FLOORS', y='TOTAL_VALUE', data=data)
         plt.xlabel('FLOORS', fontsize=12)
         plt.ylabel('TOTAL_VALUE', fontsize=12)
         st.pyplot()
   
   with st.container():
      col1, col2 = st.columns(2)
      with col1:
         st.write("Plot for Rooms")
      with col2:
         fig, ax=plt.subplots(figsize=(12,8))
         data['ROOMS'].value_counts().sort_values(ascending=False).head(12).plot.bar(width=0.6,edgecolor='black',align='center',linewidth=2.2)
         plt.xlabel('Number of rooms',fontsize=16)
         plt.ylabel('Count',fontsize=16)
         ax.tick_params(labelsize=20)
         plt.title('Count of number rooms',fontsize=18)
         plt.grid()
         st.pyplot()

   with st.container():
      col1, col2 = st.columns(2)
      with col1:
         st.write("Scatter plot for Rooms and Total value")
      with col2:
         plt.scatter(data['ROOMS'],data['TOTAL_VALUE']) 
         plt.xlabel("ROOMS")                                    
         plt.ylabel("TOTAL_VALUE")   
         plt.title("Effect of ROOMS on the Total value")
         st.pyplot()

   with st.container():
      col1,col2 = st.columns(2)
      with col1:
         st.write("Scatter plot for bedrooms and Total value")
      with col2:
         plt.scatter(data['BEDROOMS'],data['TOTAL_VALUE']) 
         plt.xlabel("BEDROOMS")                                    
         plt.ylabel("TOTAL_VALUE")   
         plt.title("Effect of BEDROOMS on the Total value")
         st.pyplot()

   with st.container():
      col1,col2 = st.columns(2)
      with col1:
         st.write("Plot for FULL BATH")
      with col2:
         plt.scatter(data['FULL_BATH'],data['TOTAL_VALUE'])  
         plt.xlabel("FULL_BATH")                                    
         plt.ylabel("TOTAL_VALUE")   
         plt.title("Effect of FULL_BATH on the Total value")
         st.pyplot()
      
   with st.container():
      col1,col2 = st.columns(2)
      with col1:
         st.write("Plot for fireplace")
      with col2:
         plt.scatter(data['FIREPLACE'],data['TOTAL_VALUE'])  
         plt.xlabel("FIREPLACE")                                    
         plt.ylabel("TOTAL_VALUE")   
         plt.title("Effect of FIREPLACE on the Total value")
         st.pyplot()


   with st.container():
      col1, col2 = st.columns(2)
      with col1:
         st.write("Plot for Half Bath")
      with col2:
         plt.scatter(data['HALF_BATH'],data['TOTAL_VALUE'])  
         plt.xlabel("HALF_BATH")                                    
         plt.ylabel("TOTAL_VALUE")   
         plt.title("Effect of HALF_BATH on the Total value")
         st.pyplot()

   with st.container():
      col1, col2 = st.columns(2)
      with col1:
         st.write("Plot for Kitchen")
      with col2:
         plt.scatter(data['KITCHEN'],data['TOTAL_VALUE'])  
         plt.xlabel("KITCHEN")                                    
         plt.ylabel("TOTAL_VALUE")   
         plt.title("Effect of KITCHEN on the Total value")
         st.pyplot()

   with st.container():
      col1, col2 = st.columns(2)
      with col1:
         st.write("Box plot for Lot size")
      with col2: 
         sns.boxplot(x=data['LOT_SQFT'])
         st.pyplot()

   with st.container():
      col1, col2 = st.columns(2)
      with col1:
         st.write("Plot for Living area")
      with col2:
         sns.kdeplot(data['LIVING_AREA'])
         st.pyplot()

   with st.container():
      ds = pd.get_dummies(data,columns=['REMODEL'])
      ds = pd.get_dummies(ds, drop_first = True)
      col1, col2 = st.columns(2)
      with col1:
         st.write("Violin ")
      with col2:
         plt.figure(figsize=(12,8))
         sns.violinplot(x='REMODEL_Recent',y='TOTAL_VALUE', data=ds)
         plt.xlabel('REMODEL_Recent', fontsize=12)
         plt.ylabel('TOTAL_VALUE', fontsize=12)
         st.pyplot()
   



   with st.container():
      col1, col2 = st.columns(2)
      with col1:
         st.write("Heat Map")
      with col2:
         def value_to_color(val):
            ind = int((float(val - corr['value'].min()) / (corr['value'].max() - corr['value'].min()))*255)
            return palette[ind]
         
         corr = ds.corr()
         corr = pd.melt(corr.reset_index(), id_vars='index') 
         fig, ax = plt.subplots(figsize=(20,10))
         n_colors = 256
         palette = sns.cubehelix_palette(n_colors) 
         color_min, color_max = [-1, 1] 

         ax.scatter(x = corr['index'].map({p[1]:p[0] for p in enumerate(ds.columns)}),    y = corr['variable'].map({p[1]:p[0] for p in enumerate(ds.columns)}),    s = corr['value'].abs() * 1000,    c = corr['value'].apply(value_to_color),     marker='s')
         ax.set_xticks([x for x in range(len(ds.columns))])
         ax.set_xticklabels(ds.columns, rotation=30, horizontalalignment='right')
         ax.set_yticks([x for x in range(len(ds.columns))])
         ax.set_yticklabels(ds.columns)
         st.pyplot()

          
with tab3:
   st.subheader("Step 5: Machine Learning Algorithms and Step 6: Visualization")
   
   #28 Encode categorical data
   data = pd.get_dummies(data,columns=['REMODEL'])
   data = pd.get_dummies(data, drop_first = True)
   
   multi, stats, decision, randomFor, svr, knn, gradBoost, lasso = st.tabs(["Multiple Linear Regression", "Stats Modeling", "Decision Tree","Random Forest", "SVR","KNN","Gradient Boosting", "Lasso Regression"])
   # Dividing the data
   predictors_df1 = data[['LOT_SQFT', 'YR_BUILT', 'LIVING_AREA', 'FLOORS', 'ROOMS','BEDROOMS', 'FULL_BATH', 'HALF_BATH', 
              'KITCHEN', 'FIREPLACE','REMODEL_Old', 'REMODEL_Recent','REMODEL_None']]
   response_df = data['TOTAL_VALUE']

   z_score_norm = preprocessing.StandardScaler()
   predictor_df_normalized = z_score_norm.fit_transform(predictors_df1)
   predictor_df_normalized = pd.DataFrame(predictor_df_normalized, columns = predictors_df1.columns)
   #predictor_df_normalized

   with multi:
      if st.button("Multiple Linear Regression"):
         # partition data into train and test sets
         X = predictor_df_normalized
         y = response_df
         train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.3, random_state=1)

         # train the LR model
         linear_model = LinearRegression()
         linear_model = linear_model.fit(train_X, train_y)

         # print performance metrics on training set using regressionSummary()
         predicted_y_training = linear_model.predict(train_X)
         regressionSummary(train_y, predicted_y_training)

         # deploy the model on the test data
         predicted_y_test = linear_model.predict(test_X)

         result = pd.DataFrame({'Predicted': predicted_y_test, 'Actual': test_y,'Residual': test_y - predicted_y_test})
         summ = regressionSummary(test_y, predicted_y_test)
         st.write("Regression Summary:",summ)

         # Checking for accuracy
         accuracy = linear_model.score(test_X, test_y)
         st.write("Model Accuracy:",accuracy*100)

         with st.container():
            col1, col2 = st.columns(2)
            with col1:
               # Checking if our residuals are normally distributed
               residuals = test_y - predicted_y_test
               plt.hist(residuals, bins = 50)
               plt.xlim([-200,200])
               plt.xlabel('Residuals Normalised',fontsize=16)
               plt.ylabel('Count',fontsize=16)
               plt.tight_layout()
               st.pyplot()

            with col2:
               # Fitted vs. Residuals Plot
               plt.figure(figsize=(10,10))
               p=plt.scatter(x=predicted_y_test,y=residuals,edgecolor='k')
               xmin = predicted_y_test.min()
               xmax = max(predicted_y_test)
               plt.hlines(y=0,xmin=xmin*1,xmax=xmax*0.9,color='green',linestyle=':',lw=4)
               plt.xlabel("Fitted values",fontsize=15)
               plt.ylabel("Residuals",fontsize=15)
               plt.title("Fitted versus residuals plot",fontsize=16)
               plt.grid(True)
               plt.tight_layout()
               st.pyplot()


         

         

         model_accuracies['Linear regression'] = accuracy         
         model_rmse['Linear regression'] = math.sqrt(mean_squared_error(train_y, predicted_y_training))
         st.session_state.multiAcc = accuracy*100
         st.session_state.multirsme = math.sqrt(mean_squared_error(train_y, predicted_y_training))
      
    
   with stats:
      if st.button("Stats Models"):
         # partition data into train and test sets
         X = predictor_df_normalized
         y = response_df
         train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.3, random_state=1)
         
         # train the  model
         train_X = sm.add_constant(train_X)
         test_X = sm.add_constant(test_X)

         linear_model2 = sm.OLS(list(train_y), train_X).fit()


         # print the coefficients
         #print('intercept ', linear_model.intercept_)
         #print(pd.DataFrame({'Predictor': X.columns, 'coefficient': linear_model.coef_}))
         
         # print performance metrics on training set using regressionSummary()
         predicted_y_training2 = linear_model2.predict(train_X)
         regressionSummary(train_y, predicted_y_training2)

         #drop the variables that are not significant (i.e., p>0.05)
         train_X = train_X.drop(["YR_BUILT","ROOMS","BEDROOMS"], axis = 1)
         test_X = test_X.drop(["YR_BUILT","ROOMS","BEDROOMS"], axis = 1)

         linear_model3 = sm.OLS(list(train_y), train_X).fit()
         predicted_y_training3 = linear_model3.predict(train_X)
         regressionSummary(train_y, predicted_y_training3)

         predicted_y_test3 = linear_model3.predict(test_X)
         regressionSummary(test_y, predicted_y_test3)

         # Checking for accuracy
         accuracy = linear_model3.rsquared
         st.write("Model Accuracy:",accuracy*100)

         # Checking if our residuals are normally distributed
         residuals = test_y - predicted_y_test3
         plt.hist(residuals, bins = 50)
         plt.xlim([-200,200])
         plt.xlabel('Residuals Normalised',fontsize=16)
         plt.ylabel('Count',fontsize=16)
         plt.tight_layout()

         st.pyplot()

   with decision:
      if st.button("Decision tree"):
         # partition data into train and test sets
         X_prediction = predictors_df1
         y_prediction = response_df
         train_X_prediction, test_X_prediction, train_y_prediction, test_y_prediction = train_test_split(X_prediction, 
                                                               y_prediction, test_size=0.3, random_state=616)
         
         # normalise using Z-score
         z_score_norm2 = preprocessing.StandardScaler()
         z_score_norm2.fit(predictors_df1)
         train_X_prediction = pd.DataFrame(z_score_norm2.transform(train_X_prediction), columns = predictors_df1.columns)
         test_X_prediction = pd.DataFrame(z_score_norm2.transform(test_X_prediction), columns = predictors_df1.columns)

         # train the Decision tree model
         WestRoxbury = DecisionTreeRegressor(max_depth=5, random_state=13, splitter="best").fit(train_X_prediction, train_y_prediction) #not allowing it to grow fully,by setting max depth to 7
         predicted_y_training_Rox = WestRoxbury.predict(train_X_prediction)

         predicted_y_test_Rox = WestRoxbury.predict(test_X_prediction)

         r2 = r2_score(test_y_prediction ,predicted_y_test_Rox)
         st.write("Model Accuracy:",r2*100)

         with st.container():
            col1, col2 = st.columns(2)
            with col1:
               # Decision tree for our dataset
               plt.figure(figsize=(20,10))
               plot_tree(WestRoxbury, filled=True,fontsize=10)
               #plt.savefig('Decision_tree.png', dpi=300)
               st.pyplot()
            with col2:
               # Checking if our residuals are normally distributed
               residuals = test_y_prediction - predicted_y_test_Rox
               plt.hist(residuals, bins = 50)
               plt.xlim([-200,200])
               plt.xlabel('Residuals Normalised',fontsize=16)
               plt.ylabel('Count',fontsize=16)
               plt.tight_layout()
               st.pyplot()
          
         model_accuracies['Decision tree'] = r2
         model_rmse['Decision tree'] = math.sqrt(mean_squared_error(test_y_prediction, predicted_y_test_Rox))
         st.session_state.deciAcc = r2*100
         st.session_state.decirsme = math.sqrt(mean_squared_error(test_y_prediction, predicted_y_test_Rox))

   with randomFor:
      if st.button("Random Forest Regressor"):
         # Loading dataset and split it into training and testing sets
         X_prediction = predictors_df1
         y_prediction = response_df
         train_X_prediction, test_X_prediction, train_y_prediction, test_y_prediction = train_test_split(X_prediction, y_prediction, test_size=0.3, random_state=42)
         
         # Train the model
         model = RandomForestRegressor(n_estimators=100, random_state=42)
         model.fit(train_X_prediction, train_y_prediction)

         # Extract feature of importances
         importances = model.feature_importances_

         # Sort feature importances in descending order
         indices = np.argsort(importances)[::-1]

         # Make predictions
         y_pred_rf = model.predict(test_X_prediction)

         # Evaluate the model
         mse = mean_squared_error(test_y_prediction, y_pred_rf)

         predicted_y_training_corollas = model.predict(train_X_prediction)

         # Check R^2 of our test data
         r2 = r2_score(test_y_prediction, y_pred_rf)
         st.write("Model Accuracy:",r2*100)


         with st.container():
            col1, col2 = st.columns(2)
            with col1:
               # Plot the feature importances using a bar plot
               plt.figure(figsize=(10,5))
               plt.title("Feature importances")
               plt.bar(range(train_X_prediction.shape[1]), importances[indices])
               plt.xticks(range(train_X_prediction.shape[1]), train_X_prediction.columns[indices], rotation=90)
               st.pyplot()

            with col2:
               # Plot the first decision tree
               fig, ax = plt.subplots(figsize=(50, 30))
               plot_tree(model.estimators_[0], max_depth=3, ax=ax)
               st.pyplot()

         with st.container():
            col1, col2 = st.columns(2)
            with col1:
               # Plot partial dependence for a specific feature
               fig, ax = plt.subplots(figsize=(8, 6))
               plot_partial_dependence(model, train_X_prediction, ['LIVING_AREA'], ax=ax)
               st.pyplot()
            with col2:
               residuals = y_pred_rf - test_y_prediction
               plt.hist(residuals, bins = 50)
               plt.xlim([-200,200])
               plt.tight_layout()
               st.pyplot()
         


         model_accuracies['Random forest'] = r2
         model_rmse['Random forest'] = math.sqrt(mean_squared_error(test_y_prediction, y_pred_rf))
         st.session_state.randAcc = r2*100
         st.session_state.randrsme = math.sqrt(mean_squared_error(test_y_prediction, y_pred_rf))

   with svr:
      if st.button("Support Vector Regression"):
         # Load your dataset and split it into training and testing sets
         X_prediction = predictors_df1
         y_prediction = response_df
         train_X_prediction, test_X_prediction, train_y_prediction, test_y_prediction = train_test_split(X_prediction, y_prediction, test_size=0.3, random_state=42)

         # Train the model
         model = SVR(kernel='linear')
         model.fit(train_X_prediction, train_y_prediction)

         # Make predictions
         y_pred_svr = model.predict(test_X_prediction)

         # Evaluate the model
         mse = mean_squared_error(test_y_prediction, y_pred_svr)

         predicted_y_training_corollas = model.predict(train_X_prediction)

         # Check R^2 of our test data
         acc = model.score(test_X_prediction,test_y_prediction)
         st.write("Model Accuracy:",acc*100)

         with st.container():
            col1, col2 = st.columns(2)
            with col1:
               plt.scatter(test_y_prediction, y_pred_svr)
               plt.xlabel('Actual values')
               plt.ylabel('Predicted values')
               plt.title('SVR scatter plot')
               st.pyplot()

            with col2:
               residuals = test_y_prediction - y_pred_svr
               plt.hist(residuals, bins = 50)
               plt.xlim([-200,200])
               plt.tight_layout()
               st.pyplot()

      
         
            

         model_accuracies['Support vector Regressor'] = model.score(test_X_prediction,test_y_prediction)
         model_rmse['Support vector Regressor'] = math.sqrt(mean_squared_error(test_y_prediction, y_pred_svr))  
         st.session_state.svrAcc = model.score(test_X_prediction,test_y_prediction)*100
         st.session_state.svrrsme = math.sqrt(mean_squared_error(test_y_prediction, y_pred_svr)) 

   with knn:
      if st.button("KNN"):
         # Load your dataset and split it into training and testing sets
         X_prediction = predictors_df1
         y_prediction = response_df
         train_X_prediction, test_X_prediction, train_y_prediction, test_y_prediction = train_test_split(X_prediction, 
                                                               y_prediction, test_size=0.3, random_state=42)
         # Train the model
         model = KNeighborsRegressor(n_neighbors=10)
         model.fit(train_X_prediction, train_y_prediction)

         # Make predictions
         y_pred_knn = model.predict(test_X_prediction)
         # Evaluate the model
         mse = mean_squared_error(test_y_prediction, y_pred_knn)
         predicted_y_training_corollas = model.predict(train_X_prediction)

         # Check R^2 of our test data
         acc = model.score(test_X_prediction,test_y_prediction)
         st.write("Model Accuracy:",acc*100)

         with st.container():
            col1, col2 = st.columns(2)
            with col1:
               plt.scatter(test_y_prediction, y_pred_knn)
               plt.xlabel('Actual values')
               plt.ylabel('Predicted values')
               plt.title('KNN scatter plot')
               st.pyplot()

            with col2:
               plt.plot(test_y_prediction.values.ravel(), label='Actual values')
               plt.plot(y_pred_knn.ravel(), label='Predicted values')
               plt.title('KNN Regression: Actual vs. Predicted values')
               plt.xlabel('Observation')
               plt.ylabel('Target value')
               plt.legend()
               st.pyplot()




         

         

         model_accuracies['KNN'] = model.score(test_X_prediction,test_y_prediction)
         model_rmse['KNN'] = math.sqrt(mean_squared_error(test_y_prediction, y_pred_knn))

         st.session_state.knnAcc = model.score(test_X_prediction,test_y_prediction)*100
         st.session_state.knnrsme = math.sqrt(mean_squared_error(test_y_prediction, y_pred_knn)) 


   with gradBoost:
      if st.button("Gradient boosting regression"):
         # Dividing the data
         X = predictor_df_normalized

         y = response_df
         train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.3, random_state=1)

         GB = GradientBoostingRegressor(random_state=616).fit(train_X, train_y)
         predicted_y_training = GB.predict(train_X)

         predicted_y_test = GB.predict(test_X)
         r2 = r2_score(test_y, predicted_y_test)
         st.write("Model Accuracy:",r2*100)

         with st.container():
            col1, col2 = st.columns(2)
            with col1:
               #Checking if our residuals are normally distributed
               residuals = test_y - predicted_y_test
               plt.hist(residuals, bins = 50)
               plt.xlim([-200,200])
               plt.xlabel('Residuals Normalised',fontsize=16)
               plt.ylabel('Count',fontsize=16)
               plt.tight_layout()
               st.pyplot()

            with col2:
               feature_imp_GB = pd.Series(GB.feature_importances_, index = predictors_df1.columns)
               # Checking for feature importance with Gradient boost model
               fig, ax = plt.subplots(figsize=(10, 10))
               feature_imp_GB.sort_values().plot.barh(color='red', ax=ax)
               ax.set_xlabel('Count', fontsize=16)
               ax.set_ylabel('Features', fontsize=16)
               ax.set_title('Gradient Boost model')
               st.pyplot(fig)


         

         

         model_accuracies['Gradient Boosting'] = r2
         model_rmse['Gradient Boosting'] = math.sqrt(mean_squared_error(test_y, predicted_y_test))

         st.session_state.gradBAcc = r2*100
         st.session_state.gradBrsme = math.sqrt(mean_squared_error(test_y, predicted_y_test)) 
      
      
   with lasso:
      if st.button("Lasso regression"):
         # Dividing the data
         X = predictor_df_normalized
         y = response_df


         train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.3, random_state=1)

         lasso = Lasso(alpha=0.1)
         lasso.fit(train_X, train_y)
         predicted_y_training = lasso.predict(train_X)
         predicted_y_test = lasso.predict(test_X)

         # Checking for accuracy
         r2 = r2_score(test_y, predicted_y_test)
         st.write("Model Accuracy:",r2*100)

         residuals = test_y - predicted_y_test
         fig, ax = plt.subplots()
         ax.hist(residuals, bins=50)
         ax.set_xlim([-200, 200])
         ax.set_xlabel('Residuals Normalised', fontsize=16)
         ax.set_ylabel('Count', fontsize=16)
         ax.set_title('Residuals Distribution', fontsize=20)
         st.pyplot(fig)

         # Plot for residuals in Lasso model
         visualizer = ResidualsPlot(lasso)
         visualizer.fit(train_X, train_y)  
         visualizer.score(test_X, test_y)  

         # Plot the residuals using Streamlit
         #st.pyplot(visualizer.poof())

         model_accuracies['Lasso Regression'] = r2
         model_rmse['Lasso Regression'] = math.sqrt(mean_squared_error(test_y, predicted_y_test))
         
         st.session_state.lassoAcc = r2*100
         st.session_state.lassorsme = math.sqrt(mean_squared_error(test_y, predicted_y_test)) 
      
with tab4:
   st.subheader("Step 7: Report Findings")

   with st.container():
      col1,col2 = st.columns(2)
      with col1:
         st.subheader("Accuracies:")
         dataAcc = {'Model': ['Multiple Linear Regression', 'Decision Tree', 'Random Forest Regressor', 'SVR', 'KNN', 'Gradient Boosting', 'Lasso Regression'],'Accuracy': [st.session_state.multiAcc, st.session_state.deciAcc, st.session_state.randAcc, st.session_state.svrAcc, st.session_state.knnAcc, st.session_state.gradBAcc, st.session_state.lassoAcc]}
         dfAcc = pd.DataFrame(dataAcc)
         st.table(dfAcc)
      
      with col2:
         st.subheader("RSME:")
         datarsme = {'Model': ['Multiple Linear Regression', 'Decision Tree', 'Random Forest Regressor', 'SVR', 'KNN', 'Gradient Boosting', 'Lasso Regression'],   'RMSE': [st.session_state.multirsme, st.session_state.decirsme, st.session_state.randrsme, st.session_state.svrrsme, st.session_state.knnrsme, st.session_state.gradBrsme, st.session_state.lassorsme]}
         dfrsme = pd.DataFrame(datarsme)
         st.table(dfrsme)

   #st.write("hu", str(st.session_state.lassoAcc))




with tab5:
   st.subheader("Step 8: Build Data Products")
   st.subheader("Enter the data:")

   with st.container():
    col1, col2 = st.columns(2)
    with col1:
        LOT_SQFT = st.text_input('LOT_SQFT', '5000')
        YR_BUILT = st.text_input('YR_BUILT', '1889')
        LIVING_AREA = st.text_input('LIVING_AREA', '1290')
        FLOORS = st.text_input('FLOORS', '2')
        ROOMS = st.text_input('ROOMS', '6')
        BEDROOMS = st.text_input('BEDROOMS', '3')
        FULL_BATH = st.text_input('FULL_BATH', '1')        
    with col2:     
       
       HALF_BATH = st.text_input('HALF_BATH', '0')
       KITCHEN = st.text_input('KITCHEN', '1')
       FIREPLACE = st.text_input('FIREPLACE', '0')
       REMODEL_Old = st.text_input('REMODEL_Old', '0')
       REMODEL_Recent = st.text_input('REMODEL_Recent', '1')
       REMODEL_None = st.text_input('REMODEL_None', '1')
       

       newX = pd.DataFrame({'LOT_SQFT': [LOT_SQFT], 'YR_BUILT': [YR_BUILT], 'LIVING_AREA': [LIVING_AREA], 'FLOORS':[FLOORS], 'ROOMS':[ROOMS],'BEDROOMS':[BEDROOMS], 'FULL_BATH':[FULL_BATH], 'HALF_BATH': [HALF_BATH], 'KITCHEN':[KITCHEN], 'FIREPLACE':[FIREPLACE], 'REMODEL_Old':[REMODEL_Old],'REMODEL_Recent':[REMODEL_Recent],'REMODEL_None':[REMODEL_None]})

   st.write(newX)
   if st.button("Run Gradient Boosting Model"):
      # Dividing the data
      X = predictor_df_normalized
      y = response_df
      train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.3, random_state=1)
      GB = GradientBoostingRegressor(random_state=616).fit(train_X, train_y)
      predictors_df1 = newX[['LOT_SQFT', 'YR_BUILT', 'LIVING_AREA', 'FLOORS', 'ROOMS','BEDROOMS', 'FULL_BATH', 'HALF_BATH', 
              'KITCHEN', 'FIREPLACE','REMODEL_Old', 'REMODEL_Recent','REMODEL_None']]
      z_score_norm = preprocessing.StandardScaler()
      predictor_df_normalized_new = z_score_norm.fit_transform(predictors_df1)
      predictor_df_normalized_new = pd.DataFrame(predictor_df_normalized_new, columns = predictors_df1.columns)
      predicted_y_new = GB.predict(predictor_df_normalized_new)

      st.write("Predicted Total value:", predicted_y_new)





      






