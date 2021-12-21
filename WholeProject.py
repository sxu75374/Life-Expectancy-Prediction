'''
Project Coding Part - Life Expectancy Prediction by Regression - whole project
Entirely done by
sxu75374
'''
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.spatial.distance import cdist
from sklearn.model_selection import StratifiedKFold, train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Lasso, Ridge, SGDRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler, MaxAbsScaler, MinMaxScaler, Normalizer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPRegressor
import pycountry_convert as pc
from scipy.stats import pearsonr
import seaborn as sns
import pickle
import warnings
from mpl_toolkits.mplot3d import Axes3D
warnings.filterwarnings(action='ignore')
plt.ion()
import sys
sys.path.append("/Users/xs/PycharmProjects/EE660FinalProject/TrAdaBoost1.py")


def trivial_system(X, y):
    '''calculate the mean value of Life Expectancy as the trivial system to compare'''
    ypred = [np.mean(y)]*len(y)
    return ypred


def non_trivial_system(Xtrain, ytrain, Xtest, ytest):
    '''use the basic LinearRegression() model as the baseline model'''
    baseline = LinearRegression()
    baseline.fit(Xtrain, ytrain)
    ypred = baseline.predict(Xtest)
    filename = 'non_trivial_system.sav'
    pickle.dump(baseline, open(filename, 'wb'))
    return ypred


def feature_engineer(df):
    '''dataset engineer'''
    # Change categorical feature to numeric value
    df_status = pd.get_dummies(df['Status'])
    df = pd.concat([df, df_status], axis=1)

    # Fix the columns' name (remove blanks)
    df.rename(columns={'Measles ': 'Measles', ' BMI ': 'BMI', 'under-five deaths ': 'under-five deaths',
                       'Diphtheria ': 'Diphtheria', ' HIV/AIDS': 'HIV/AIDS',
                       ' thinness  1-19 years': 'thinness 1-19 years', ' thinness 5-9 years': 'thinness 5-9 years',
                       'Life expectancy ': 'Life expectancy'}, inplace=True)

    # Fix the countries' name (remove brackets)
    df.replace('Bolivia (Plurinational State of)', 'Plurinational State of Bolivia', inplace=True)
    df.replace('Iran (Islamic Republic of)', 'Islamic Republic of Iran', inplace=True)
    df.replace('Micronesia (Federated States of)', 'Federated States of Micronesia', inplace=True)
    df.replace('Republic of Korea', 'South Korea', inplace=True)
    df.replace('The former Yugoslav republic of Macedonia', 'North Macedonia', inplace=True)
    df.replace('Venezuela (Bolivarian Republic of)', 'Bolivarian Republic of Venezuela', inplace=True)

    ''' thanks for pycountry-convert
    https://pypi.org/project/pycountry-convert/
    help me convert country names to the according continents name
    '''
    # Add new features (use country to find corresponding continent)
    countries = df['Country']
    continents = {
        'NA': 'North America',
        'SA': 'South America',
        'AS': 'Asia',
        'OC': 'Oceania',
        'AF': 'Africa',
        'EU': 'Europe',
    }
    temp = list()
    for country in countries:
        country_alpha = pc.country_name_to_country_alpha2(country)
        # if country_alpha == 'TL':
        #     print(country)
        c = continents[pc.country_alpha2_to_continent_code(country_alpha)]
        temp.append(c)
    df['Continent'] = temp

    # transform and map countries to labels
    country_le = LabelEncoder()
    country_labels = country_le.fit_transform(df['Country'])
    df['Country Labels'] = country_labels

    # transform and map continents to labels
    continent_le = LabelEncoder()
    continent_labels = continent_le.fit_transform(df['Continent'])
    df['Continent Labels'] = continent_labels

    plt.figure()
    # plt.scatter(data=df.dropna(), x='GDP', y='Life expectancy', size='Population', color='Continent')
    sns.scatterplot(data=df.dropna(), x="GDP", y="Life expectancy", size="Population", hue="Continent", alpha=0.6, sizes=(50, 600))
    plt.title('Relationship between GDP, Population, Continent and Life expectancy')

    # reorder the features -- remove the feature column 'Population', because there are too many missing values.
    df = df[
        ["Continent", "Country", "Country Labels", "Continent Labels", "Year", "Developed", "Developing", "Adult Mortality", "infant deaths", "Alcohol",
         "percentage expenditure", "Hepatitis B", "Measles", "BMI", "under-five deaths", "Polio", "Total expenditure",
         "Diphtheria", "HIV/AIDS", "GDP", "thinness 1-19 years", "thinness 5-9 years",
         "Income composition of resources", "Schooling", "Life expectancy"]]
    print('unique labels: ', len(df['Country Labels'].unique()))

    # count missing values
    # plt.figure()
    # plt.plot(df.isna().sum()/len(df))
    # plt.show()
    missing_value_after = df.isna().sum()/len(df)
    plt.figure()
    plt.bar(df.columns, missing_value_after, label='percentage of missing values')
    plt.plot([0.05] * 22, linestyle=':', label='5%')
    plt.plot([0.1] * 22, linestyle=':', label='10%')
    plt.plot([0.15] * 22, linestyle=':', label='15%')
    plt.plot([0.2] * 22, linestyle=':', label='20%')
    plt.title('percentage of missing values for each feature after adding new features and delete ole features')
    plt.xlabel('features')
    plt.xticks(rotation=90)
    plt.ylabel('percentage of missing values')
    plt.legend()
    print("-" * 50)
    print('| missing_value for each features after feature engineer: |\n', missing_value_after)

    # simply deal with the missing value
    # df.fillna(df.mean(), inplace=True)

    # fill missing values -- feature correlation analysis
    dropped_df = df.dropna(axis=0, how='any', inplace=False)
    plt.figure()
    corr = dropped_df.corr()
    sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns)
    plt.title('Correlation matrix for features')

    # fill the missing GDP columns
    lr_GDP = LinearRegression()
    lr_GDP.fit(dropped_df['percentage expenditure'].to_numpy().reshape(-1,1), dropped_df['GDP'].to_numpy().reshape(-1,1))
    pred_GDP = lr_GDP.predict(df['percentage expenditure'].to_numpy().reshape(-1,1))
    # print('pred GDP', pred_GDP)
    # print('GDP', df['GDP'])
    # fill the missing GDP columns
    # for row in df['GDP']:
    #     df['GDP'].fillna(value=pred_GDP[row])
    ety = df[df['GDP'].isnull()].index.to_numpy()
    # print(df['GDP'].loc[ety])
    # print('pred_GDP', pred_GDP[ety])
    for i in range(len(ety)):
        index = ety[i]
        df['GDP'].loc[index] = pred_GDP[index]  # pred_GDP[i][0] for linear regression
    plt.figure()
    plt.scatter(dropped_df['percentage expenditure'], dropped_df['GDP'], label='percentage expenditure vs GDP')
    plt.plot(df['percentage expenditure'], pred_GDP, color='red', label='linear relationship')
    plt.xlabel('percentage expenditure')
    plt.ylabel('GDP')
    plt.title('linear relationship between percentage expenditure and GDP')
    plt.legend()

    # print('GDP weight', lr_GDP.intercept_, lr_GDP.coef_)

    # check empty
    print(df[df['GDP'].isnull()].index.to_numpy())
    # plt.figure()
    # plt.plot(df.isna().sum()/len(df))
    # plt.show()

    #     # ety = df[df['GDP'].isnull()].index.to_numpy()
    #     # # print(df['GDP'].loc[ety])
    #     # # print(pred_GDP[ety])
    #     # df['GDP'].fillna(1192.02991098, inplace=True)
    #     # # check empty
    #     # print(df[df['GDP'].isnull()].index.to_numpy())

    # fill the missing schoolingcolumns
    df['Schooling'].fillna(df['Schooling'].mean(), inplace=True)
    lr_sch = LinearRegression()
    lr_sch.fit(dropped_df['Schooling'].to_numpy().reshape(-1,1), dropped_df['Income composition of resources'].to_numpy().reshape(-1,1))
    pred_sch = lr_sch.predict(df['Schooling'].to_numpy().reshape(-1,1))
    ety_sch = df[df['Income composition of resources'].isnull()].index.to_numpy()
    index = []
    for i in range(len(ety_sch)):
        index = ety_sch[i]
        df['Income composition of resources'].loc[index] = pred_sch[index]
    plt.figure()
    plt.scatter(dropped_df['Schooling'], dropped_df['Income composition of resources'], label='Schooling vs Income composition of resources')
    plt.plot(df['Schooling'], pred_sch, color='red', label='linear relationship')
    plt.xlabel('Schooling')
    plt.ylabel('Income composition of resources')
    plt.title('linear relationship between Income composition of resources and Schooling')
    plt.legend()

    # fill the missing Hepatitis B columns
    df['Diphtheria'].fillna(df['Diphtheria'].mean(), inplace=True)
    df['Polio'].fillna(df['Polio'].mean(), inplace=True)
    lr_HB_D = LinearRegression()
    lr_HB_D.fit(dropped_df[['Diphtheria']].to_numpy(), dropped_df['Hepatitis B'].to_numpy().reshape(-1, 1))
    pred_HB_D = lr_HB_D.predict(df[['Diphtheria']].to_numpy())
    # print('pred_HB_D', pred_HB_D)
    # print(lr_HB_D.coef_, lr_HB_D.intercept_)

    lr_HB_P = LinearRegression()
    lr_HB_P.fit(dropped_df[['Polio']].to_numpy(), dropped_df['Hepatitis B'].to_numpy().reshape(-1, 1))
    pred_HB_P = lr_HB_P.predict(df[['Polio']].to_numpy())
    # print('pred_HB_P', pred_HB_P)
    # print(lr_HB_P.coef_, lr_HB_P.intercept_)

    pred_HB = (pred_HB_D + pred_HB_P)/2

    # from mpl_toolkits.mplot3d import Axes3D
    #
    # fig = plt.figure()
    # ax = Axes3D(fig)
    # sequence_containing_x_vals = df['Diphtheria'].tolist()
    # sequence_containing_y_vals = df['Polio'].tolist()
    # sequence_containing_z_vals = pred_HB
    # ax.scatter(sequence_containing_x_vals, sequence_containing_y_vals, sequence_containing_z_vals)
    # ax.set_xlabel('Diptheria')
    # ax.set_ylabel('Polio')
    # ax.set_zlabel('Hepatitis B')

    ety = df[df['Hepatitis B'].isnull()].index.to_numpy()
    for i in range(len(ety)):
        index = ety[i]
        df['Hepatitis B'].loc[index] = pred_HB[index][0]
    # check empty
    # plt.figure()
    # plt.plot(df.isna().sum()/len(df))
    # plt.show()
    # print(df.isna().sum()/len(df))
    print(df[df['Hepatitis B'].isnull()].index.to_numpy())

    plt.figure()
    plt.scatter(dropped_df['Diphtheria'], dropped_df['Hepatitis B'], color='orange', label='Diphtheria datapoints')
    plt.scatter(dropped_df['Polio'], dropped_df['Hepatitis B'], label='Polio datapoints')
    plt.plot(df['Diphtheria'],  pred_HB_D, color='orange', label='Diphtheria predict Hepatitis B')
    plt.plot(df['Polio'],  pred_HB_P, label='Polio predict Hepatitis B')
    y = np.array((np.arange(100)+1)*(lr_HB_D.coef_+lr_HB_P.coef_)/2+(lr_HB_D.intercept_+lr_HB_P.intercept_)/2).reshape(-1,1)
    plt.plot(np.arange(100)+1, y, color='red', label='Average predicted Hepatitis B')
    plt.xlabel('Polio & Diphtheria')
    plt.ylabel('avg Hepatitis B')
    plt.legend()


    # # fill the missing Population columns
    # lr_PP = LinearRegression()
    # df['infant deaths'].fillna(df['infant deaths'].mean(), inplace=True)
    # df['under-five deaths'].fillna(df['under-five deaths'].mean(), inplace=True)
    # print(df[df['Population'].isnull()].index.to_numpy())
    # # print(dropped_df[['Diphtheria', 'Polio']].to_numpy())
    # lr_PP.fit(dropped_df[['under-five deaths', 'infant deaths']].to_numpy(), dropped_df['Population'].to_numpy().reshape(-1, 1))
    # pred_PP = lr_PP.predict(df[['under-five deaths', 'infant deaths']].to_numpy())
    # print('pred_PP', pred_PP)
    # ety = df[df['Population'].isnull()].index.to_numpy()
    # for i in range(len(ety)):
    #     index = ety[i]
    #     df['Population'].loc[index] = pred_PP[index][0]
    # # check empty
    # plt.figure()
    # plt.plot(df.isna().sum()/len(df))
    # plt.show()
    # print(df.isna().sum()/len(df))
    # print(df[df['Population'].isnull()].index.to_numpy())

    # fill missing Life expectancy columns
    df_ing = df.loc[df['Developed'] == 0]
    df_ed = df.loc[df['Developed'] == 1]
    df.loc[(df['Developed'] == 0) & df['Life expectancy']].fillna(df_ing['Life expectancy'].mean(), inplace=True)
    df.loc[(df['Developed'] == 1) & df['Life expectancy']].fillna(df_ed['Life expectancy'].mean(), inplace=True)

    plt.figure()
    plt.hist(dropped_df['Life expectancy'], label='life expectancy distribution')
    plt.title('Life expectancy distribution')
    plt.xlabel('Life expectancy')
    plt.ylabel('number')
    plt.legend()

    print('Average life expectancy for developed country is: ', df_ed['Life expectancy'].mean())
    print('Average life expectancy for developing country is: ', df_ing['Life expectancy'].mean())
    print('Average life expectancy is: ', df['Life expectancy'].mean())

    # fill missing other columns
    df.fillna(df.mean(), inplace=True)
    # check empty
    # plt.figure()
    # plt.plot(df.isna().sum() / len(df))
    # plt.show()
    print(df.isna().sum() / len(df))

    # choose the year from 2000 to 2010 as the training set
    df_train = df[df['Year'].between(2000, 2009)]
    sorted_df_train = df_train.sort_values(by=["Year"], ascending=True)

    # choose the year from 2011 to 2012 as the validation set
    df_val = df[df['Year'].between(2010, 2012)]
    sorted_df_val = df_val.sort_values(by=["Year"], ascending=True)

    # choose the year from 2013 to 2015 as the test set
    df_test = df[df['Year'].between(2013, 2015)]
    sorted_df_test = df_test.sort_values(by=["Year"], ascending=True)

    return df, sorted_df_train, sorted_df_val, sorted_df_test


'''TL for developed = 1, developing = 0, Classification'''
# def get_TL_dataset(df):
#     df_TL = df[
#         ["Continent", "Country", "Continent Labels", "Country Labels", "Year", "Life expectancy", "Adult Mortality", "infant deaths", "Alcohol",
#          "percentage expenditure", "Hepatitis B", "Measles", "BMI", "under-five deaths", "Polio", "Total expenditure",
#          "Diphtheria", "HIV/AIDS", "GDP", "thinness 1-19 years", "thinness 5-9 years",
#          "Income composition of resources", "Schooling", "Developed"]]
#     df_ing = df_TL.loc[df_TL['Developed'] == 0]
#     df_ed = df_TL.loc[df_TL['Developed'] == 1]
#     print('df_ing', df_ing)
#     print('df_ed', df_ed)
#     Xs = df_ing.to_numpy()[:, 2:-1]  # Xs Developing (2426, 21)
#     ys = df_ing['Developed'].to_numpy()  # 00000 ys Developing (2426,)
#     print('Xs Developing', Xs)
#     print('ys Developing', ys)
#     data_T = df_ed[df_ed['Year'].between(2000, 2000)]
#     Xt = data_T.to_numpy()[:, 2:-1]  # Xt Developed(2000) (32, 21)
#     yt = data_T['Developed'].to_numpy()  # yt Developed(2000) (32,)
#     print('Xt Developed(2000)', Xt.shape)
#     print('yt Developed(2000)', yt.shape)
#     data_T_test = df_ed[df_ed['Year'].between(2001, 2015)]  # Xt_test (480, 21)
#     Xt_test = data_T_test.to_numpy()[:, 2:-1]  # yt_test (480,)
#     yt_test = data_T_test['Developed'].to_numpy()
#     print('Xt_test Developed(2001-2015)', Xt_test.shape)
#     print('yt_test Developed(2001-2015)', yt_test.shape)
#     return Xs, ys, Xt, yt, Xt_test, yt_test


def get_TL_dataset(df):
    '''TL for life expectancy regression'''
    df_TL = df[
        ["Continent", "Country", "Country Labels", "Continent Labels",  "Year", "Developed", "Developing", "Adult Mortality", "infant deaths", "Alcohol",
         "percentage expenditure", "Hepatitis B", "Measles", "BMI", "under-five deaths", "Polio", "Total expenditure",
         "Diphtheria", "HIV/AIDS", "GDP", "thinness 1-19 years", "thinness 5-9 years",
         "Income composition of resources", "Schooling", "Life expectancy"]]
    df_ing = df_TL.loc[df_TL['Developed'] == 0]
    df_ed = df_TL.loc[df_TL['Developed'] == 1]
    print('df_ing', df_ing)
    print('df_ed', df_ed)
    Xs = df_ing.to_numpy()[:, 3:-1]  # Xs Developing (2426, 22)
    ys = df_ing['Life expectancy'].to_numpy()  # Life expectancy Developing  ys (2426,)
    print('Xs Developing', Xs.shape)
    print('ys Developing', ys.shape)
    data_T = df_ed[df_ed['Year'].between(2000, 2000)]
    Xt = data_T.to_numpy()[:, 3:-1]  # Xt Developed(2000) (32, 22)
    yt = data_T['Life expectancy'].to_numpy()  # yt Developed(2000) (32,)
    print('Xt Developed(2015)', Xt.shape)
    print('yt Developed(2015)', yt.shape)
    data_T_val = df_ed[df_ed['Year'].between(2001, 2004)]  # Xt_val (480, 22)
    Xt_val = data_T_val.to_numpy()[:, 3:-1]  # yt_val (480,)
    yt_val = data_T_val['Life expectancy'].to_numpy()
    print('Xt_val Developed(2000-2003)', Xt_val.shape)
    print('yt_val Developed(2000-2003)', yt_val.shape)
    data_T_test = df_ed[df_ed['Year'].between(2005, 2015)]  # Xt_test (480, 22)
    Xt_test = data_T_test.to_numpy()[:, 3:-1]  # yt_test (480,)
    yt_test = data_T_test['Life expectancy'].to_numpy()
    print('Xt_test Developed(2004-2014)', Xt_test.shape)
    print('yt_test Developed(2004-2014)', yt_test.shape)
    # ss_tls = StandardScaler()
    # Xs = ss_tls.fit_transform(Xs, ys)
    # ss_tlt = StandardScaler()
    # Xt = ss_tlt.fit_transform(Xt, yt)
    # Xt_val = ss_tlt.transform(Xt_val)
    # Xt_test = ss_tlt.transform(Xt_test)
    return Xs, ys, Xt, yt, Xt_val, yt_val, Xt_test, yt_test


def get_mse(ytrue, ypred):
    mse = mean_squared_error(ytrue, ypred)
    return mse


def ML_system(model, xtrain, ytrain, xtest, ytest):
    model.fit(xtrain, ytrain)
    ypred = model.predict(xtest)
    mse = get_mse(ytest, ypred)
    return mse


def model_selection(model, features_train, label_train):
    temp = []
    for i in range(5):
        X_train, X_test, y_train, y_test = train_test_split(features_train, label_train, test_size=0.2, train_size=0.8,
                                                            shuffle=True)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = get_mse(y_test, y_pred)
        temp.append(mse)
    all_MSE = temp
    mean_MSE = np.mean(temp)
    std_MSE = np.std(temp)
    return all_MSE, mean_MSE, std_MSE


def select_normalization_techniques(model, normalization_techniques, x_train, y_train, times):
    cv_result = []
    mean_mse = []
    std_mse = []
    # ncompo = np.linspace(1, 50, 50)
    # print(ncompo)
    for i in range(times):
        data = normalization_techniques.fit_transform(x_train)  # wine_train_features
        # pf = PolynomialFeatures(degree=2)
        # Xtrain_expand = pf.fit_transform(Xtrain, Ytrain)
        # pca = PCA(n_components=int(ncompo[i]))
        # Xtrain_pca = pca.fit_transform(Xtrain_expand, Ytrain)
        result, mean, std = model_selection(model, data, y_train)
        cv_result.append(result)
        mean_mse.append(mean)
        std_mse.append(std)
    cv_result = np.array(cv_result)
    mean_mse = np.array(mean_mse)
    std_mse = np.array(std_mse)
    print('all median steps: ', cv_result)
    print('validation mean MSE for normalization technique:', normalization_techniques, ' is:', np.mean(mean_mse),
          'median steps is:', mean_mse)
    print('validation mean std of MSE for normalization technique:', normalization_techniques, ' is: ', np.mean(std_mse),
          'median step is:', std_mse)
    print('\n')


def select_best_k_features(model, kbest, x_train, y_train, times):
    cv_result = []
    mean_mse = []
    std_mse = []
    for i in range(times):
        result, mean, std = model_selection(model, x_train, y_train)
        cv_result.append(result)
        mean_mse.append(mean)
        std_mse.append(std)
    cv_result = np.array(cv_result)
    mean_mse = np.array(mean_mse)
    std_mse = np.array(std_mse)
    # print('median steps: ', cv_result)
    print('validation mean MSE for ', kbest, ' best features based on feature importance is:', np.mean(mean_mse), 'median steps is:', mean_mse)
    print('validation mean std of MSE for ', kbest, ' best features based on feature importance is:', np.mean(std_mse), 'median steps is:', std_mse)
    print('\n')
    return np.mean(mean_mse), np.mean(std_mse)


def get_SSL_dataset(df, u_percent):
    remove_n = round(len(df_engineered)/(2015-2000) * u_percent)
    print('remove n', remove_n)
    temp = list()
    for i in range(15):
        year = 2000+i
        # print('year', year)
        idx = df_engineered[df_engineered['Year'].between(year, year)].index.tolist()
        # print('idx',idx)
        drop_indices = np.random.choice(idx, remove_n, replace=False)
        # print('drop indices', drop_indices)
        temp = np.concatenate([temp, drop_indices], axis=0)
    temp = np.array(temp)
    n_u = len(temp)
    df_subset = df_engineered
    for t in range(len(temp)):
        df_subset.loc[temp[t], 'Life expectancy'] = -1  # np.nan
    U_X = df_subset.to_numpy()[:, 3:-1]
    U_y = df_subset.to_numpy()[:, -1]
    return U_X, U_y, n_u


def get_SSL_dataset4LP(df):
    # choose the year from 2000 to 2007 as the training set
    df_train = df[df['Year'].between(2000, 2007)]
    sorted_df_train = df_train.sort_values(by=["Year"], ascending=True)
    get_X_l_lp = sorted_df_train[sorted_df_train.columns.drop(['Life expectancy'])]
    get_y_l_lp = sorted_df_train['Life expectancy']

    # choose the year from 2008 to 2009 as the validation set
    df_val = df[df['Year'].between(2008, 2009)]
    sorted_df_val = df_val.sort_values(by=["Year"], ascending=True)
    get_X_val_lp = sorted_df_val[sorted_df_val.columns.drop(['Life expectancy'])]
    get_y_val_lp = sorted_df_val['Life expectancy']

    # choose the year from 2010 to 2012 as the test set
    df_test = df[df['Year'].between(2010, 2012)]
    sorted_df_test = df_test.sort_values(by=["Year"], ascending=True)
    get_X_test_lp = sorted_df_test[sorted_df_test.columns.drop(['Life expectancy'])]
    get_y_test_lp = sorted_df_test['Life expectancy']

    # choose the year from 2013 to 2015 as the training set
    df_u = df[df['Year'].between(2013, 2015)]
    sorted_df_u = df_u.sort_values(by=["Year"], ascending=True)
    get_X_u_lp = sorted_df_u[sorted_df_u.columns.drop(['Life expectancy'])]
    get_y_u_lp = sorted_df_u['Life expectancy']

    return get_X_l_lp.to_numpy()[:, 3:], get_y_l_lp.to_numpy(), get_X_val_lp.to_numpy()[:, 3:],\
           get_y_val_lp.to_numpy(), get_X_test_lp.to_numpy()[:, 3:], get_y_test_lp.to_numpy(),\
           get_X_u_lp.to_numpy()[:, 3:], get_y_u_lp.to_numpy()


def plot_grid_search(cv_results, grid_param_1, grid_param_2, name_param_1, name_param_2):
    # Get Test Scores Mean and std for each grid search
    scores_mean_train = cv_results['mean_train_score']
    scores_mean_train = np.array(scores_mean_train).reshape(len(grid_param_2),len(grid_param_1))
    scores_mean = cv_results['mean_test_score']
    scores_mean = np.array(scores_mean).reshape(len(grid_param_2),len(grid_param_1))

    # scores_sd_train = cv_results['std_train_score']
    # scores_sd_train = np.array(scores_sd_train).reshape(len(grid_param_2),len(grid_param_1))
    scores_sd = cv_results['std_test_score']
    scores_sd = np.array(scores_sd).reshape(len(grid_param_2),len(grid_param_1))

    # Plot Grid search scores
    _, ax = plt.subplots(1,1)

    # Param1 is the X-axis, Param 2 is represented as a different curve (color line)
    for idx, val in enumerate(grid_param_2):
        ax.plot(grid_param_1, scores_mean[idx,:], '-o', label=name_param_2 + str(val) + ' (testset) ')
        ax.plot(grid_param_1, scores_mean_train[idx, :], '-o', label=name_param_2 + str(val) + ' (trainset) ')

    ax.set_title("Grid Search Scores", fontsize=20, fontweight='bold')
    ax.set_xlabel(name_param_1, fontsize=16)
    ax.set_ylabel('CV Average Score', fontsize=16)
    ax.legend(loc="best", fontsize=15)
    ax.grid('on')


def regression_visualization(X4plot, Y, final_model):

    fig = plt.figure()
    ax = Axes3D(fig)
    sequence_containing_x_vals = X4plot[:, 0]
    sequence_containing_y_vals = X4plot[:, 1]
    sequence_containing_z_vals = Y
    ax.scatter(sequence_containing_x_vals, sequence_containing_y_vals, sequence_containing_z_vals)
    ax.set_xlabel('HIV/AIDS')
    ax.set_ylabel('Adult Mortality')
    ax.set_zlabel('Life expectancy')

    coefs = final_model.coef_
    intercept = final_model.intercept_
    xs = np.tile(np.arange(10), (10,1))
    ys = np.tile(np.arange(10), (10,1)).T
    zs = xs*coefs[0]+ys*coefs[1]+intercept
    print("Equation: y = {:.2f} + {:.2f}x1 + {:.2f}x2".format(intercept, coefs[0],
                                                              coefs[1]))
    ax.plot_surface(xs,ys,zs, alpha=0.5)


def tree_regression_visualization(XP, YP, model, pointlabel, curvelabel, Xlabel, Ylabel, Title):
    x_min_cart2 = np.min(XP)
    x_max_cart2 = np.max(XP)
    xrange_cart2 = np.arange(x_min_cart2, x_max_cart2, 1)

    model.fit(XP.reshape(-1,1), YP.reshape(-1,1))
    pred_cartp2 = model.predict(xrange_cart2.reshape(-1,1))
    plt.figure()
    plt.scatter(XP, YP, label=pointlabel)
    plt.plot(xrange_cart2, pred_cartp2, color='red', label=curvelabel)
    plt.xlabel(Xlabel)
    plt.ylabel(Ylabel)
    plt.title(Title)
    plt.legend()


def TL_visualization(XP, YP,XT, YT, model, pointlabels,pointlabelt, curvelabel, Xlabel, Ylabel, Title):
    x_min_cart2 = np.min(XP)
    x_max_cart2 = np.max(XP)
    xrange_cart2 = np.arange(x_min_cart2, x_max_cart2, 1)

    model.fit(XP.reshape(-1,1), YP)
    pred_cartp2 = model.predict(xrange_cart2.reshape(-1,1))
    plt.figure()
    plt.scatter(XP[:2000], YP[:2000], label=pointlabels)
    plt.scatter(XT,YT, label=pointlabelt)
    plt.plot(xrange_cart2, pred_cartp2, color='red', label=curvelabel)
    plt.xlabel(Xlabel)
    plt.ylabel(Ylabel)
    plt.title(Title)
    plt.legend()


def LPvisualization(X,y, B, LA1, LA2, Xlabel, Ylabel, Title):
    x_min_cart2 = np.min(X)
    x_max_cart2 = np.max(X)
    xrange_cart2 = np.arange(x_min_cart2, x_max_cart2, 1)
    plt.figure()
    plt.scatter(X, y, label=LA1)
    plt.plot(xrange_cart2, B, color='red', label=LA2)
    plt.xlabel(Xlabel)
    plt.ylabel(Ylabel)
    plt.title(Title)
    plt.legend()
    plt.show()


'''load data for Supervised Learning'''
# Load DataFrame
df_original = pd.read_csv('Life Expectancy Data.csv')
print('unique: ', len(df_original['Country'].unique()))

# Original data set feature analysis (all features)
missing_value = df_original.isna().sum() / len(df_original)
plt.figure()
plt.bar(df_original.columns, missing_value, label='percentage of missing values')
plt.plot([0.05]*22, linestyle=':', label='5%')
plt.plot([0.1]*22, linestyle=':', label='10%')
plt.plot([0.15]*22, linestyle=':', label='15%')
plt.title('percentage of missing values for each feature of original dataset')
plt.xlabel('features')
plt.xticks(rotation=90)
plt.ylabel('percentage of missing values')
plt.legend()
print("-"*50)
print('| Missing value for each features: |\n', missing_value)

# Data set after feature engineering (1. add new features, 2. remove features, 3. fill missing values)
df_engineered, Dtrain, Dval, Dtest = feature_engineer(df_original)
print('df eg', df_engineered)
print('df_engineered shape', len(df_engineered))
print('Train data', Dtrain.shape)
print('Val data', Dval.shape)
print('Test data', Dtest.shape)

# X, y for Supervised Learning
Xtrain = Dtrain.to_numpy()[:, 3:-1]  # (2013, 22)
Ytrain = Dtrain.to_numpy()[:, -1]  # (2013, )
Xval = Dval.to_numpy()[:, 3:-1]  # (366, 22)
Yval = Dval.to_numpy()[:, -1]  # (366, )

# set aside test set for Supervised Learning
Xtest = Dtest.to_numpy()[:, 3:-1]  # (559, 22)
Ytest = Dtest.to_numpy()[:, -1]  # (559, )

'''before training SL model'''
# 11111111111 feature analysis!!!!!!!!!!!!!!!!!!!!!!!!!! in order to #2

# 1. get feature importance by RF and sort descending
rfr = RandomForestRegressor(n_estimators=50)
rfr.fit(Xtrain, Ytrain)
FI_df = pd.DataFrame({'Features': list(Dtrain.columns.drop(['Continent', 'Country', "Continent Labels",  'Life expectancy'])),
                      'Feature importance': rfr.feature_importances_})
FI = FI_df.sort_values('Feature importance', ascending=False)
print('Feature Importance based on engineered feature space: \n', FI)
plt.figure()
plt.bar(FI['Features'], FI['Feature importance'])
plt.title('Feature Importance based on original feature space')
plt.xlabel('Features')
plt.xticks(rotation=90)
plt.ylabel('Feature Importance')

# usc PCA to check data distribution
from sklearn.decomposition import PCA
# pca = PCA(n_components=1)
# Xtrain_pca = pca.fit_transform(Dtrain[Dtrain.columns.drop(['Continent', 'Country','Developed', 'Developing'])], Dtrain['Developed'])
# Xval_pca = pca.transform(Dval[Dval.columns.drop(['Continent', 'Country','Developed', 'Developing'])])
# Xtest_pca = pca.transform(Dtest[Dtest.columns.drop(['Continent', 'Country','Developed', 'Developing'])])
plt.figure()
plt.scatter(Dtrain['Life expectancy'], Dtrain['Developed'], color='blue')
plt.scatter(Dval['Life expectancy'], Dval['Developed'], color='red')
plt.scatter(Dtest['Life expectancy'], Dtest['Developed'], color='green')

# 22222222 feature preprocessing normal, feature expansion/reduction..... # for good result

# find best normalization techniques
print('=========================== find best normalization technique ===========================')
print('\n')
lr_normal_technique = LinearRegression()
tech1 = StandardScaler()
select_normalization_techniques(model=lr_normal_technique, normalization_techniques=tech1,
                                x_train=Xtrain, y_train=Ytrain, times=5)
tech2 = MaxAbsScaler()
select_normalization_techniques(model=lr_normal_technique, normalization_techniques=tech2,
                                x_train=Xtrain, y_train=Ytrain, times=5)
tech3 = MinMaxScaler()
select_normalization_techniques(model=lr_normal_technique, normalization_techniques=tech3,
                                x_train=Xtrain, y_train=Ytrain, times=5)
tech4 = Normalizer()
select_normalization_techniques(model=lr_normal_technique, normalization_techniques=tech4,
                                x_train=Xtrain, y_train=Ytrain, times=5)

# feature dimension adjustment
model_k_best = DecisionTreeRegressor()  # RandomForestRegressor(n_estimators=3)=9, DecisionTreeRegressor()=10, LR=22
MSE_k = list()
STD_k = list()
K = list()
for k in range(len(FI)):  # len(FI)=22
    print('iter k =', k+1)
    col = FI.nlargest(k+1, 'Feature importance')['Feature importance'].index
    col = col.to_numpy()
    # print('col', col)
    Xtrain_k = Xtrain[:, col]
    # print('X train of k best feature',Xtrain_k, Xtrain_k.shape)
    mse_k, std_k = select_best_k_features(model=model_k_best, kbest=k+1, x_train=Xtrain_k, y_train=Ytrain, times=5)
    MSE_k.append(mse_k)
    STD_k.append(std_k)
    K.append(k+1)
MSE_k = np.array(MSE_k)
STD_k = np.array(STD_k)
print('best k = ', np.argmin(MSE_k)+1)
print('best MSE with k features is: ', MSE_k[np.argmin(MSE_k)])
print('std of best MSE with k features is: ', STD_k[np.argmin(MSE_k)])
plt.figure()
plt.scatter(np.argmin(MSE_k)+1, MSE_k[np.argmin(MSE_k)], marker='o', label='Best feature number K')
plt.plot(K, MSE_k, label='best K features vs MSE score')
plt.title('best feature number K vs MSE score')
plt.xlabel('best feature number K')
plt.ylabel('MSE score')
plt.legend()

# 3333333 model selection feature preprcessing techniques # for good result


'''Supervised Learning'''
# data after feature engineer and feature selection, k=6
col_SL = FI.nlargest(6, 'Feature importance')['Feature importance'].index
print('col SL', col_SL)
# ss = StandardScaler()
# X_train_SL = ss.fit_transform(Xtrain[:, col_SL], Ytrain)
# Y_train_SL = Ytrain
# X_val_SL = ss.transform(Xval[:, col_SL])
# Y_val_SL = Yval
# X_test_SL = ss.transform(Xtest[:, col_SL])
# Y_test_SL = Ytest
#
# ply = PolynomialFeatures(degree=2)
# X_train_SL = ply.fit_transform(X_train_SL, Y_train_SL)
# X_val_SL = ply.transform(X_val_SL)
# X_test_SL = ply.transform(X_test_SL)

ply = PolynomialFeatures(degree=2)
X_train_SL = ply.fit_transform(Xtrain[:, col_SL], Ytrain)
Y_train_SL = Ytrain
X_val_SL = ply.transform(Xval[:, col_SL])
Y_val_SL = Yval
X_test_SL = ply.transform(Xtest[:, col_SL])
Y_test_SL = Ytest
ss = StandardScaler()
X_train_SL = ss.fit_transform(X_train_SL, Y_train_SL)
X_val_SL = ss.transform(X_val_SL)
X_test_SL = ss.transform(X_test_SL)

# data for plot
ssp = StandardScaler()
xp = ssp.fit_transform(Xtrain[:, col_SL], Ytrain)
yp = Ytrain
xtp = ssp.transform(Xtest[:, col_SL])
ytp = Ytest

print('================================ Supervised Learning Part ================================')


# model 0: trivial system
pred_trivial = trivial_system(X_test_SL, Y_test_SL)
mse_trivial = get_mse(Y_test_SL, pred_trivial)
print('Test MSE of Baseline trivial system: ', mse_trivial)
print('Test R^2 of Baseline trivial system: ', r2_score(Y_test_SL, pred_trivial))

# model 0: non-trivial system
pred_non_trivial = non_trivial_system(X_train_SL, Y_train_SL, X_test_SL, Y_test_SL)
mse_non_trivial = get_mse(Y_test_SL, pred_non_trivial)
print('Test MSE of Baseline non-trivial system: ', mse_non_trivial)
print('Test R^2 of Baseline non-trivial system: ', r2_score(Y_test_SL, pred_non_trivial))


# # model 1: Polynomial Linear Regression
# print('\n')
# print('=== Model 1: Polynomial Linear Regression ===')
#
# # best model parameter selection - Polynomial features
# D = list()
# MSE_val_p = list()
# for d in range(4):
#     poly = PolynomialFeatures(degree=d+1)  # best poly=2
#     D.append(d+1)
#     X_train_SL_p = poly.fit_transform(X_train_SL, Y_train_SL)
#     X_val_SL_p = poly.transform(X_val_SL)
#     X_test_SL_p = poly.transform(X_test_SL)
#     lr = LinearRegression()
#     lr.fit(X_train_SL_p, Y_train_SL)
#     val_pred = lr.predict(X_val_SL_p)
#     # print(val_pred)
#     mse_val_p = mean_squared_error(Y_val_SL, val_pred)
#     print('Polynomial degree = ', d+1, ', Validation MSE of Polynomial regression = ', mse_val_p)
#     MSE_val_p.append(mse_val_p)
# print('best polynomial feature degree D is: ', np.argmin(MSE_val_p)+1)
# print('best MSE with', np.argmin(MSE_val_p)+1, ' feature degree is: ', MSE_val_p[np.argmin(MSE_val_p)])
# plt.figure()
# plt.plot(D, MSE_val_p, label='polynomial feature degree vs MSE score')
# plt.scatter(np.argmin(MSE_val_p)+1, MSE_val_p[np.argmin(MSE_val_p)], marker='o')
# plt.title('polynomial feature degree D vs MSE score')
# plt.xlabel('polynomial feature degree D')
# plt.ylabel('MSE score')
# plt.legend()
#
#
# best_degree = 2
# polyt = PolynomialFeatures(degree=best_degree)
# X_train_SL_pt = polyt.fit_transform(X_train_SL, Y_train_SL)
# lrt = LinearRegression()
# lrt.fit(X_train_SL_pt, Y_train_SL)
# pred_poly_train = lrt.predict(X_train_SL_pt)
# print('Training MSE of Polynomial regression = ', mean_squared_error(Y_train_SL, pred_poly_train))
#
# # test
# X_test_SL_pt = polyt.transform(X_test_SL)
# pred_poly_test = lrt.predict(X_test_SL_pt)
# # print(val_pred)
# print('when best degree = ', best_degree, ', Test MSE of Polynomial regression = ', mean_squared_error(Y_test_SL, pred_poly_test))


# model 1: SVR
from sklearn.svm import SVR
print('\n')
print('=== Model 1: SVR ===')

C = list()
for c in range(7):
    temp = -3+c
    C.append(10**(temp))
print('C for svr', C)

Gamma = list()
for g in range(7):
    temp = -4+g
    Gamma.append(10**(temp))
print('Gamma for svr', Gamma)

# best model parameter selection
param_grid_svr= [{'C': C,
                  'gamma': Gamma
                  }]
grid_svr = SVR()
grid_search_svr = GridSearchCV(grid_svr, param_grid_svr, cv=5, verbose=1, return_train_score=True)
grid_search_svr.fit(X_train_SL, Y_train_SL)
best_svr = grid_search_svr.best_estimator_
best_svr.fit(X_train_SL, Y_train_SL)
pred_best_svr = best_svr.predict(X_val_SL)

print('Validation MSE of svr best model is: ', get_mse(Y_val_SL, pred_best_svr))
print('Validation svr best parameter is: ', grid_search_svr.best_params_)
print('lasso cv_result:', grid_search_svr.cv_results_)

pred_best_svr_train = best_svr.predict(X_train_SL)
print('Training MSE of svr best model is: ', mean_squared_error(Y_train_SL, pred_best_svr_train))

# test
pred_best_svr_test = best_svr.predict(X_test_SL)
print('Test MSE of svr best model is: ', mean_squared_error(Y_test_SL, pred_best_svr_test))
print('Test R^2 of svr best model is: ', r2_score(Y_test_SL, pred_best_svr_test))
print('number of support vector', best_svr.n_support_)

svr_new = SVR(C=100, gamma=0.1)
svr_new.fit(X_train_SL, Y_train_SL)
svrt = svr_new.predict(X_test_SL)
print('Training MSE of svr 100 0.1 is: ', mean_squared_error(Y_train_SL, svr_new.predict(X_train_SL)))
print('Test MSE of svr 100 0.1 is: ', mean_squared_error(Y_test_SL, svrt))
print('Test R^2 of svr 100 0.1 is: ', r2_score(Y_test_SL, svrt))

# Calling Method
scores = grid_search_svr.cv_results_['mean_test_score'].reshape(7,7)
plt.figure(figsize=(8, 6))
plt.subplots_adjust(left=0.2, right=0.95, bottom=0.15, top=0.95)
plt.imshow(
    scores,
    interpolation="nearest",
    cmap=plt.cm.hot
)
plt.xlabel("gamma")
plt.ylabel("C")
plt.colorbar()
plt.xticks(np.arange(7), Gamma, rotation=45)
plt.yticks(np.arange(7), C)
plt.title("Validation accuracy")

# visualization
'''
thanks for the user Saul Dobilas for the code of visualization
https://towardsdatascience.com/support-vector-regression-svr-one-of-the-most-flexible-yet-robust-prediction-algorithms-4d25fbdaca60
to show the 3D regression plane for SVR
'''
import plotly.graph_objects as go
import plotly.express as px
x_min = np.min(xp[:,0])
x_max = np.max(xp[:,0])
y_min = np.min(xp[:,1])
y_max = np.max(xp[:,1])
mesh_size = 1
xrange = np.arange(x_min, x_max, mesh_size)
yrange = np.arange(y_min, y_max, mesh_size)
xx, yy = np.meshgrid(xrange, yrange)
svr_z = SVR(C=100, gamma=0.1)
svr_z.fit(xp[:,0:2], Y_train_SL)
svrp_z = svr_z.predict(np.c_[xx.ravel(), yy.ravel()])
svrp_z = svrp_z.reshape(xx.shape)
# Create a 3D scatter plot with predictions
fig = px.scatter_3d(xp, xp[:,0], xp[:,1], yp,
                 opacity=0.8, color_discrete_sequence=['black'])

# Set figure title and colors
fig.update_layout(title_text="Scatter 3D Plot with SVR Prediction Surface",
                  scene = dict(xaxis=dict(backgroundcolor='white',
                                          color='black',
                                          gridcolor='lightgrey'),
                               yaxis=dict(backgroundcolor='white',
                                          color='black',
                                          gridcolor='lightgrey'
                                          ),
                               zaxis=dict(backgroundcolor='white',
                                          color='black',
                                          gridcolor='lightgrey')))
# Update marker size
fig.update_traces(marker=dict(size=3))

# Add prediction plane
fig.add_traces(go.Surface(x=xrange, y=yrange, z=svrp_z, name='SVR',
                          colorscale=px.colors.sequential.Plotly3))
fig.show()


# model 2: l1 lasso
print('\n')
print('=== Model 2: Lasso Regression (Linear Regression with l1 regularization) ===')

# best model parameter selection
alpha = list()
for d in range(13):
    temp = -10+d
    alpha.append(2**(temp))
print('Alpha for lasso', alpha)
param_grid_lasso = [{'alpha': alpha}]
grid_lasso = Lasso()
grid_search_lasso = GridSearchCV(grid_lasso, param_grid_lasso, cv=5, verbose=1, return_train_score=True)
grid_search_lasso.fit(X_train_SL, Y_train_SL)
best_lasso = grid_search_lasso.best_estimator_
best_lasso.fit(X_train_SL, Y_train_SL)
pred_best_lasso = best_lasso.predict(X_val_SL)

print('Validation MSE of Lasso best model is: ', get_mse(Y_val_SL, pred_best_lasso))
print('Validation Lasso best parameter is: ', grid_search_lasso.best_params_)
print('lasso cv_result:', grid_search_lasso.cv_results_)

pred_best_lasso_train = best_lasso.predict(X_train_SL)
print('Training MSE of Lasso best model is: ', mean_squared_error(Y_train_SL, pred_best_lasso_train))

# test
pred_best_lasso_test = best_lasso.predict(X_test_SL)
print('Test MSE of Lasso best model is: ', mean_squared_error(Y_test_SL, pred_best_lasso_test))
print('Test R^2 of Lasso best model is: ', r2_score(Y_test_SL, pred_best_lasso_test))

# Calling Method
Alpha = alpha
maxiter = ['None']
plot_grid_search(grid_search_lasso.cv_results_, Alpha, maxiter, 'Alpha', 'lasso Max iter = ')

'''
Thanks for user aricooperdavis provides the method of plot a 3d regression plane
https://gist.github.com/aricooperdavis/c658fc1c5d9bdc5b50ec94602328073b
'''
fig = plt.figure()
ax = Axes3D(fig)
sequence_containing_x_vals = xp[:, 0]
print(xp)
sequence_containing_y_vals = xp[:, 1]
print('sex', sequence_containing_x_vals)
print('sey', sequence_containing_y_vals)
sequence_containing_z_vals = Y_train_SL
ax.scatter(sequence_containing_x_vals, sequence_containing_y_vals, sequence_containing_z_vals)
ax.set_xlabel('HIV/AIDS')
ax.set_ylabel('Adult Mortality')
ax.set_zlabel('Life expectancy')
plt.title('regression plane')

# best_lasso.fit(X_train_SL[:, 0:1], Y_train_SL)
# best_lasso.predict(X_test_SL[:, 0:1])

coefs = best_lasso.coef_
intercept = best_lasso.intercept_
xs = np.tile(np.arange(10), (10,1))
ys = np.tile(np.arange(10), (10,1)).T
zs = xs*coefs[0]+ys*coefs[1]+intercept
print("Equation: y = {:.2f} + {:.2f}x1 + {:.2f}x2".format(intercept, coefs[0],
                                                          coefs[1]))
ax.plot_surface(xs,ys,zs, alpha=0.5)


# X_grid = np.arange(min(X), max(X), 0.01)
# X_grid = X_grid.reshape((len(X_grid), 1))
# plt.scatter(X, y, color = 'red')
# plt.plot(X_grid, best_lasso.predict(X_grid), color = 'blue')
# plt.title('Truth or Bluff (Decision Tree Regression)')
# plt.xlabel('Position level')
# plt.ylabel('Salary')
# plt.show()


# model 3: l2 ridge
print('\n')
print('=== Model 3: Ridge Regression (Linear Regression with l2 regularization) ===')

# best model parameter selection
print('Alpha for ridge', alpha)
param_grid_ridge = [{'alpha': alpha}]
grid_ridge = Ridge()
grid_search_ridge = GridSearchCV(grid_ridge, param_grid_ridge, cv=5, verbose=1, return_train_score=True)
grid_search_ridge.fit(X_train_SL, Y_train_SL)
best_ridge = grid_search_ridge.best_estimator_
best_ridge.fit(X_train_SL, Y_train_SL)
pred_best_ridge = best_ridge.predict(X_val_SL)

print('Validation MSE of Ridge best model is: ', get_mse(Y_val_SL, pred_best_ridge))
print('Validation Ridge best parameter is: ', grid_search_ridge.best_params_)
print('ridge cv_result:', grid_search_ridge.cv_results_)

pred_best_ridge_train = best_ridge.predict(X_train_SL)
print('Training MSE of Ridge best model is: ', mean_squared_error(Y_train_SL, pred_best_ridge_train))

# test
pred_best_ridge_test = best_ridge.predict(X_test_SL)
print('Test MSE of Ridge best model is: ', mean_squared_error(Y_test_SL, pred_best_ridge_test))
print('Test R^2 of Ridge best model is: ', r2_score(Y_test_SL, pred_best_ridge_test))

# Calling Method
Alpha = alpha
maxiter = ['None']
plot_grid_search(grid_search_ridge.cv_results_, Alpha, maxiter, 'Alpha', 'ridge Max iter = ')

# model 4: CART
print('\n')
print('=== Model 4: CART ===')

# best model parameter selection
param_grid_cart = [{'max_depth': np.arange(3,30,1).tolist()}]
grid_cart = DecisionTreeRegressor(random_state = 0)
grid_search_cart = GridSearchCV(grid_cart, param_grid_cart, cv=5, verbose=1, return_train_score=True)
grid_search_cart.fit(X_train_SL, Y_train_SL)
best_cart = grid_search_cart.best_estimator_
best_cart.fit(X_train_SL, Y_train_SL)
pred_best_cart = best_cart.predict(X_val_SL)

print('Validation MSE of CART best model is: ', get_mse(Y_val_SL, pred_best_cart))
print('Validation CART best parameter is: ', grid_search_cart.best_params_)
print('cart cv_result:', grid_search_cart.cv_results_)

pred_best_cart_train = best_cart.predict(X_train_SL)
print('Training MSE of CART best model is: ', mean_squared_error(Y_train_SL, pred_best_cart_train))

# test
pred_best_cart_test = best_cart.predict(X_test_SL)
print('Test MSE of CART best model is: ', mean_squared_error(Y_test_SL, pred_best_cart_test))
print('Test R^2 of CART best model is: ', r2_score(Y_test_SL, pred_best_cart_test))

cart_new = DecisionTreeRegressor(max_depth=11, random_state=0)
cart_new.fit(X_train_SL, Y_train_SL)
cartt=cart_new.predict(X_test_SL)
print('Training MSE of cart 11 is: ', mean_squared_error(Y_train_SL, cart_new.predict(X_train_SL)))
print('Test MSE of cart 11 is: ', mean_squared_error(Y_test_SL, cartt))
print('Test R^2 of cart 11 is: ', r2_score(Y_test_SL, cartt))

# Calling Method
max_depth = np.arange(3,30,1).tolist()
max_features = ['None']
plot_grid_search(grid_search_cart.cv_results_, max_depth, max_features, 'N Estimators', 'CART Max Features = ')

cartp = DecisionTreeRegressor(max_depth=11, random_state=0)
tree_regression_visualization(xp[:,0], yp, cartp,
                              pointlabel='HIV/AIDS vs Life expectancy',
                              curvelabel='CART regression',
                              Xlabel='HIV/AIDS',
                              Ylabel='Life expectancy',
                              Title='CART Regression for one salient feature (HIV/AIDS)'
                              )
tree_regression_visualization(xp[:,1], yp, cartp,
                              pointlabel='Adult Morality vs Life expectancy',
                              curvelabel='CART regression',
                              Xlabel='Adult Morality',
                              Ylabel='Life expectancy',
                              Title='CART Regression for one salient feature (Adult Morality)'
                              )

# model 5: random forest
print('\n')
print('=== Model 5: Random Forest ===')

# best model parameter selection
param_grid_rf = [{ 'max_depth': [11],#[5, 7, 8, 9, 11, 15, 20, 25,30,35,50],
                   'n_estimators':  [3, 8, 10, 12, 15, 20, 25, 30, 35,36,37,38,39,40,41,42,43,44,45,47,48,50,53,55,57,60]}]
grid_rf = RandomForestRegressor(random_state=0)
grid_search_rf = GridSearchCV(grid_rf, param_grid_rf, cv=5, verbose=1, return_train_score=True)
grid_search_rf.fit(X_train_SL, Y_train_SL)
best_rf = grid_search_rf.best_estimator_
best_rf.fit(X_train_SL, Y_train_SL)
pred_best_rf = best_rf.predict(X_val_SL)
print('Validation MSE of RF best model is: ', get_mse(Y_val_SL, pred_best_rf))
print('Validation RF best parameter is: ', grid_search_rf.best_params_)
print('RF cv_result:', grid_search_rf.cv_results_)

pred_best_rf_train = best_rf.predict(X_train_SL)
print('Training MSE of RF best model is: ', mean_squared_error(Y_train_SL, pred_best_rf_train))

# test
pred_best_rf_test = best_rf.predict(X_test_SL)
print('Test MSE of RF best model is: ', mean_squared_error(Y_test_SL, pred_best_rf_test))
print('Test R^2 of RF best model is: ', r2_score(Y_test_SL, pred_best_rf_test))

rf_new = RandomForestRegressor(n_estimators=35, max_depth=11, random_state=0)
rf_new.fit(X_train_SL, Y_train_SL)
rft=rf_new.predict(X_test_SL)
print('Training MSE of rf 35 is: ', mean_squared_error(Y_train_SL, rf_new.predict(X_train_SL)))
print('Test MSE of rf 35 is: ', mean_squared_error(Y_test_SL, rft))
print('Test R^2 of rf 35 is: ', r2_score(Y_test_SL, rft))

# Calling Method
nestimator =  [3, 8, 10, 12, 15, 20, 25, 30, 35,36,37,38,39,40,41,42,43,44,45,47,48,50,53,55,57,60]
maxdepth = [11]
plot_grid_search(grid_search_rf.cv_results_, nestimator, maxdepth, 'N Estimators', 'RF Max depth = ')

# relation
rfp = RandomForestRegressor(n_estimators=35, max_depth=11, random_state=0)
tree_regression_visualization(xp[:,0], yp, rfp,
                              pointlabel='HIV/AIDS vs Life expectancy',
                              curvelabel='Random Forest regression',
                              Xlabel='HIV/AIDS',
                              Ylabel='Life expectancy',
                              Title='Random Forest Regression for one salient feature (HIV/AIDS)'
                              )
tree_regression_visualization(xp[:,1], yp, rfp,
                              pointlabel='Adult Morality vs Life expectancy',
                              curvelabel='Random Forest regression',
                              Xlabel='Adult Morality',
                              Ylabel='Life expectancy',
                              Title='Random Forest Regression for one salient feature (Adult Morality)'
                              )

# X_grid = np.arange(0, 100, 0.001)
# X_grid = X_grid.reshape((len(X_grid), 1))
# best_rf.fit(df_engineered['Schooling'].to_numpy().reshape(-1,1), df_engineered['Life expectancy'].to_numpy().reshape(-1,1))
#
# plt.figure()
# plt.scatter(X_grid, best_rf.predict(X_grid))
# plt.show()
# # visualization
# X_grid = np.arange(min(X), max(X), 0.01)
# X_grid = X_grid.reshape((len(X_grid), 1))
# plt.scatter(X, y, color = 'red')
# plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
# plt.title('Truth or Bluff (Decision Tree Regression)')
# plt.xlabel('Position level')
# plt.ylabel('Salary')
# plt.show()

# model 6: Adaboost
print('\n')
print('=== Model 6: Adaboost ===')

# best model parameter selection
param_grid_ada = [{'n_estimators': [35],
                   'learning_rate': np.linspace(0.0005,0.3,50).tolist()}]
grid_ada = AdaBoostRegressor(base_estimator=DecisionTreeRegressor(max_depth=11), random_state = 0)
grid_search_ada = GridSearchCV(grid_ada, param_grid_ada, cv=5, verbose=1, return_train_score=True)
grid_search_ada.fit(X_train_SL, Y_train_SL)
best_ada = grid_search_ada.best_estimator_
best_ada.fit(X_train_SL, Y_train_SL)
pred_best_ada = best_ada.predict(X_val_SL)
print('Validation MSE of AdaBoost best model is: ', get_mse(Y_val_SL, pred_best_ada))
print('Validation AdaBoost best parameter is: ', grid_search_ada.best_params_)
print('AdaBoost cv_result:', grid_search_ada.cv_results_)

pred_best_ada_train = best_ada.predict(X_train_SL)
print('Training MSE of AdaBoost best model is: ', mean_squared_error(Y_train_SL, pred_best_ada_train))

# test
pred_best_ada_test = best_ada.predict(X_test_SL)
print('Test MSE of AdaBoost best model is: ', mean_squared_error(Y_test_SL, pred_best_ada_test))
print('Test R^2 of AdaBoost best model is: ', r2_score(Y_test_SL, pred_best_ada_test))

ada_new = AdaBoostRegressor(base_estimator=DecisionTreeRegressor(max_depth=11), n_estimators=35, learning_rate=0.2, random_state=0)
ada_new.fit(X_train_SL, Y_train_SL)
adat = ada_new.predict(X_test_SL)
print('Training MSE of ada 0.2 is: ', mean_squared_error(Y_train_SL, ada_new.predict(X_train_SL)))
print('Test MSE of ada 0.2 is: ', mean_squared_error(Y_test_SL, adat))
print('Test R^2 of ada 0.2 is: ', r2_score(Y_test_SL, adat))

# Calling Method
nestimator = [35]
learningrate = np.linspace(0.0005,0.3,50).tolist()
plot_grid_search(grid_search_ada.cv_results_, learningrate, nestimator,  'learning rate', 'N Estimators = ')


adap = AdaBoostRegressor(base_estimator=DecisionTreeRegressor(max_depth=11), n_estimators=35, learning_rate=0.2, random_state=0)
tree_regression_visualization(xp[:,0], yp, adap,
                              pointlabel='HIV/AIDS vs Life expectancy',
                              curvelabel='AdaBoost regression',
                              Xlabel='HIV/AIDS',
                              Ylabel='Life expectancy',
                              Title='AdaBoostt Regression for one salient feature (HIV/AIDS)'
                              )
tree_regression_visualization(xp[:,1], yp, adap,
                              pointlabel='Adult Morality vs Life expectancy',
                              curvelabel='AdaBoost regression',
                              Xlabel='Adult Morality',
                              Ylabel='Life expectancy',
                              Title='AdaBoost Regression for one salient feature (Adult Morality)'
                              )


# # model 7: gradient descend
# print('\n')
# print('=== Model 7: SGD regression ===')
#
# # best model parameter selection
# param_grid_sgd = [{'tol': [0.00000001, 0.0000005, 0.000001, 0.00005, 0.0001, 0.001, 0.005],
#                    'alpha': [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.001, 0.05, 0.1, 0.2, 0.5]
#                    }]
# grid_sgd = SGDRegressor(max_iter=100000000)
# grid_search_sgd = GridSearchCV(grid_sgd, param_grid_sgd, cv=5, verbose=1)
# grid_search_sgd.fit(X_train_SL, Y_train_SL)
# best_sgd = grid_search_sgd.best_estimator_
# best_sgd.fit(X_train_SL, Y_train_SL)
# pred_best_sgd = best_sgd.predict(X_val_SL)
#
# print('Validation MSE of SGD best model is: ', get_mse(Y_val_SL, pred_best_sgd))
# print('Validation SGD best parameter is: ', grid_search_sgd.best_params_)
# # print('sgd cv_result:', grid_search_sgd.cv_results_)
#
# pred_best_sgd_train = best_sgd.predict(X_train_SL)
# print('Training MSE of SGD best model is: ', mean_squared_error(Y_train_SL, pred_best_sgd_train))
#
# # test
# pred_best_sgd_test = best_sgd.predict(X_test_SL)
# print('Test MSE of SGD best model is: ', mean_squared_error(Y_test_SL, pred_best_sgd_test))
# print('Test R^2 of SGD best model is: ', r2_score(Y_test_SL, pred_best_sgd_test))

# model 8: Gaussian Mixture Model for regression
# from GMM_GMR import GMM_GMR
# gmm = GMM_GMR()
# gmm.fit(X_train_SL)
# gmm.predict(Y_train_SL)
# val_pred_mlp = gmm.predict(Xval)
# print('val MSE of mlp = ', mean_squared_error(Yval, val_pred_mlp))

# model 9: KNN regressor
print('\n')
print('=== Model 9: KNN Regressor ===')

# best model parameter selection
param_grid_knn = [{'n_neighbors': [2,3,4,5,6,7,8,9,10,11,12,13,14,15],
                   'p': [1,2,3]}]
grid_knn = KNeighborsRegressor()
grid_search_knn = GridSearchCV(grid_knn, param_grid_knn, cv=5, verbose=1, return_train_score=True)
grid_search_knn.fit(X_train_SL, Y_train_SL)
best_knn = grid_search_knn.best_estimator_
best_knn.fit(X_train_SL, Y_train_SL)
pred_best_knn = best_knn.predict(X_val_SL)

print('Validation MSE of KNN best model is: ', get_mse(Y_val_SL, pred_best_knn))
print('Validation KNN best parameter is: ', grid_search_knn.best_params_)
print('knn cv_result:', grid_search_knn.cv_results_)

pred_best_knn_train = best_knn.predict(X_train_SL)
print('Training MSE of KNN best model is: ', mean_squared_error(Y_train_SL, pred_best_knn_train))

# test
pred_best_knn_test = best_knn.predict(X_test_SL)
print('Test MSE of KNN best model is: ', mean_squared_error(Y_test_SL, pred_best_knn_test))
print('Test R^2 of KNN best model is: ', r2_score(Y_test_SL, pred_best_knn_test))

knn_new = KNeighborsRegressor(n_neighbors=8, p=1)
knn_new.fit(X_train_SL, Y_train_SL)
knnt=knn_new.predict(X_test_SL)
print('Training MSE of KNN 8 is: ', mean_squared_error(Y_train_SL, knn_new.predict(X_train_SL)))
print('Test MSE of KNN 8 is: ', mean_squared_error(Y_test_SL, knnt))
print('Test R^2 of KNN 8 is: ', r2_score(Y_test_SL, knnt))
# Calling Method
nneighbors = [2,3,4,5,6,7,8,9,10,11,12,13,14,15]
ppp = [1,2,3]
plot_grid_search(grid_search_knn.cv_results_, nneighbors, ppp, 'N neighbors', 'p value = ')

knnp = KNeighborsRegressor(n_neighbors=8, p=1)
tree_regression_visualization(xp[:,0], yp, knnp,
                              pointlabel='HIV/AIDS vs Life expectancy',
                              curvelabel='KNN regression',
                              Xlabel='HIV/AIDS',
                              Ylabel='Life expectancy',
                              Title='KNN Regression for one salient feature (HIV/AIDS)'
                              )
tree_regression_visualization(xp[:,1], yp, knnp,
                              pointlabel='Adult Morality vs Life expectancy',
                              curvelabel='KNN regression',
                              Xlabel='Adult Morality',
                              Ylabel='Life expectancy',
                              Title='KNN Regression for one salient feature (Adult Morality)'
                              )

# save data set and models
np.save('data/X_test_SL.npy', X_test_SL)
np.save('data/Y_test_SL.npy', Y_test_SL)

# # save the SL models
#
# filename1 = 'best_lasso.sav'
# pickle.dump(best_lasso, open(filename1, 'wb'))
# filename2 = 'best_ridge.sav'
# pickle.dump(best_ridge, open(filename2, 'wb'))
# filename3 = 'best_cart.sav'
# pickle.dump(cart_new, open(filename3, 'wb'))
# filename4 = 'best_rf.sav'
# pickle.dump(rf_new, open(filename4, 'wb'))
# filename5 = 'best_ada.sav'
# pickle.dump(ada_new, open(filename5, 'wb'))
# filename6 = 'best_knn.sav'
# pickle.dump(knn_new, open(filename6, 'wb'))
# filename7 = 'best_svr.sav'
# pickle.dump(best_svr, open(filename7, 'wb'))


print('================================ Transfer Learning Part ================================')

# # TL classification problem
# # get TL dataset
# from TrAdaBoost1 import TrAdaBoost
# from sklearn.ensemble import AdaBoostClassifier
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.metrics import accuracy_score
# Xs, ys, Xt, yt, Xt_test, yt_test = get_TR_dataset(df_engineered)
# base_estimator = DecisionTreeClassifier(max_depth=1)
# clf = TrAdaBoost(N=3, base_estimator=base_estimator, score=accuracy_score)
# clf.fit(Xs, Xt, ys, yt)
# ys_pred = clf.predict(Xs)
# yt_pred = clf.predict(Xt)
# yt_test_pred = clf.predict(Xt_test)
# print(ys_pred)
# print(yt_pred)
# print(yt_test_pred)
# print('TrAdaBoost')
# print('train acc of Xs:', accuracy_score(ys, ys_pred))
# print('target acc:', accuracy_score(yt, yt_pred))
# print('target_test acc:', accuracy_score(yt_test, yt_test_pred))
# print('==='*20)
#
# # compare
# from sklearn.ensemble import AdaBoostClassifier
# baseline = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=1), n_estimators=3)
# baseline.fit(Xs, ys)
# print('AdaBoost')
# print('train acc of Xs:', accuracy_score(ys,baseline.predict(Xs)))
# print('target acc:', accuracy_score(yt,baseline.predict(Xt)))
# print('target_test acc:', accuracy_score(yt_test,baseline.predict(Xt_test)))
# print('==='*20)

# get TL dataset for regression problem
'''
Thanks user jay15summer https://github.com/jay15summer/Two-stage-TrAdaboost.R2
for providing code of two-stage TrAdaBoost.R2 according to
the paper "Boosting for Regression Transfer (ICML 2010)"
'''
from sklearn.ensemble import AdaBoostRegressor
from TwoStageTrAdaBoostR2 import TwoStageTrAdaBoostR2  # import the two-stage algorithm
Xs, ys, Xt, yt, Xt_val, yt_val, Xt_test, yt_test = get_TL_dataset(df_engineered)

# save data set and models
np.save('data/X_test_TL.npy', Xt_test)
np.save('data/Y_test_TL.npy', yt_test)

print('xs',Xs)
print('xt',Xt)
print('xtt',Xt_test)
TL_data = np.concatenate((Xs, Xt), axis=0)
TL_label = np.concatenate((ys, yt), axis=0)
print(TL_data.shape)
print(TL_label.shape)
# Apply Two stage TrAdaBoost.R2 for TL regression problem
sample_size = [len(Xs), len(Xt)]  # 2426 32
ntr = [3,4,5,8,10,13,15,20,25,30]
dtr = [1,2,3]
temp = 0
mse_val_d = []
for d in range(len(dtr)):
    print('d=', dtr[d])
    mse_val_tr = []
    for n in range(len(ntr)):
        print('ntr=',ntr[n])
        rgs = TwoStageTrAdaBoostR2(base_estimator=DecisionTreeRegressor(max_depth=dtr[d], random_state=0), n_estimators=ntr[n], sample_size=sample_size)
        rgs.fit(TL_data, TL_label)
        pred_val_tr = rgs.predict(Xt_val)
        mse_val_tr.append(mean_squared_error(pred_val_tr, yt_val))
    mse_val_d.append(mse_val_tr)
mse_val_d = np.array(mse_val_d).reshape(len(dtr), len(ntr))
print('mse  val d,n', mse_val_d)
min_dn = np.where(mse_val_d == np.min(mse_val_d))
print(min_dn)
best_d_tr = dtr[min_dn[0][0]]
best_n_tr = ntr[min_dn[1][0]]
print('best max depth d: ', best_d_tr)
print('best n estimator: ', best_n_tr)
print('best validation mse is', np.min(mse_val_d))
best_tr = TwoStageTrAdaBoostR2(base_estimator=DecisionTreeRegressor(max_depth=best_d_tr,random_state=0), n_estimators=best_n_tr, sample_size=sample_size)
best_tr.fit(TL_data, TL_label)
ys_pred = best_tr.predict(Xs)
yt_pred = best_tr.predict(Xt)
yt_test_pred = best_tr.predict(Xt_test)
print('TrAdaBoost for Regression')
print('train MSE of Xs for best model:', mean_squared_error(ys, ys_pred))
print('target train MSE for best model:', mean_squared_error(yt, yt_pred))
print('target test MSE for best model:', mean_squared_error(yt_test, yt_test_pred))
print('==='*20)

# comparison with AdaBoost Regressor
baseline = AdaBoostRegressor(base_estimator=DecisionTreeRegressor(max_depth=best_d_tr, random_state=0), n_estimators=best_n_tr)
baseline.fit(Xs, ys)
print('AdaBoost for Regression')
ys_pred_base = baseline.predict(Xs)
yt_pred_base = baseline.predict(Xt)
yt_test_pred_base = baseline.predict(Xt_test)
# print(ys_pred_base)
# print(yt_pred_base)
# print(yt_test_pred_base)
print('train acc of Xs:', mean_squared_error(ys, ys_pred_base))
print('target train MSE:', mean_squared_error(yt, yt_pred_base))
print('target test MSE:', mean_squared_error(yt_test, yt_test_pred_base))
print('\n')


trp = TwoStageTrAdaBoostR2(base_estimator=DecisionTreeRegressor(max_depth=4), n_estimators=5, sample_size=sample_size)
TL_visualization(TL_data[:, 15],TL_label, Xt_test[:, 15], yt_test, trp,
                 pointlabels='training data (source+target)',
                 pointlabelt='target domain test set',
              curvelabel='Two-stage TrAdaBoost.R2 regression',
              Xlabel='HIV/AIDS',
              Ylabel='Life expectancy',
              Title='Two-stage TrAdaBoost.R2 regression for one salient feature (HIV/AIDS)')
TL_visualization(TL_data[:, 4],TL_label, Xt_test[:, 4], yt_test, trp,
                 pointlabels='training data (source+target)',
                 pointlabelt='target domain test set',
                 curvelabel='Two-stage TrAdaBoost.R2 regression',
                 Xlabel='Adult Morality',
                 Ylabel='Life expectancy',
                 Title='Two-stage TrAdaBoost.R2 regression for one salient feature (Adult Morality)')

# filename8 = 'best_TL.sav'
# pickle.dump(best_tr, open(filename8, 'wb'))
# filename9 = 'best_TL_baseline.sav'
# pickle.dump(baseline, open(filename9, 'wb'))


print('================================ Semi-Supervised Learning Part ================================')

# Create SSL dataset (22 features, based on st original Xtrain, randomly add NaN in Y)
# ss_ssl = StandardScaler()
# X_SSL, Y_SSL, N_U = get_SSL_dataset(df_engineered, u_percent=0.1)
# print('X_SSL', X_SSL)
# print('Y_SSL', Y_SSL.tolist())
# print('y nan idx', np.argwhere(np.isnan(np.array(Y_SSL.tolist()))))
# X_SSL_st = ss_ssl.fit_transform(X_SSL, Y_SSL)
# Y_SSL_st = Y_SSL
# print('X_SSL_st', X_SSL_st)
# print('Y_SSL_st', Y_SSL_st)


# model 1: Co-training SSL Regression
'''
thanks for the code https://github.com/nealjean/coreg
by user nealjean for Co-training regressor(COREG)
based on the paper [Semi-Supervised Regression with Co-Training] 
(http://dl.acm.org/citation.cfm?id=1642439) by Zhou and Li (IJCAI, 2005).
I modified the output part
'''

print('\n')
print('=== model 1: Co-training SSL Regression ===')

from coreg import Coreg
# Create SSL dataset for Coreg (22 features, beased on st original Xtrain, no NaN value in Y)
x_l_ssl, y_l_ssl, X_val_ssl, y_val_ssl, X_test_ssl, y_test_ssl, X_u_ssl, y_u_ssl = get_SSL_dataset4LP(df_engineered)
ss_ssl = StandardScaler()
X_l_ssl_st = ss_ssl.fit_transform(x_l_ssl, y_l_ssl)
Y_l_ssl_st = y_l_ssl.reshape(-1, 1)
X_val_ssl_st = ss_ssl.transform(X_val_ssl)
Y_val_ssl_st = y_val_ssl.reshape(-1,1)
X_test_ssl_st = ss_ssl.transform(X_test_ssl)
Y_test_ssl_st = y_test_ssl.reshape(-1,1)
X_u_ssl_st = ss_ssl.transform(X_u_ssl)
Y_u_ssl_st = y_u_ssl.reshape(-1,1)


# save data set and models
np.save('data/X_test_SSL.npy', X_test_ssl_st)
np.save('data/Y_test_SSL.npy', Y_test_ssl_st)
np.save('data/X_l_SSL.npy', X_l_ssl_st)
np.save('data/Y_l_SSL.npy', Y_l_ssl_st)
np.save('data/X_u_SSL.npy', X_u_ssl_st)


print('shape X_train_ssl_st', X_l_ssl_st.shape)
print('shape X_val_ssl_st', X_val_ssl_st.shape)
print('shape X_test_ssl_st', X_test_ssl_st.shape)
print('shape X_u_ssl_st', X_u_ssl_st.shape)
X_coreg = np.vstack((X_l_ssl_st, X_test_ssl_st, X_u_ssl_st))
Y_coreg = np.vstack((Y_l_ssl_st, Y_test_ssl_st, Y_u_ssl_st))
print('shape X_coreg', X_coreg.shape)
# print('X coreg', X_coreg)
print('shape Y_coreg', Y_coreg.shape)
print('\n')

# baseline KNN

# knn_baseline = KNeighborsRegressor()
# knn_baseline.fit(X_l_ssl_st, Y_l_ssl_st)
# print('knn() baseline test MSE', mean_squared_error(Y_test_ssl_st, knn_baseline.predict(X_test_ssl_st)))


# Model selection for best knn param for both baseline and co-training SSL regression (for comparison)
param_grid_baseline_ssl = [{'n_neighbors': [4,5,6,7,8,9,10],
                            'p': [1,2,3,4]}]
grid_baseline_ssl = KNeighborsRegressor()
grid_search_baseline_ssl = GridSearchCV(grid_baseline_ssl, param_grid_baseline_ssl, cv=5, verbose=1)
grid_search_baseline_ssl.fit(X_l_ssl_st, Y_l_ssl_st)
best_baseline_ssl = grid_search_baseline_ssl.best_estimator_
best_baseline_ssl.fit(X_l_ssl_st, Y_l_ssl_st)
pred_best_baseline_ssl = best_baseline_ssl.predict(X_val_ssl_st)
print('baseline KNN for Regression')
print('Validation MSE of SSL baseline KNN best model is: ', get_mse(Y_val_ssl_st, pred_best_baseline_ssl))
print('Validation SSL baseline KNN best parameter is: ', grid_search_baseline_ssl.best_params_)
print(grid_search_baseline_ssl.cv_results_)
#  baseline knn training MSE
pred_best_baseline_train_ssl = best_baseline_ssl.predict(X_l_ssl_st)
print('Training MSE of KNN baseline best model is: ', mean_squared_error(Y_l_ssl_st, pred_best_baseline_train_ssl))

# baseline knn test MSE
pred_best_baseline_test_ssl = best_baseline_ssl.predict(X_test_ssl_st)
print('Test MSE of KNN baseline best model is: ', mean_squared_error(Y_test_ssl_st, pred_best_baseline_test_ssl))

# # save model
# filename10 = 'best_baseline_ssl.sav'
# pickle.dump(best_baseline_ssl, open(filename10, 'wb'))


# Co-training SSL Regression Part
print('Co-training KNN for SSL Regression')
verbose = True
random_state = -1
num_labeled = 1464  # (train)
num_test = 549
cr = Coreg(k1=4, k2=4, p1=1, p2=2, max_iters=100, pool_size=80)
cr.add_data(X_coreg, Y_coreg)

# Run training
cr.run_trials(num_train=732, trials=1, verbose=True)
knn1_train_mse, knn2_train_mse, knn_co_train_mse, knnco_best_train_mse, knn1_test_mse, knn2_test_mse, knn_co_test_mse, knnco_best_test_mse, coregmodel1, coregmodel2 = cr.plot_mse()
print('knnco_best_train_mse:', knnco_best_train_mse)
print('knn1_train_mse: \n', np.array(knn1_train_mse))
print('knn2_train_mse: \n', np.array(knn2_train_mse))
print('knn_co_train_mse: \n', np.array(knn_co_train_mse))

# test
print('knnco_best_test_mse:', knnco_best_test_mse)
print('knn1_test_mse: \n', np.array(knn1_test_mse))
print('knn2_test_mse: \n', np.array(knn2_test_mse))
print('knn_co_test_mse: \n', np.array(knn_co_test_mse))

# # save models
# filename11 = 'coregmodel1.sav'
# pickle.dump(coregmodel1, open(filename11, 'wb'))
# filename12 = 'coregmodel2.sav'
# pickle.dump(coregmodel2, open(filename12, 'wb'))

print('coreg1', mean_squared_error(Y_test_ssl_st, coregmodel1.predict(X_test_ssl_st)))
print('coreg2', mean_squared_error(Y_test_ssl_st, coregmodel2.predict(X_test_ssl_st)))
coreg_model_pred = (coregmodel1.predict(X_test_ssl_st) + coregmodel2.predict(X_test_ssl_st))/2
print('cotrain:',mean_squared_error(Y_test_ssl_st, coreg_model_pred))


# visualization
crp1= Coreg(k1=4, k2=4, p1=1, p2=2, max_iters=100, pool_size=80)
crp1.add_data(X_coreg[:,15].reshape(-1,1), Y_coreg)
crp1.run_trials(num_train=732, trials=1, verbose=True)
crp1.visualization('HIV/AIDS vs Life expectancy', 'Co-training Regression', 'HIV/AIDS',
                  'Life expectancy', 'Co-training Regression for one salient feature (HIV/AIDS)')
crp2= Coreg(k1=4, k2=4, p1=1, p2=2, max_iters=100, pool_size=80)
crp2.add_data(X_coreg[:,4].reshape(-1,1), Y_coreg)
crp2.run_trials(num_train=732, trials=1, verbose=True)
crp2.visualization('Adult Morality vs Life expectancy', 'Co-training Regression', 'Adult Morality',
                  'Life expectancy', 'Co-training Regression for one salient feature (Adult Morality)')


# model 2: label propagation by KNN

def label_propagation_regression(X_l, y_l, X_u, X_val, y_val, nn, pi, sigma_2):
    '''
    Thanks for user ermongroup for providing code of Label Propagation
    https://github.com/ermongroup/ssdkl/tree/master/labelprop_and_meanteacher
    '''
    # concatenate all the X's and y's
    X_all = np.concatenate([X_l, X_u, X_val], axis=0)
    knn = KNeighborsRegressor(n_neighbors=nn, p=pi,weights='distance')
    knn.fit(X_l, y_l)
    y_u = knn.predict(X_u)
    y_val_pred = knn.predict(X_val)
    y_all = np.concatenate([y_l, y_u, y_val_pred])

    # compute the kernel
    print("Compute kernel")
    T = np.exp(-cdist(X_all, X_all, metric='sqeuclidean') / sigma_2)
    print('T')
    # row normalize the kernel
    T /= np.sum(T, axis=1)[:, np.newaxis]

    print("kernel done")
    delta = np.inf
    tol = 3e-6
    i = 0
    while delta > tol:
        y_all_new = T.dot(y_all)
        # clamp the labels known
        y_all_new[:X_l.shape[0]] = y_l
        delta = np.mean(y_all_new - y_all)
        y_all = y_all_new
        i += 1
        val_loss = np.mean(np.square(y_all[-X_val.shape[0]:] - y_val))
        if i % 10 == 0:
            print("Iter {}: delta={}, val_loss={}".format(i, delta, val_loss))
        if i > 500:
            break
    # return final val loss (MSE)
    return val_loss, y_val_pred


test_mse = []
best_val_mse = []
mse_sigma_nn = []
mse_nntrain = []
mse_sigma_nntrain = []
# search over sigma_2 by cross validation
print('begin validate===========')
sigma = np.linspace(0.01, 2.0, 40)
print(sigma)
nn = [4]
for i in range(len(nn)):
    mse_nn = []
    for j in range(len(sigma)):
        temp,_ = label_propagation_regression(X_l_ssl_st, Y_l_ssl_st, X_u_ssl_st, X_val_ssl_st, Y_val_ssl_st, nn[i], 1, sigma[j])
        temp_train,_ = label_propagation_regression(X_l_ssl_st, Y_l_ssl_st, X_u_ssl_st, X_l_ssl_st, Y_l_ssl_st, nn[i], 1, sigma[j])
        mse_nn.append(temp)
        mse_nntrain.append(temp_train)
    mse_sigma_nn.append(mse_nn)
    mse_sigma_nntrain.append(mse_nntrain)
mse_sigma_nn = np.array(mse_sigma_nn)
mse_sigma_nntrain = np.array(mse_sigma_nntrain)
print('mse_sigma_nn', mse_sigma_nn)
print('mse_sigma_nntrain', mse_sigma_nntrain)
min_ns = np.where(mse_sigma_nn == np.min(mse_sigma_nn))
print(min_ns)
best_n_ssl = nn[min_ns[0][0]]
best_sigma_ssl = sigma[min_ns[1][0]]
print('best_n_ssl', best_n_ssl)
print('best_sigma_ssl', best_sigma_ssl)
print('best validation MSE of LP is: ', mse_sigma_nn.min())
# xgrid = np.arange(len(sigma))
plt.figure()
plt.plot(sigma, mse_sigma_nntrain[0], label='training MSE')
plt.plot(sigma, mse_sigma_nn[0], label='test MSE')
plt.xlabel('alpha')
plt.ylabel('MSE')
plt.title('Label Propagation alpha vs MSE')
plt.legend()
print('end validate=============')

# test with the best
print('test lp========')
test_loss = label_propagation_regression(X_l_ssl_st, Y_l_ssl_st, X_u_ssl_st, X_test_ssl_st, Y_test_ssl_st, 4, 1, best_sigma_ssl)
print('LP test_mse', test_loss)

x_min_lp1 = np.min(X_l_ssl_st[:,4])
x_max_lp1 = np.max(X_l_ssl_st[:,4])
xrange_lp1 = np.arange(x_min_lp1, x_max_lp1, 1)
x_min_lp2 = np.min(X_l_ssl_st[:,15])
x_max_lp2 = np.max(X_l_ssl_st[:,15])
xrange_lp2 = np.arange(x_min_lp2, x_max_lp2, 1)
_,b1=label_propagation_regression(X_l_ssl_st[:,4].reshape(-1,1), Y_l_ssl_st, X_u_ssl_st[:,4].reshape(-1,1), xrange_lp1.reshape(-1,1), xrange_lp1.reshape(-1,1), 4, 1, 0.3161538461538462)
_,b2=label_propagation_regression(X_l_ssl_st[:,15].reshape(-1,1), Y_l_ssl_st, X_u_ssl_st[:,15].reshape(-1,1), xrange_lp2.reshape(-1,1), xrange_lp2.reshape(-1,1), 4, 1, 0.3161538461538462)

LPvisualization(X_l_ssl_st[:,4].reshape(-1,1), Y_l_ssl_st, b1, 'HIV/ADISvs Life expectancy', 'Label Propagation Regression', 'HIV/AIDS',
                  'Life expectancy', 'Label Propagation Regression for one salient feature (HIV/AIDS)')

LPvisualization(X_l_ssl_st[:,15].reshape(-1,1), Y_l_ssl_st, b2, 'Adult Morality vs Life expectancy', 'Label Propagation Regression', 'Adult Morality',
                  'Life expectancy', 'Label Propagation Regression for one salient feature (Adult Morality)')


# model 3: Expectation Maximum algorithm for SSL regression problem
# import math as mt
# # Function to compute multivariate gaussian
# def gaussian_function(row, mean_val, cov_value, num_of_features):
#     diff_data_mean = np.array(row - mean_val).reshape(1, num_of_features)
#     exp = np.exp(-0.5 * np.dot(np.dot(diff_data_mean, np.linalg.inv(cov_value)), diff_data_mean.T))
#     return (1 / np.sqrt(((2 * mt.pi) ** num_of_features) * np.linalg.det(cov_value))) * exp
#
# # Function to compute log likelihood
# def compute_log_likelihood(data, mean_list, cov_list, lambda_list, k_value, num_of_features):
#     log_sum = 0.0
#
#     # Iterating all data
#     for ii in range(len(data)):
#         inner_sum = 0.0
#
#         # Iterating all k
#         for kk in range(k_value):
#             inner_sum += lambda_list[kk] * gaussian_function(data[ii], mean_list[kk], cov_list[kk], num_of_features)
#
#         log_sum += np.log(inner_sum)
#
#     return log_sum
#
# def gmm_predict(data, mean_value, covar_value, lambda_value, k_value, num_of_features):
#     prediction = []
#
#     # Iterating each data
#     for pos in range(len(data)):
#
#         best_likelihood = None
#         best_cluster = None
#
#         # Iterating each cluster k
#         for k_num in range(k_value):
#
#             # Computing likelihood value
#             likelihood_value = lambda_arr[k_num] * gaussian_function(data[pos], mean_value[k_num], covar_value[k_num], num_of_features)
#
#             # Check if best value
#             if best_likelihood is None or best_likelihood <= likelihood_value:
#                 best_likelihood = likelihood_value
#                 best_cluster = k_num
#
#         # Append to prediction array
#         prediction.append(best_cluster)
#
#     return prediction
#
#
# # K array
# k_array = [12, 18, 24, 36, 42]
#
# # Get GMM objective loss array and compute mean and variance
# gmm_loss_array = []
#
# # GMM Model stores
# gmm_mean_array = []
# gmm_covar_array = []
# gmm_lambda_array = []
#
# feature_length = len(X_train_st[0])
# print(feature_length)
# data_length = len(X_train_st)
# # For each cluster size k
# for k in k_array:
#
#     # For 20 iterations
#     for i in range(2):
#
#         # Initializing mean array
#         mean_arr = np.empty((k, feature_length), dtype=np.float64)
#         for j in range(k):
#             mean_arr[j] = np.array(np.random.choice(np.arange(-3, 4, 1), feature_length)).reshape(1, feature_length)
#
#         # Initializing co-variance matrix
#         cov_matrix_arr = np.empty((k, feature_length, feature_length))
#         for j in range(k):
#             cov_matrix_arr[j] = np.identity(n=feature_length, dtype=np.float64)
#
#         # Initializing lambda array
#         lambda_arr = np.empty((k, 1), dtype=np.float64)
#         for j in range(k):
#             lambda_arr[j] = 1/k
#
#         # Initial log likelihood value
#         log_like_val = compute_log_likelihood(X_train_st, mean_arr, cov_matrix_arr, lambda_arr, k, feature_length)
#         iteration_counter = 1
#
#         # Begin EM iterations
#         while True:
#
#             # E Step block
#             q_array = np.empty((data_length, k), dtype=np.float64)
#
#             # Iterating data
#             for x in range(data_length):
#
#                 den_sum = 0.0
#
#                 # Iterating k values
#                 for k_val in range(k):
#                     q_array[x, k_val] = lambda_arr[k_val] * gaussian_function(X_train_st[x], mean_arr[k_val], cov_matrix_arr[k_val], feature_length)
#                     den_sum += q_array[x, k_val]
#
#                 q_array[x] = q_array[x] / den_sum
#
#             # M Step block
#             # Updating mean array
#             for k_val in range(k):
#                 num_total = 0.0
#                 den_total = 0.0
#
#                 for m in range(data_length):
#                     num_total += q_array[m, k_val] * X_train_st[m]
#                     den_total += q_array[m, k_val]
#
#                 mean_arr[k_val] = num_total / den_total
#
#             # Updating covariance array
#             for k_val in range(k):
#                 num_total = 0.0
#                 den_total = 0.0
#
#                 for m in range(data_length):
#                     diff_vector = X_train_st[m] - mean_arr[k_val]
#                     diff_vector = np.array(diff_vector).reshape((1, feature_length))
#                     num_total += q_array[m, k_val] * np.dot(diff_vector.T, diff_vector)
#                     den_total += q_array[m, k_val]
#
#                 cov_matrix_arr[k_val] = num_total / den_total
#                 cov_matrix_arr[k_val] += np.identity(n=feature_length)
#
#             # Updating lambda array
#             for k_val in range(k):
#                 num_total = 0.0
#
#                 for m in range(data_length):
#                     num_total += q_array[m, k_val]
#
#             lambda_arr[k_val] = num_total / data_length
#
#             # Compute log likelihood value
#             prev_log_like_val = log_like_val
#             log_like_val = compute_log_likelihood(X_train_st, mean_arr, cov_matrix_arr, lambda_arr, k, feature_length)
#
#             # Status
#             print("K value:", k, "Iteration:", i, "Counter:", iteration_counter)
#
#             # Increment iteration
#             iteration_counter += 1
#
#             # Checking for convergence
#             if prev_log_like_val >= log_like_val:
#                 gmm_loss_array.append(log_like_val)
#                 gmm_mean_array.append(mean_arr)
#                 gmm_covar_array.append(cov_matrix_arr)
#                 gmm_lambda_array.append(lambda_arr)
#                 break
#
# # Mean and variance of converged log likelihood for each k
# print("GMM objective for k: 12 - Mean:", np.mean(gmm_loss_array[0:2]), "Variance:", np.var(gmm_loss_array[0:2]))
# print("GMM objective for k: 18 - Mean:", np.mean(gmm_loss_array[2:4]), "Variance:", np.var(gmm_loss_array[2:4]))
# print("GMM objective for k: 24 - Mean:", np.mean(gmm_loss_array[4:6]), "Variance:", np.var(gmm_loss_array[4:6]))
# print("GMM objective for k: 36 - Mean:", np.mean(gmm_loss_array[6:8]), "Variance:", np.var(gmm_loss_array[6:8]))
# print("GMM objective for k: 42 - Mean:", np.mean(gmm_loss_array[8:10]), "Variance:", np.var(gmm_loss_array[8:10]))
#
# # Predict clusters with k = 36
# acc = 0.0
# sil_acc = 0.0
# temp_data = np.append(X_train_st, np.array(Y_train_st - 1).reshape((data_length, 1)), axis=1)
# for i in range(2):
#     predict_array = gmm_predict(X_train_st, gmm_mean_array[6+i], gmm_covar_array[6+i], gmm_lambda_array[6+i], 36, feature_length)
#     acc += mean_squared_error(Y_train_st, predict_array)
#     # sil_acc += silhouette_score(temp_data, predict_array, sample_size=20)
#
# print("Average Similarity Measure (Adjusted Rand Index) of the GMM model with k: 36 is", acc/2)
# print("Average Silhouette Coefficient for all samples of the GMM model with k: 36 is", sil_acc/2)


plt.ioff()
plt.show()
