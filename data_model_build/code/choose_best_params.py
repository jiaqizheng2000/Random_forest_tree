# importing standard libraries
import pandas
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

def random_tree_model(input_data):
    random_seed = 42
    random_forest_seed = np.random.randint(low=1, high=230)
    #
    # Search optimal hyperparameter
    n_estimators_range = [int(x) for x in np.linspace(start=100, stop=500, num=10)]
    max_features_range = ['auto', 'sqrt']
    max_depth_range = [int(x) for x in np.linspace(5, 100, num=20)]
    max_depth_range.append(None)
    min_samples_split_range = [2, 5, 10]
    min_samples_leaf_range = [1, 2, 4, 8]
    bootstrap_range = [True, False]

    random_forest_hp_range = {'n_estimators': n_estimators_range,
                              'max_features': max_features_range,
                              'max_depth': max_depth_range,
                              'min_samples_split': min_samples_split_range,
                              'min_samples_leaf': min_samples_leaf_range,
                               'bootstrap':bootstrap_range
                              }

    df = pd.read_csv("metal_ceramic_data_all_with_A_T.csv")  # read file
    x = df.drop(columns=['Wetting angle', 'Metal', 'Substrate'])  # x as predictor,y as result
    # x = df(columns=['Me_MagpieData mean NdValence', 'Me_MagpieData mean CovalentRadius', 'Me_MagpieData mean GSmagmom', 'Me_MagpieData mean Electronegativity', 'Testing temperature (K)', 'Me_MagpieData mean GSbandgap', 'Ce_MagpieData mean NUnfilled', 'Me_MagpieData mean SpaceGroupNumber', 'Ce_MagpieData mean MeltingT'])
    y = df["Wetting angle"]

    scalar = StandardScaler()
    x_scaled = scalar.fit_transform(x)

    # Splitting the dataset for train and test
    x_train, x_validation_test, y_train, y_validation_test = train_test_split(x_scaled, y, test_size=0.20, random_state=42)
    x_validation,x_test,y_validation,y_test  = train_test_split(x_validation_test,y_validation_test,test_size=0.5,random_state=42)


    random_forest_model_test_base = RandomForestRegressor()
    random_forest_model_test_base.fit(x_train, y_train)
    print("Random forest Score of train data= %.3f" % random_forest_model_test_base.score(x_train,y_train))
    print("Random forest base Score of validation&test data = %.3f" % random_forest_model_test_base.score(x_validation_test,y_validation_test))

    random_forest_model_test_random = RandomizedSearchCV(estimator=random_forest_model_test_base,
                                                         param_distributions=random_forest_hp_range,
                                                         scoring='r2',
                                                         n_iter=200,
                                                         n_jobs=-1,
                                                         cv=3,
                                                         verbose=1,
                                                         random_state=random_forest_seed
                                                         )
    random_forest_model_test_random.fit(x_validation, y_validation)
    print("Random forest Score of validation data= %.3f"%random_forest_model_test_random.score(x_validation,y_validation))
    best_hp_now = random_forest_model_test_random.best_params_
    print(best_hp_now)

    #test data
    score=random_forest_model_test_random.score(x_test,y_test)
    print("Random forest Score of test data = %.3f"%score)

    data_scaled = scalar.fit_transform(input_data)
    y_pre = random_forest_model_test_random.predict(data_scaled)
    global prediction
    prediction=pandas.DataFrame(columns=['predicted_angle'],data=y_pre)

if __name__=="__main__":
    prediction=pd.DataFrame()
    data=pd.read_csv("metal_ceramic_data_all_with_A_T.csv")
    # data=data[900:]
    data.reset_index(drop=True,inplace=True)
    x_data=data.drop(columns=['Wetting angle', 'Metal', 'Substrate'])
    random_tree_model(x_data)
    # data=pd.concat([data,prediction],axis=1)
    # data.to_csv("prediction_result.csv")
