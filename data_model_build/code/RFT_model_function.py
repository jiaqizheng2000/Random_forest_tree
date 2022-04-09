# importing standard libraries
import pandas
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

def random_tree_model(input_data,numbers_estimators,max_depth_for_RAN):
    # random_seed = 44
    # random_forest_seed = np.random.randint(low=1, high=230)
    #
    # # Search optimal hyperparameter
    # n_estimators_range = [int(x) for x in np.linspace(start=50, stop=3000, num=50)]
    # max_features_range = ['auto', 'sqrt']
    # max_depth_range = [int(x) for x in np.linspace(5, 100, num=5)]
    # max_depth_range.append(None)
    # min_samples_split_range = [2, 5, 10]
    # min_samples_leaf_range = [1, 2, 4, 8]
    # bootstrap_range = [True, False]
    #
    # random_forest_hp_range = {'n_estimators': n_estimators_range,
    #                           'max_features': max_features_range,
    #                           'max_depth': max_depth_range,
    #                           'min_samples_split': min_samples_split_range,
    #                           'min_samples_leaf': min_samples_leaf_range,
    #                            'bootstrap':bootstrap_range
    #                           }

    df = pd.read_csv("metal_ceramic_data_all_with_A_T.csv")  # read file
    x = df.drop(columns=['Wetting angle', 'Metal', 'Substrate'])  # x as predictor,y as result
    # x = df(columns=['Me_MagpieData mean NdValence', 'Me_MagpieData mean CovalentRadius', 'Me_MagpieData mean GSmagmom', 'Me_MagpieData mean Electronegativity', 'Testing temperature (K)', 'Me_MagpieData mean GSbandgap', 'Ce_MagpieData mean NUnfilled', 'Me_MagpieData mean SpaceGroupNumber', 'Ce_MagpieData mean MeltingT'])
    y = df["Wetting angle"]

    scalar = StandardScaler()
    x_scaled = scalar.fit_transform(x)

    # Splitting the dataset for train and test
    x_train, x_validation_test, y_train, y_validation_test = train_test_split(x_scaled, y, test_size=0.20, random_state=470)
    x_validation,x_test,y_validation,y_test  = train_test_split(x_validation_test,y_validation_test,test_size=0.5,random_state=42)

    # Importing the models for training the dataset
    dtr = DecisionTreeRegressor()
    # ran = RandomForestRegressor(n_estimators=90)
    ran = RandomForestRegressor(n_estimators=numbers_estimators, max_depth=max_depth_for_RAN, random_state=42)
    lin = LinearRegression()

    # random_forest_model_test_base = RandomForestRegressor()
    # random_forest_model_test_random = RandomizedSearchCV(estimator=random_forest_model_test_base,
    #                                                      param_distributions=random_forest_hp_range,
    #                                                      n_iter=200,
    #                                                      n_jobs=-1,
    #                                                      cv=3,
    #                                                      verbose=1,
    #                                                      random_state=random_forest_seed
    #                                                      )
    # random_forest_model_test_random.fit(x_validation, y_validation)
    # best_hp_now = random_forest_model_test_random.best_params_
    #
    # # Grid Search
    # random_forest_hp_range_2 = {'n_estimators': [60, 100, 200],
    #                             'max_features': [12, 13],
    #                             'max_depth': [350, 400, 450],
    #                             'min_samples_split': [2, 3]  # Greater than 1
    #                             # 'min_samples_leaf':[1,2]
    #                             # 'bootstrap':bootstrap_range
    #                             }
    # random_forest_model_test_2_base = RandomForestRegressor()
    # random_forest_model_test_2_random = GridSearchCV(estimator=random_forest_model_test_2_base,
    #                                                  param_grid=random_forest_hp_range_2,
    #                                                  cv=3,
    #                                                  verbose=1,
    #                                                  n_jobs=-1)
    # random_forest_model_test_2_random.fit(x_validation, y_validation)

    # random_forest_model_final = random_forest_model_test_random.best_estimator_

    # #test data
    # score=random_forest_model_test_random.score(x_test,y_test)
    # print("Random forest Score = %.3f"%score)


    models = {"Decision tree": dtr,
              "Random forest": ran,
              "Linear Regression": lin}
    scores = {}

    for key, value in models.items():
        model = value
        model.fit(x_train, y_train)
        # noinspection PyUnresolvedReferences
        scores[key] = model.score(x_validation, y_validation)

    scores_frame = pd.DataFrame(scores, index=["Accuracy Score"]).T
    scores_frame.sort_values(by=["Accuracy Score"], axis=0, ascending=False, inplace=True)
    print(scores_frame)

    data_scaled = scalar.fit_transform(input_data)
    y_pre = ran.predict(data_scaled)
    global prediction
    prediction=pandas.DataFrame(columns=['predicted_angle'],data=y_pre)

if __name__=="__main__":
    prediction=pd.DataFrame()
    data=pd.read_csv("metal_ceramic_data_all_with_A_T.csv")
    data=data[900:]
    data.reset_index(drop=True,inplace=True)
    x_data=data.drop(columns=['Wetting angle', 'Metal', 'Substrate'])
    random_tree_model(x_data,300,30)
    data=pd.concat([data,prediction],axis=1)
    data.to_csv("prediction_result.csv")
