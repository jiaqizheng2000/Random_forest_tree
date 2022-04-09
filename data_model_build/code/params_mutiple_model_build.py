# importing standard libraries
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
import joblib
Score=[0.0,0.0,0.0,0.0]
Score_ALL=[]
BASEPATH = 'D:/Contact_Angle_prediction/data_model_build'
params=[]
random_seed = 53


def save_model(model,model_name):
    MODELPATH=os.path.join(BASEPATH,'MODELS','42_features_used_All',str(random_seed))
    # MODELPATH = os.path.join(BASEPATH, 'MODELS', '8_features_used_All', str(random_seed))
    # MODELPATH = os.path.join(BASEPATH, 'MODELS', '9_features_used_All', str(random_seed))
    if not os.path.exists(MODELPATH):
        os.mkdir(MODELPATH)

    savepath=os.path.join(MODELPATH,model_name +'.pkl')
    joblib.dump(model,savepath)

def load_model(model_name,num):
    MODELPATH=os.path.join(BASEPATH,'MODELS','42_features_used_All')
    # MODELPATH=os.path.join(BASEPATH,'MODELS','42_features_used_All')
    # MODELPATH=os.path.join(BASEPATH,'MODELS','42_features_used_All')
    loadpath=os.path.join(MODELPATH,str(num),model_name +'.pkl')
    model=joblib.load(loadpath)
    return model

# def random_tree_model(input_data):
def random_tree_model():
    random_forest_seed = np.random.randint(low=1, high=230)
    #
    # Search optimal hyperparameter
    n_estimators_range = [int(x) for x in np.linspace(start=100, stop=1000, num=10)]
    max_features_range = ['auto', 'sqrt']
    max_depth_range = [int(x) for x in np.linspace(1, 200, num=40)]
    min_samples_split_range = [2, 5, 10]
    min_samples_leaf_range = [1, 2, 4, 8,16]
    bootstrap_range = [True, False]

    random_forest_hp_range = {'n_estimators': n_estimators_range,
                              'max_features': max_features_range,
                              'max_depth': max_depth_range,
                              'min_samples_split': min_samples_split_range,
                              'min_samples_leaf': min_samples_leaf_range,
                               'bootstrap':bootstrap_range
                              }

    datapath=os.path.join(BASEPATH,'data','metal_ceramic_data_all_with_A_T.csv')
    #datapath = os.path.join(BASEPATH, 'data', 'metal_ceramic_data_all_with_A_T_reduced_MIT.csv')
    #datapath = os.path.join(BASEPATH, 'data', 'metal_ceramic_data_all_with_A_T_reduced.csv')
    df = pd.read_csv(datapath)  # read file
    x = df.drop(columns=['Wetting angle', 'Metal', 'Substrate'])  # x as predictor,y as result
    # x = df(columns=['Me_MagpieData mean NdValence', 'Me_MagpieData mean CovalentRadius', 'Me_MagpieData mean GSmagmom', 'Me_MagpieData mean Electronegativity', 'Testing temperature (K)', 'Me_MagpieData mean GSbandgap', 'Ce_MagpieData mean NUnfilled', 'Me_MagpieData mean SpaceGroupNumber', 'Ce_MagpieData mean MeltingT'])
    y = df["Wetting angle"]

    scalar = StandardScaler()
    x_scaled = scalar.fit_transform(x)

    # Splitting the dataset for train and test
    x_train, x_validation_test, y_train, y_validation_test = train_test_split(x_scaled, y, test_size=0.20, random_state=random_seed)
    x_validation,x_test,y_validation,y_test  = train_test_split(x_validation_test,y_validation_test,test_size=0.5,random_state=random_seed)

    random_forest_model_test_base = RandomForestRegressor()
    random_forest_model_test_base.fit(x_train, y_train)
    print("Random forest Score of train data= %.3f" % random_forest_model_test_base.score(x_train,y_train))
    print("Random forest base Score of validation&test data = %.3f" % random_forest_model_test_base.score(x_validation_test,y_validation_test))
    Score[0]=random_forest_model_test_base.score(x_train,y_train)
    Score[1]=random_forest_model_test_base.score(x_validation_test,y_validation_test)

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
    Score[2]=random_forest_model_test_random.score(x_validation,y_validation)
    best_hp_now = random_forest_model_test_random.best_params_
    print(best_hp_now)

    #test data
    score=random_forest_model_test_random.score(x_test,y_test)
    Score[3]=random_forest_model_test_random.score(x_test,y_test)
    print("Random forest Score of test data = %.3f"%score)

    #save best parmas and model
    params.append(best_hp_now)
    save_model(random_forest_model_test_random,'RTF_%.3f_%.3f_%.3f_%.3f_%d'%(Score[0],Score[1],Score[2],Score[3],random_seed))
    SCORE= {"Train_80%": Score[0], "Valid_test_%20": Score[1], "Valid_%10": Score[2], "Test_%10": Score[3]}
    Score_ALL.append(SCORE)

    #totally new data as test data
    # data_scaled = scalar.fit_transform(input_data)
    # y_pre = random_forest_model_test_random.predict(data_scaled)
    # global prediction
    # prediction=pandas.DataFrame(columns=['predicted_angle'],data=y_pre)

if __name__=="__main__":
    # prediction=pd.DataFrame()
    # data=pd.read_csv("metal_ceramic_data_all_with_A_T.csv")
    # # data=data[900:]
    # data.reset_index(drop=True,inplace=True)
    # x_data=data.drop(columns=['Wetting angle', 'Metal', 'Substrate'])
    # random_tree_model(x_data)
    # # data=pd.concat([data,prediction],axis=1)
    # # data.to_csv("prediction_result.csv")
    for i in range(10):
        random_tree_model()

    # save params to csv file
    paramters=pd.DataFrame(params)
    params_path=os.path.join(BASEPATH,'PARAMS','42_features_used_All','params_%d.csv'%random_seed)
    # params_path=os.path.join(BASEPATH,'PARAMS','8_features_used_All','params.csv')
    # params_path=os.path.join(BASEPATH,'PARAMS','9_features_used_All','params.csv')
    paramters.to_csv(params_path)

    #save results to csv file
    results=pd.DataFrame(Score_ALL)
    results_path=os.path.join(BASEPATH,'RESULTS','42_features_used_All','results_%d.csv'%random_seed)
    # results_path=os.path.join(BASEPATH,'RESULTS','8_features_used_All','results_%d.csv'%random_seed)
    # results_path=os.path.join(BASEPATH,'RESULTS','9_features_used_All','results_%d.csv'%random_seed)
    results.to_csv(results_path)