# importing standard libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import export_graphviz

pd.set_option('display.max_rows',500)
pd.set_option('display.max_columns',100)
pd.set_option('display.width',1000)

if __name__=="__main__":
    df=pd.read_csv("housing.csv") #read file
    df = df.drop(["id", "url", "region_url", "image_url", "description"], axis=1) #drop unuseful data
    df = df.drop(["state"], axis=1)
    print(df.head())
   #
   #  df['laundry_options'] = df['laundry_options'].fillna(df['laundry_options'].mode()[0]) #impute the null values with proper values of central tendencies
   #  df['parking_options'] = df['parking_options'].fillna(df['parking_options'].mode()[0])
   #  df['lat'] = df['lat'].fillna(df['lat'].mean())
   #  df['long'] = df['long'].fillna(df['long'].mean())
   #  #print(df.parking_options.values_count())
   #
   #  outlier1 = ((df["beds"] > 4) | (df["baths"] > 4)) #delete unuseful data
   #  df = df[~outlier1]
   # # print("There is {} outlier".format(df[outlier1]["beds"].count()))
   #  outlier2 = ((df["sqfeet"] < 120) | (df["sqfeet"] > 5000) | (df["price"] < 100) | (df["price"] > 10000))
   #  df = df[~outlier2]
   #  df = df.drop(["cats_allowed"], axis=1)
   #  df.rename(columns={'dogs_allowed': 'pets_allowed'}, inplace=True)
   #  df["baths"] = df["baths"].astype("int")
   #
   #  sns.countplot(x="type",data=df)
   #  fig = plt.gcf()
   #  fig.set_size_inches(15, 10)
   #  plt.title('Which type of house is more')
   #  plt.show()
   #
   #  le = LabelEncoder()#Label Encoding the categorical string values
   #  db = df
   #  db["region"] = le.fit_transform(df["region"])
   #  db["type"] = le.fit_transform(df["type"])
   #  db["laundry_options"] = le.fit_transform(df["laundry_options"])
   #  db["parking_options"] = le.fit_transform(df["parking_options"])
   #  #print(db.head())
   #
   #  x = db.drop(columns=["price"])#x as predictor,y as result
   #  feature_list = list(x.columns)
   #  y = db["price"]
   #
   #  plt.figure(figsize=(20, 30), facecolor='white')
   #  plotnumber = 1
   #
   #  for column in x:
   #      if plotnumber <= 16:
   #          ax = plt.subplot(4, 4, plotnumber)
   #          plt.scatter(x[column], y)
   #          plt.xlabel(column, fontsize=20)
   #          plt.ylabel('Price', fontsize=20)
   #      plotnumber += 1
   #  plt.tight_layout()
   #  plt.show()
   #
   #  scalar = StandardScaler()
   #  x_scaled = scalar.fit_transform(x)
   #
   #  vif = pd.DataFrame() # variance_inflation_factor to measure how much the variance of
   #                      # an estimated regression cofficient is increased because of collinerarity
   #  vif["VIF"] = [variance_inflation_factor(x_scaled, i) for i in range(x_scaled.shape[1])]
   #  vif["Features"] = x.columns
   #
   #  corrl = db.corr() # plot heat map
   #  plt.figure(figsize=(20, 20))
   #  sns.heatmap(corrl, cbar=True, square=True, fmt='.1f', annot=True, annot_kws={'size': 12}, cmap='twilight_shifted_r')
   #  plt.show()
   #
   #  #Splitting the dataset for train and test
   #  x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.30, random_state=470)
   #  train_X_column_name = list(x.columns)
   #
   #  #Importing the models for training the dataset
   #  dtr = DecisionTreeRegressor()
   #  #ran = RandomForestRegressor(n_estimators=90)
   #  ran=RandomForestRegressor(n_estimators=10, max_depth = 3, random_state=42)
   #  lin = LinearRegression()
   #
   #  models = {"Decision tree": dtr,
   #            "Random forest": ran,
   #            "Linear Regression": lin}
   #  scores = {}
   #
   #  for key, value in models.items():
   #      model = value
   #      model.fit(x_train, y_train)
   #      scores[key] = model.score(x_test, y_test)
   #
   #  scores_frame = pd.DataFrame(scores, index=["Accuracy Score"]).T
   #  scores_frame.sort_values(by=["Accuracy Score"], axis=0, ascending=False, inplace=True)
   #  print(scores_frame)
   #
   #  # get a tree in model
   #  tree = ran.estimators_[5]
   #  # output as dot 文件
   #  export_graphviz(tree, out_file='tree.dot', feature_names=feature_list, rounded=True, precision=1)
   #
   #  # Calculate the importance of variables
   #
   #  random_forest_importance = list(tree.feature_importances_)
   #  random_forest_feature_importance = [(feature, round(importance, 8))
   #                                      for feature, importance in zip(train_X_column_name, random_forest_importance)]
   #  random_forest_feature_importance = sorted(random_forest_feature_importance, key=lambda x: x[1], reverse=True)
   #  plt.figure(3)
   #  plt.clf()
   #  importance_plot_x_values = list(range(len(random_forest_importance)))
   #  plt.bar(importance_plot_x_values, random_forest_importance, orientation='vertical')
   #  plt.xticks(importance_plot_x_values, train_X_column_name, rotation='vertical')
   #  plt.xlabel('Variable')
   #  plt.ylabel('Importance')
   #  plt.title('Variable Importances')
   #  plt.tight_layout()
   #  plt.show()
