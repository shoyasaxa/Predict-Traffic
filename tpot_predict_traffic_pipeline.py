import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from sklearn.tree import DecisionTreeRegressor
from tpot.builtins import StackingEstimator

# NOTE: Make sure that the class is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1).values
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'].values, random_state=42)

# Score on the training set was:-25732.179911816576
exported_pipeline = make_pipeline(
    StackingEstimator(estimator=RandomForestRegressor(bootstrap=True, max_features=0.3, min_samples_leaf=8, min_samples_split=4, n_estimators=100)),
    StackingEstimator(estimator=ExtraTreesRegressor(bootstrap=False, max_features=0.15000000000000002, min_samples_leaf=5, min_samples_split=9, n_estimators=100)),
    StackingEstimator(estimator=ElasticNetCV(l1_ratio=0.9500000000000001, tol=0.1)),
    DecisionTreeRegressor(max_depth=4, min_samples_leaf=2, min_samples_split=6)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
