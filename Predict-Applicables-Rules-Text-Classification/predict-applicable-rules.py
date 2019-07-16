import sklearn
import pandas as pd
from numpy import argmax
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline, FeatureUnion, make_pipeline, make_union
from sklearn.model_selection import GridSearchCV

from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import FrenchStemmer
from nltk.corpus import stopwords

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

import os
from time import gmtime, strftime
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
#import jsonpickle

from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
#from xgboost import XGBRegressor
from mlens.ensemble import SuperLearner

from sklearn.metrics import make_scorer, log_loss

print("sklearn version - ", sklearn.__version__)


# Stemming
stemmer = FrenchStemmer()
analyzer = TfidfVectorizer().build_analyzer()


def stemmed_words(doc):
    return (stemmer.stem(w) for w in analyzer(doc) if w not in stopwords.words('english'))


def stemmed_words2(doc):
    return (stemmer.stem(w) for w in analyzer(doc) if w not in stopwords.words('english'))


########################################################################
# Load Data :-
########################################################################
cwd = os.getcwd()
print("Read csv files")
X_train = pd.read_csv(cwd + "\\data\\X_train.csv")
y_train = pd.read_csv(cwd + "\\data\\y_train.csv")
X_test = pd.read_csv(cwd + "\\data\\X_test.csv")


#print("Data cleaning")
print("Replace missing values (nan's) by the most frequent")
X_train = X_train.fillna(X_train.mode().iloc[0])
X_test = X_test.fillna(X_test.mode().iloc[0])


'''
print("Read list of countries in EU and clean the countries column")
fp = open(cwd + "\\data\\countries_in_EU.txt", "r")
countriesList = fp.readlines()

for i in range(len(countriesList)):
    countriesList[i] = countriesList[i].strip()

countries = ""
for c in countriesList:
    countries += (c + " ")


for i, v in X_train['Compliance Countries'].iteritems():
    if v.strip() == "EU":
        X_train['Compliance Countries'][i] = countries

    else:
        X_train['Compliance Countries'][i] = X_train['Compliance Countries'][i].replace("United Kingdom", "UnitedKingdom")
        X_train['Compliance Countries'][i] = X_train['Compliance Countries'][i].replace("United States", "UnitedStates")
        X_train['Compliance Countries'][i] = X_train['Compliance Countries'][i].replace("United Arab Emirates", "UnitedArabEmirates")


for i, v in X_test['Compliance Countries'].iteritems():
    if v.strip() == "EU":
        X_test['Compliance Countries'][i] = countries

    else:
        X_test['Compliance Countries'][i] = X_test['Compliance Countries'][i].replace("United Kingdom", "UnitedKingdom")
        X_test['Compliance Countries'][i] = X_test['Compliance Countries'][i].replace("United States", "UnitedStates")
        X_test['Compliance Countries'][i] = X_test['Compliance Countries'][i].replace("United Arab Emirates", "UnitedArabEmirates")
'''


fields = ['Title', 'Event Summary', 'Compliance Organisations', 'Compliance Countries', 'Extraterritorial']

print("Unicode encoding")
for i in fields:
    X_train[i] = X_train[i].astype('unicode')
    X_test[i] = X_test[i].astype('unicode')

y_train['weight'] = y_train['weight'].astype(int)


'''
# Serialize dataframes
X_train.to_pickle("X_train")
X_test.to_pickle("X_test")
y_train.to_pickle("y_train")

# Deserialize dataframes
X_train = pd.read_pickle("X_train")
X_test = pd.read_pickle("X_test")
'''


###############################################################################
# Feature Transforms
###############################################################################
print("Feature transform and Training classifier")

titleTfIdfVectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1,3), min_df=0.01, max_df=0.9,
                                       analyzer='word', use_idf=True)

summaryTfIdfVectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1,3), min_df=0.01, max_df=0.9,
                                       analyzer='word', use_idf=True)


def tokenizer(text):
    return text.split("|")


#orgOneHotEncoder = OneHotEncoder(categories='auto', handle_unknown='ignore')
orgCountVect = CountVectorizer(tokenizer=lambda text: tokenizer(text))

countriesCountVect = CountVectorizer()
extraTerritorialOHE = OneHotEncoder(categories='auto', handle_unknown='ignore')


###############################################################################
# Classifier
###############################################################################
# SVM - Set probability to true to use probability based metric like log loss
# Hyper-parameters to be optimised via cross validation

# 1. SVM
svClassifier = SVC(probability=True, cache_size=7000)         # loss = 0.60
parameters = {'clf__kernel': ['linear', 'rbf'], 'clf__C': [0    .1, 1, 10], 'clf__gamma': [0.001]
              #, 'clf__class_weight: [{'A': 10, 'B': 5, 'C': 1}]
              }

# 2. KNN
classifier = KNeighborsClassifier(algorithm='auto')         # 1.0
parameters = {'clf__n_neighbours': [3, 4, 5, 6, 7, 8, 9, 10], 'clf__p': [1, 2]}

#3. NuSVC
classifier = NuSVC(probability=True, cache_size=7000)       # 0.64
parameters = {'clf__kernel': ['linear', 'rbf', 'sigmoid'], 'clf__gamma': [0.001]}

# 4. DTree
classifier = DecisionTreeClassifier()                       # 12.4
parameters = {}

# 5. RandForest
rfClassifier = RandomForestClassifier()                       # 0.639
parameters = {'clf__n_estimators': [10, 50, 100, 200, 500, 1000]}

# 6. AdaBoost
abClassifier = AdaBoostClassifier()
parameters = {'clf__n_estimators': [50, 100]}

# 7. GradientBoosting
gbClassifier = GradientBoostingClassifier()
parameters = {}

# 8. NB (requires dense data)
classifier = GaussianNB()
parameters = {}

# 9. Linear Discriminant (requires dense data)
classifier = LinearDiscriminantAnalysis()
parameters = {}

# 10. Quadratic Discriminant (requires dense data)
classifier = QuadraticDiscriminantAnalysis()
parameters = {}

# 11. SGD                                                   # 0.61, 0.599, 0.583 after Org column cleaning
sgdClassifier = SGDClassifier(loss='log')
parameters = {'clf__loss': ['modified_huber', 'log'], 'clf__class_weight': [{'A': 1, 'B': 5, 'C': 10}]}

# 12. MLP                                                   # 1.14
classifier = MLPClassifier(hidden_layer_sizes=(100, 100, 100), max_iter=500, alpha=0.0001, verbose=10, random_state=21, tol=0.000000001)
parameters = {'clf__solver': ['sgd', 'adam']}

# 13. LinearSVC (No predict_proba method)
classifier = LinearSVC()
parameters = {'clf__C': [0.1, 1, 10]}

# 14. LogistricRegression                                   # 0.5525 (Best loss)
lrClassifier = LogisticRegression()
parameters = {'clf__C': [0.001, 0.01, 0.1, 0.25, 0.5, 1, 5, 10], 'clf__max_iter': [100, 200, 300, 400],
              'clf__penalty': ['l2'], 'clf__class_weight': [{'A': 1, 'B': 5, 'C': 10}],
          #    'clf__solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
             }

# 15. Ensemble model - VotingClassifier                     # 0.62
estimators = [('SVM', svClassifier), ('RF', rfClassifier), ('GB', gbClassifier), ('SGD', sgdClassifier), ('AB', abClassifier), ('LR', lrClassifier)]
classifier = VotingClassifier(estimators=estimators, voting='soft')

# 16. Ensemble model - Stacking                             # 0.82
baseLearners = {'SVM': svClassifier, 'RF': rfClassifier, 'GB': gbClassifier, 'SGD': sgdClassifier,'AB': abClassifier, 'LR': lrClassifier}
metaLearner = GradientBoostingClassifier(n_estimators=1000)
classifier = SuperLearner(folds=10, backend="multiprocessing")

classifier.add(list(baseLearners.values()), proba=True)
classifier.add_meta(metaLearner, proba=True)

# 17. XGBoost
classifier = XGBClassifier()
parameters = {'clf__max_depth': [3, 4], 'clf__n_estimators': [100, 200, 500], 'clf__learning_rate': [0.05, 0.1, 0.15]}


#######################################################################
# Use pipeline to apply operations sequentially :-
#######################################################################
text_clf_pipeline = Pipeline([
    ('feature_transforms', FeatureUnion([
        ('title_pipeline', Pipeline([
            ('extract_field', FunctionTransformer(lambda x: x['Title'], validate=False)),
            ('title_tfidf', titleTfIdfVectorizer)
        ])),
        ('summary_pipeline', Pipeline([
            ('extract_field', FunctionTransformer(lambda x: x['Event Summary'], validate=False)),
            ('summary_tfidf', summaryTfIdfVectorizer)
        ])),
        ('organisation_pipeline', Pipeline([
            #('extract_field', FunctionTransformer(lambda x: x['Compliance Organisations'].reshape(-1, 1), validate=False)),
            ('extract_field', FunctionTransformer(lambda x: x['Compliance Organisations'], validate=False)),
            ('org_count', orgCountVect)
        ])),
        ('countries_pipeline', Pipeline([
            ('extract_field', FunctionTransformer(lambda x: x['Compliance Countries'], validate=False)),
            ('countries_count', countriesCountVect)
        ])),
        ('extraterritorial_pipeline', Pipeline([
            ('extract_field', FunctionTransformer(lambda x: x['Extraterritorial'], validate=False)),
            ('extraterritorial_one_hot', extraTerritorialOHE)
        ]))
    ]),
     ('clf', classifier
     )
    )
])


#customScoring = make_scorer(log_loss, sample_weight=y_train['weight'])

gsCv = GridSearchCV(text_clf_pipeline, scoring="neg_log_loss", cv=2, param_grid=parameters, refit=True, return_train_score=False, lid=True)


########################################################################
# Fit pipeline to training data
########################################################################
gsCv.fit(X=X_train[['Title', 'Event Summary', 'Compliance Organisations', 'Compliance Countries', 'Extraterritorial']],
         y=y_train['category'],
         #, clf__sample_weight=y_train['weight']
         )


########################################################################
# Make prediction from X_test data using the pipeline
########################################################################
y_pred_prob = gsCv.predict_proba(X_test[['Title', 'Event Summary', 'Compliance Organisations', 'Compliance Countries', 'Extraterritorial']])

y_pred_cat = gsCv.predict(X_test[['Title', 'Event Summary', 'Compliance Organisations', 'Compliance Countries', 'Extraterritorial']])


########################################################################
# Format output for submission
########################################################################
cols_prob = ['confidence(A)', 'confidence(B)', 'confidence(C)']

# Create dataframe from numpy array
df_prob = pd.DataFrame(y_pred_prob, columns=cols_prob)

## Only for superlearner ensemble classifier -------------------------
# dict = {0: 'A', 1: 'B', 2: 'C'}
# y_pred_cat = []

# for y in y_pred_prob:
#   y_pred_cat.append(dict[argmax(y)])

## -------------------------------------------------------------------

cols_cat = ['prediction(category)']

df_cat = pd.DataFrame(y_pred_cat, columns=cols_cat)


# Format output submission:-
# Make dataframe with key (serial number of row in test data csv) taken from raw test data
df_key = X_test.loc[:, ['key']]

dfs = [df_key, df_prob, df_cat] # Dataframes to join

df_results = pd.concat(dfs, axis=1)


##########################################################################
# Save the prediction output result in csv
##########################################################################
filename = "prediction_" + strftime("%Y%m%d_%H%M", gmtime()) + ".csv"

df_results.to_csv(cwd + "\\data\\" + filename, index=False)
