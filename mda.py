import pandas as pdata
import numpy as ndata
from sklearn.model_selection import train_test_split as ttrain
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as Ldisc
from sklearn.metrics import f1_score as f1eval
from sklearn.preprocessing import StandardScaler as scaleprep
from itertools import product as prodset
from sklearn.metrics import precision_score as preciseval
from itertools import combinations as combsets

datafile = r'data_SEDAC.csv'
datapull = pdata.read_csv(datafile, encoding='ISO-8859-1')

datapull['GDP04'] = pdata.to_numeric(datapull['GDP04'], errors='coerce')

columns = ['EPI2006', 'ENVHEALEPI', 'BIODIVEPI', 'ENERGYEPI', 'CO2GDPRAW',
           'ENRGYINT', 'INVEST', 'EXTDEBT', 'LANDAREA', 'WATICEAREA',
           'TOTALAREA', 'GDPPC05', 'FOREST', 'WATSUPRAW', 'RECYCLE']

datapull[columns] = datapull[columns].apply(pdata.to_numeric, errors='coerce')

lowlimit = datapull['GDP04'].quantile(0.20)
highlimit = datapull['GDP04'].quantile(0.80)

datapull['groupGDP'] = ndata.nan
datapull['groupGDP'] = datapull['groupGDP'].astype(object)

datapull['groupGDP'] = ndata.where(datapull['GDP04'] >= highlimit, 'Upper GDP', datapull['groupGDP'])
datapull['groupGDP'] = ndata.where(datapull['GDP04'] <= lowlimit, 'Lower GDP', datapull['groupGDP'])

datapull = datapull.dropna(subset=['groupGDP'])

def fill_missing(datapull, columns):
    for cols in columns:
        for grp in ['Upper GDP', 'Lower GDP']:
            avg = datapull[datapull['groupGDP'] == grp][cols].mean()
            if ndata.isnan(avg):
                allmean = datapull[cols].mean()
                datapull.loc[(datapull['groupGDP'] == grp) & (datapull[cols].isna()), cols] = allmean
            else:
                datapull.loc[(datapull['groupGDP'] == grp) & (datapull[cols].isna()), cols] = avg
    return datapull

datapull = fill_missing(datapull, columns)

response = datapull['groupGDP'].map({'Upper GDP': 1, 'Lower GDP': 0})

def check_combos(columns, datapull, response):
    top_score = 0
    top_combo = None
    all_combos = list(combsets(columns, 5))
    total = len(all_combos)

    for idx, comb in enumerate(all_combos):
        if idx % 100 == 0:
            print(f"Testing combo {idx + 1} of {total}...")

        featureset = datapull[list(comb)]
        xtrain, xtest, ytrain, ytest = ttrain(featureset, response, test_size=0.3, random_state=42)

        lda_model = Ldisc()
        lda_model.fit(xtrain, ytrain)
        predictions = lda_model.predict(xtest)

        precision = preciseval(ytest, predictions, zero_division=0)

        if precision > top_score:
            top_score = precision
            top_combo = comb

    return top_score, top_combo

top_score, top_combo = check_combos(columns, datapull, response)

print(f"Highest precision score: {top_score}")
print(f"Top feature combo: {top_combo}")

datapull = pdata.read_csv(datafile, encoding='ISO-8859-1')
datapull['GDP04'] = pdata.to_numeric(datapull['GDP04'], errors='coerce')

best_combo = ['EPI2006', 'ENVHEALEPI', 'BIODIVEPI', 'ENERGYEPI', 'INVEST']

low20 = datapull['GDP04'].quantile(0.20)
high80 = datapull['GDP04'].quantile(0.80)

datapull['groupGDP'] = ndata.nan
datapull['groupGDP'] = datapull['groupGDP'].astype(object)

datapull['groupGDP'] = ndata.where(datapull['GDP04'] >= high80, 'Upper GDP', datapull['groupGDP'])
datapull['groupGDP'] = ndata.where(datapull['GDP04'] <= low20, 'Lower GDP', datapull['groupGDP'])

datapull = datapull.dropna(subset=['groupGDP'])

def clean_columns(datapull, columns):
    for cols in columns:
        datapull[cols] = pdata.to_numeric(datapull[cols], errors='coerce')
    return datapull

datapull = clean_columns(datapull, best_combo)

def fill_missing_vals(datapull, columns):
    for cols in columns:
        for grp in ['Upper GDP', 'Lower GDP']:
            avg = datapull[datapull['groupGDP'] == grp][cols].mean()
            if ndata.isnan(avg):
                allmean = datapull[cols].mean()
                datapull.loc[(datapull['groupGDP'] == grp) & (datapull[cols].isna()), cols] = allmean
            else:
                datapull.loc[(datapull['groupGDP'] == grp) & (datapull[cols].isna()), cols] = avg
    return datapull

datapull = fill_missing_vals(datapull, best_combo)

response = datapull['groupGDP'].map({'Upper GDP': 1, 'Lower GDP': 0})
features = datapull[best_combo]

xtrain, xtest, ytrain, ytest = ttrain(features, response, test_size=0.3, random_state=42)

scaler = scaleprep()
xtrain = scaler.fit_transform(xtrain)
xtest = scaler.transform(xtest)

weights = ndata.linspace(0.1, 2.0, 10)

best_f1 = 0
best_wts = None

def test_weights(xtrain, xtest, ytrain, ytest, weights):
    xtrain_w = xtrain * weights
    xtest_w = xtest * weights

    lda_model = Ldisc()
    lda_model.fit(xtrain_w, ytrain)

    predictions = lda_model.predict(xtest_w)
    f1_val = f1eval(ytest, predictions)

    return f1_val

comb_count = len(weights) ** len(best_combo)
comb_done = 0

for wts in prodset(weights, repeat=len(best_combo)):
    wts = ndata.array(wts)
    f1_val = test_weights(xtrain, xtest, ytrain, ytest, wts)

    if f1_val > best_f1:
        best_f1 = f1_val
        best_wts = wts

    comb_done += 1
    prog = (comb_done / comb_count) * 100
    print(f"progress: {prog:.2f}% done", end='\r')

print(f"\nProgress complete: 100.00%")

print(f"Top F1-score: {best_f1:.4f}")
print("Best weights for features:")
for feat, wt in zip(best_combo, best_wts):
    print(f"{feat}: {wt:.4f}")

