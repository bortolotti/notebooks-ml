import pandas as pd
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns

from pandas.plotting import scatter_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.linear_model import SGDClassifier

mpl.style.use('ggplot')
# %matplotlib inline

def notebook():

    # Cria as funções para mostrar os gráficos
    def plot_correlation_map(df):
        corr = df.corr()
        _, ax = plt.subplots(figsize = (12, 10))
        cmap = sns.diverging_palette(220, 10, as_cmap=True)
        _ = sns.heatmap(
            corr,
            cmap = cmap,
            square = True,
            cbar_kws={'shrink' : .9},
            ax=ax,
            annot=True,
            annot_kws={'fontsize' : 12}
        )

    def plot_line(loss, ylabel, xlabel='Epochs'):
        fig = plt.figure()
        plt.plot(loss)
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        plt.show()

    def plot_histograms(df, variables, n_rows, n_cols):
        fig = plt.figure(figsize=(16,12))
        for i, var_name in enumerate(variables):
            ax=fig.add_subplot(n_rows, n_cols, i+1)
            df[var_name].hist(bins=10, ax=ax)
            ax.set_title('Skew' + str(round(float(df[var_name].skew()),)))
            ax.set_xticklabels([], visible=False)
            ax.set_yticklabels([], visible=False)
        fig.tight_layout()
        plt.show()

    def plot_distribuition(df, var, target, **kwargs):
        row = kwargs.get('row', None)
        col = kwargs.get('col', None)
        facet = sns.FacetGrid(df, hue=target, aspect=4, row=row, col = col)
        facet.map(sns.kdeplot, var, shade=True)
        facet.set(xlim=(0, df[var].max()))
        facet.add_legend()

    def plot_categories(df, cat, target, **kwargs):
        row = kwargs.get('row', None)
        col = kwargs.get('col', None)
        facet = sns.FacetGrid(df, row= row, col = col)
        facet.map(sns.barplot, cat, target)
        facet.add_legend()

    # Bases Fornecidas
    print('Columns', dataset.columns.values)
    print('')
    print(dataset.info())
    print(dataset.describe(include='all'))
    print(dataset.describe(include=['O']))

    for column in ['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Cabin']:
        print(column, 'unique values:', dataset[column].unique())

    print(dataset.Cabin.isnull().count())

    dataset = dataset.drop(['PassengerId', 'Name', 'Ticket'], axis=1)

    #from pandas.tools.plotting import scatter_matrix

    scatter_matrix(dataset, alpha=0.2, figsize=(30, 30), diagonal='hist')

    plot_correlation_map(dataset)

    plot_distribuition(dataset, var='Age', target='Survived', row='Sex')

    plot_categories(dataset, cat='Embarked', target='Survived')

    plot_categories(dataset, cat='Sex', target='Survived')

    plot_categories(dataset, cat='Pclass', target='Survived')

    # # Converter Sex e Embarked para valores numéricos
    replace = {'Sex' : {'male' : 0, 'female' : 1}, 'Embarked' : { 'C' : 0, 'Q' : 1, 'S' : 2}}
    dataset = dataset.replace(replace)

    print(dataset.Embarked.isnull().sum())
    print(dataset.groupby('Embarked').Embarked.count())

    dataset.Embarked = dataset.Embarked.fillna(2)
    print(dataset.Embarked.isnull().sum())

    dataset.Age = dataset.Age.fillna(dataset.Age.mean())
    print(dataset.Age.isnull().sum())

    dataset.Fare = dataset.Fare.fillna(dataset.Fare.mean())
    print(dataset.Fare.isnull().sum())

    embarked = pd.get_dummies(dataset.Embarked, prefix='Embarked')
    pclass = pd.get_dummies(dataset.Pclass, prefix='Pclass')
    print(embarked.head())

    # # 25
    dataset = pd.concat([dataset, embarked, pclass], axis = 1)
    print(dataset.head())

    cabin = pd.DataFrame()

    # Substitui os dados de cabines faltantes com U
    cabin.Cabin = dataset.Cabin.fillna('U')

    # Mapear cada valor de cabine com a letra da cabine
    cabin.Cabin = cabin.Cabin.map(lambda c: c[0])
    cabin = pd.get_dummies(cabin.Cabin, prefix='Cabin')
    dataset = pd.concat([dataset, cabin], axis=1)
    cabin.tail()

    dataset.tail()
    dataset = dataset.drop(['Cabin', 'Embarked', 'Pclass'], axis=1)
    dataset.head()


def preprocessing(df):
    df = df.drop(['PassengerId', 'Name', 'Ticket'], axis=1)
    replace = {'Sex': {'male': 0, 'female': 1}, 'Embarked': {'C': 0, 'Q': 1, 'S': 2}}
    df = df.replace(replace)
    df.Embarked = df.Embarked.fillna(2)
    df.Age = df.Age.fillna(dataset.Age.mean())
    df.Fare = df.Fare.fillna(dataset.Fare.mean())
    embarked = pd.get_dummies(df.Embarked, prefix='Embarked')
    pclass = pd.get_dummies(df.Pclass, prefix='Pclass')
    cabin = pd.DataFrame()
    cabin.Cabin = df.Cabin.fillna('U')
    cabin.Cabin = cabin.Cabin.map(lambda c: c[0])
    cabin = pd.get_dummies(cabin.Cabin, prefix='Cabin')
    df = pd.concat([df, cabin, embarked, pclass] , axis=1)
    df = df.drop(['Cabin', 'Embarked', 'Pclass'], axis=1)

    # Normalizando os dados
    norm = df.apply(lambda x: (x - np.mean(x)) / (np.max(x) - np.min(x)))
    df.Age = norm.Age
    df.Fare = norm.Fare
    return df

# Programa Principal
train_df = pd.read_csv("E:/Documentos/OneDrive/Documents/Bigdata/Machine Learning/Pratica/notebooks-ml/titanic/data/raw/train.csv")
test_df = pd.read_csv("E:/Documentos/OneDrive/Documents/Bigdata/Machine Learning/Pratica/notebooks-ml/titanic/data/raw/test.csv")
dataset = pd.concat([test_df, train_df])

train_valid_features = preprocessing(train_df)
test_features = preprocessing(test_df)
test_features['Cabin_T'] = 0
train_valid_features.head()
test_features.head()

train_valid_labels = train_valid_features.Survived
train_valid_features = train_valid_features.drop('Survived', axis=1)

train_features, valid_features, train_labels, valid_labels = train_test_split(train_valid_features, train_valid_labels, train_size=0.7)

classifier_lor = LogisticRegression()
classifier_lor.fit(train_features, train_labels)
print(classifier_lor.score(valid_features, valid_labels))

valid_labels_predicted_lor = classifier_lor.predict(valid_features)
print(classification_report(valid_labels, valid_labels_predicted_lor, target_names=['Morreu', 'Sobreviveu']))

classifier_slr = SGDClassifier(loss='log', penalty='l1')
classifier_slr.fit(train_features, train_labels)
print(classifier_slr.score(valid_features, valid_labels))

valid_labels_predicted_slr = classifier_slr.predict(valid_features)
print(classification_report(valid_labels, valid_labels_predicted_slr, target_names=['Morreu', 'Sobreviveu']))
