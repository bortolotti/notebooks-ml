# http://dados.gov.br/dataset/ocorrencias-aeronauticas-da-aviacao-civil-brasileira

import pandas as pd

from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report
from sklearn import svm
from sklearn import tree


# Métodos da aplicação

def gerar_faixa_horario(horario: float):
    """ 
    Gerar a faixa de horário conforme o horário da ocorrência 
    0 = 00:00 até 05:59 
    1 = 06:00 até 11:59
    2 = 12:00 até 17:59
    3 = 18:00 até 23:59
    """
    r = 0
    if horario <= 5:
        r = 0
    elif horario <= 11:
        r = 1
    elif horario <= 17:
        r = 2
    elif horario <= 23:
        r = 3
    return r

def processar_dados(dados_total: pd.DataFrame):
    """
    Formatar e processar a tabela de dados
    """
    # Para aeronaves sem idade atribuir nulo neste passo, pois depois será atribuído a idade média
    dados_total.aeronave_ano_fabricacao = dados_total.aeronave_ano_fabricacao.apply(lambda x: None if x <= 0 else x)

    # Criar coluna com idade da aeronave
    dados_total['aeronave_idade'] = data_atual.year - dados_total['aeronave_ano_fabricacao']

    # Preencher a idade da aeronave com a idade média das aeronaves
    dados_total.aeronave_idade = dados_total.aeronave_idade.fillna(dados_total.aeronave_idade.mean())

    # Criar coluna que identifica se a ocorrência foi fatal
    dados_total['ocorrencia_fatal'] = dados_total["total_fatalidades"] > 0
    dados_total.ocorrencia_fatal = dados_total.ocorrencia_fatal.apply(lambda x: 1 if x else 0)

    # Criar coluna que identifica o período da ocorrencia
    dados_total['faixa_horario'] = dados_total['ocorrencia_horario'].apply(lambda x: str(x[:2])).apply(lambda x: gerar_faixa_horario(float(x)))

    # Transformar o estado em colunas para possuir somente valores 0 ou 1
    coluna_ocorrencia_uf = pd.get_dummies(dados_total.ocorrencia_uf, prefix='uf')

    # Transformar o fabricante em colunas para possuir somente valores 0 ou 1
    coluna_aeronave_fabricante = pd.get_dummies(dados_total.aeronave_fabricante, prefix='fabricante')

    # Adicionar as colunas transformadas ao modelo de dados
    dados_total = pd.concat([dados_total, coluna_ocorrencia_uf, coluna_aeronave_fabricante], axis=1)

    # Eliminar colunas desnecessárias
    dados_total = dados_total.drop(['ocorrencia_horario', 'aeronave_ano_fabricacao', 'total_fatalidades', 'codigo_ocorrencia', 'ocorrencia_uf', 'aeronave_fabricante'], axis=1)

    return dados_total

# Variáveis principais
nome_arquivo_ocorrencia = 'E:/Documentos/OneDrive/Documents/Bigdata/Machine Learning/Pratica/notebooks-ml/aviacao/raw/oco.csv'
nome_arquivo_aeronave = 'E:/Documentos/OneDrive/Documents/Bigdata/Machine Learning/Pratica/notebooks-ml/aviacao/raw/anv.csv'
nome_arquivo_input = 'E:/Documentos/OneDrive/Documents/Bigdata/Machine Learning/Pratica/notebooks-ml/aviacao/raw/input.csv'
data_atual = datetime.now()

# Carregar dados da ocorrência
dados_ocorrencia = pd.read_csv(nome_arquivo_ocorrencia, sep='~', encoding='utf-8')
dados_ocorrencia = dados_ocorrencia.loc[(dados_ocorrencia["ocorrencia_classificacao"] != ""), ["codigo_ocorrencia", "ocorrencia_horario", "ocorrencia_uf"]]

# Carregar dados da aeronave
dados_aeronave = pd.read_csv(nome_arquivo_aeronave, sep='~', encoding='utf-8')
dados_aeronave = dados_aeronave.loc[(dados_aeronave["total_fatalidades"] >= 0) & (dados_aeronave["aeronave_tipo_veiculo"] == "AVIÃO"), ["codigo_ocorrencia", "aeronave_fabricante", "aeronave_ano_fabricacao", "total_fatalidades"]]

# Unificar a base
dados_total = pd.merge(dados_ocorrencia, dados_aeronave, how='inner', on=['codigo_ocorrencia', 'codigo_ocorrencia'])

# Input manual para previsão de acidente fatal
dados_input = pd.read_csv(nome_arquivo_input, sep=';', encoding='utf-8')
dados_input_label = dados_input.ocorrencia_fatal
dados_input = dados_input.drop('ocorrencia_fatal', axis = 1)

# ? Remover dados de treino do dados_input_total?
# Criar algumas colunas faltantes
# dados_input = processar_dados(dados_input)

dados_treino = processar_dados(dados_total)
# print(dados_treino.head())
# dados_treino.to_csv(nome_arquivo_input, sep='~', encoding='utf-8')

# Treinar o modelo
train_valid_labels = dados_treino.ocorrencia_fatal
train_valid_features = dados_treino.drop('ocorrencia_fatal', axis = 1)

train_features, valid_features, train_labels, valid_labels = train_test_split(train_valid_features, train_valid_labels, train_size=0.7)

# Classificador "logistic regression"
#print(classification_report(valid_labels, valid_labels_predicted_lor, target_names=['Não Fatal', 'Fatal']))

classifier_lor = LogisticRegression()
classifier_lor.fit(train_features, train_labels)

print(classifier_lor.score(valid_features, valid_labels))

valid_labels_predicted_lor = classifier_lor.predict(dados_input)
print (valid_labels_predicted_lor)


# Classificador Linear SGDC
classifier_slr = SGDClassifier(loss='log', penalty='l1')
classifier_slr.fit(train_features, train_labels)
print(classifier_slr.score(valid_features, valid_labels))

valid_labels_predicted_slr = classifier_slr.predict(dados_input)
print(valid_labels_predicted_slr)


# Classificador Linear SVC
classifier_svm = svm.SVC(probability=True)
classifier_svm.fit(train_features, train_labels)
print(classifier_svm.score(valid_features, valid_labels))

valid_labels_predicted_svm = classifier_svm.predict(dados_input)
print(valid_labels_predicted_svm)


# Classificador Árvore de Decisão
classifier_dt = tree.DecisionTreeClassifier()
classifier_dt.fit(train_features, train_labels)
print(classifier_dt.score(valid_features, valid_labels))

valid_labels_predicted_dt = classifier_dt.predict(dados_input)
print(valid_labels_predicted_dt)

print(classification_report(dados_input_label, valid_labels_predicted_dt, target_names=['Não Fatal', 'Fatal']))
