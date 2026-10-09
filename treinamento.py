import os
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print("## 1- Carregando e preparando os dados ##")
dataset = pd.read_csv('dataset/Crop_recommendation.csv')

#Renomeando colunas para padronização
col_names = ['Nitrogenio', 'Fosforo', 'Potassio', 'Temperatura', 'Umidade', 'pH', 'Precipitação', 'Label']
dataset.columns = col_names

#Separação de Features(X) e Target(y)
X = dataset.iloc[:, :-1].values
y_raw = dataset.iloc[:, -1].values

#Codificação da variável alvo
le = LabelEncoder()
y = le.fit_transform(y_raw)

#Divisão estratificada (80% treino / 20% teste)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("## 2- Treinando o modelo para comparativo ##")
#Modelo 1: Árvore Reduzida
clf_reduzida = DecisionTreeClassifier(criterion='gini', max_depth=5, random_state=42)
clf_reduzida.fit(X_train, y_train)
y_pred_reduzida = clf_reduzida.predict(X_test)
accuracy_reduzida = accuracy_score(y_test, y_pred_reduzida)

#Modelo 2: Árvore Completa/Otimizada
clf = DecisionTreeClassifier(criterion='gini', max_depth=11, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("\n--- Comparativo de Hiperparâmetros ---")
df_comparacao = pd.DataFrame({
    'Modelo': ['Árvore Reduzida', 'Árvore Completa (Escolhida)'],
    'Profundidade': [5, 11],
    'Critério': ['Gini', 'Gini'],
    'Acurácia Teste': [f"{accuracy_reduzida*100:.2f}%", f"{accuracy*100:.2f}%"],
    'Diagnóstico': ['Underfitting', 'Ajuste Ideal']
})
print(df_comparacao.to_string(index=False))

print("\n--- Importância dos Atributos ---")
feature_names = col_names[:-1]
for feature, importancia in zip(feature_names, clf.feature_importances_):
    print(f"- {feature}: {importancia*100:.2f}%")

print("\n--- Avaliação Detalhada ---")
print(classification_report(y_test, y_pred, target_names=le.classes_, zero_division=0))

print("## 3- Teste de Inferência Pontual ##")
novo_dado = [[90, 42, 43, 20.8, 82.0, 6.5, 202.9]]
pred_classe = clf.predict(novo_dado)
cultura = le.inverse_transform(pred_classe)[0]
confianca = max(clf.predict_proba(novo_dado)[0]) * 100
print(f"Exemplo de predição: {cultura.upper()} ({confianca:.2f}% de confiança)")

print("## 4- Salvando para deploy ##")
os.makedirs('modelo', exist_ok=True)
joblib.dump(clf, 'modelo/modelo_arvore_culturas.pkl')
joblib.dump(le, 'modelo/label_encoder.pkl')
print("|- Modelos exportados na pasta 'modelo/'. -|")