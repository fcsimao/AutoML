# 📊 AutoML para Predição de Evasão Escolar

Este projeto oferece uma aplicação completa de aprendizado de máquina automatizado (AutoML) para prever evasão escolar. Desenvolvido com Python e integrado com Streamlit, o sistema utiliza diversas técnicas avançadas de balanceamento, otimização de hiperparâmetros e análise de fairness para gerar insights claros e acionáveis.

---

## 🚀 Funcionalidades

- **Carregamento de Dados:** Upload simples de arquivos CSV.
- **Pré-processamento:** Escalonamento e balanceamento automático dos dados.
- **Balanceamento de Dados:**
  - SMOTE
  - Oversampling
  - Undersampling
  - Tomek Links
  - Híbrido (SMOTE + Tomek Links)
- **Otimização de Modelos:**
  - Optuna
  - RandomizedSearchCV
  - GridSearchCV
  - BayesSearchCV
- **Modelos Utilizados:**
  - RandomForestClassifier
  - GradientBoostingClassifier
  - XGBClassifier
  - LGBMClassifier
  - LogisticRegression
  - SVC
  - KNeighborsClassifier
  - DecisionTreeClassifier
- **Avaliação de Desempenho:**
  - Curva ROC
  - Curva Precision-Recall
  - Matriz de Confusão
  - Importância das Features (SHAP, LIME)
- **Análise de Fairness:**
  - Demographic Parity
  - Equalized Odds

---

## 📁 Dataset Utilizado

Este projeto utilizou o conjunto de dados aberto `dados_exportados.csv`, que contém variáveis relacionadas à evasão escolar, incluindo:

- **Dados Demográficos:** idade, gênero e nível socioeconômico dos estudantes.
- **Histórico Acadêmico:** notas, frequência escolar e histórico de reprovações.
- **Informações Comportamentais:** engajamento do estudante, comportamento em sala de aula e interações sociais.

O objetivo principal do dataset é fornecer informações que permitam a construção de modelos preditivos capazes de identificar alunos com maior risco de evasão, possibilitando intervenções preventivas eficazes pelas instituições educacionais.

---

## 🛠️ Requisitos

- Python 3.10+
- Bibliotecas Python:
```bash
streamlit
pandas
numpy
scikit-learn
xgboost
lightgbm
shap
lime
fairlearn
optuna
langchain
openai
python-dotenv
matplotlib
seaborn
```

Instale as dependências utilizando:

```bash
pip install -r requirements.txt
```

---

## 🖥️ Como Utilizar

1. Clone este repositório:

```bash
git clone https://github.com/seuusuario/automl-evasaoscolar.git
cd automl-evasaoscolar
```

2. Configure suas credenciais do OpenAI no arquivo `.env`:

```bash
OPENAI_API_KEY="sua-chave-api"
```

3. Execute a aplicação com Streamlit:

```bash
streamlit run autoML.py
```

---

## 📈 Resultados Esperados

Após a execução, você obterá:

- Uma interface gráfica interativa com resultados detalhados de todos os modelos testados.
- Rankings automáticos baseados nas principais métricas de performance.
- Relatórios automáticos gerados por IA (LangChain + OpenAI) com insights explicativos.
- Recomendações práticas para gestores educacionais baseadas em análises quantitativas e qualitativas.

---

## 🤝 Contribuições

Contribuições são bem-vindas! Siga os passos padrão:

1. Fork este repositório.
2. Crie sua branch de feature (`git checkout -b feature/AmazingFeature`).
3. Commit suas alterações (`git commit -m 'Add some AmazingFeature'`).
4. Envie a branch (`git push origin feature/AmazingFeature`).
5. Abra um Pull Request.

---

## 📃 Licença

Este projeto está sob a licença MIT. Veja o arquivo `LICENSE` para detalhes.

---
