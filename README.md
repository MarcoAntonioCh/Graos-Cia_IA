# Grãos & Cia — Recomendação Inteligente de Culturas Agrícolas

Projeto prático desenvolvido para a disciplina de **Tópicos de Inteligência Artificial** do **Instituto Federal de Educação, Ciência e Tecnologia de São Paulo (IFSP) — Campus Catanduva**.

* **Docente:** Prof. Dr. Márcio Andrey Teixeira
* **Integrantes do Grupo:**
  * Ana Letícia Penariol Fabri
  * Luiza Degan de Rizzo
  * Marco Antônio Chaves do Nascimento

---

## Link da Aplicação em Produção (Deploy)

A aplicação web interativa desenvolvida com **Streamlit** está hospedada e acessível publicamente através do link:

> **Deploy no Streamlit Community Cloud:** `https://graos-cia-ia.streamlit.app`  

---

## Descrição do Problema e Solução

O objetivo do projeto é auxiliar produtores rurais e cooperativas agrícolas na tomada de decisão sobre qual cultura plantar em determinada propriedade, mitigando riscos de perda de safra e otimizando o manejo agronômico.

A solução utiliza **Aprendizado de Máquina Supervisionado (Classificação)** com o algoritmo de **Árvore de Decisão (*Decision Tree*)** treinado sobre parâmetros químicos do solo e variáveis climáticas locais para recomendar uma entre 22 culturas agronômicas padrão.

### Variáveis de Entrada (*Features*)
1. **Nitrogênio ($N$):** Conteúdo relativo no solo ($kg/ha$).
2. **Fósforo ($P$):** Conteúdo relativo no solo ($kg/ha$).
3. **Potássio ($K$):** Conteúdo relativo no solo ($kg/ha$).
4. **Temperatura:** Temperatura média ambiente (°C).
5. **Umidade Relativa:** Umidade do ar (%).
6. **pH:** Potencial hidrogeniônico do solo (0 a 14).
7. **Precipitação Pluviométrica:** Volume de chuva esperado ($mm$).

### Variável Alvo (*Target*)
* Recomendação ótima entre 22 culturas: *apple, banana, blackgram, chickpea, coconut, coffee, cotton, grapes, jute, kidneybeans, lentil, mango, mothbeans, mungbean, muskmelon, orange, papaya, pigeonpeas, pomegranate, rice, watermelon* e *maize*.

---

## Síntese dos Modelos e Resultados

Para atender aos requisitos de comparação técnica de hiperparâmetros, duas configurações de profundidade máxima da Árvore de Decisão foram avaliadas:

| Modelo | Hiperparâmetro (`max_depth`) | Critério de Impureza | Acurácia no Teste | Diagnóstico Técnico |
| :--- | :---: | :---: | :---: | :--- |
| **Árvore Podada** | `max_depth = 5` | Gini | 40,68% | *Underfitting* (subajuste severo devido à complexidade de 22 classes) |
| **Árvore Completa (Deploy)** | `max_depth = 11` | Gini | **97,73%** | Ajuste ótimo com alta capacidade de generalização |

O modelo com **`max_depth=11`** foi o selecionado para a aplicação final devido ao equilíbrio entre alta precisão e capacidade de retorno de probabilidades de incerteza em zonas limítrofes.

---

## Estrutura de Arquivos do Projeto

```text
Graos-Cia_IA/
├── .venv/                         # Ambiente virtual Python (isolado)
├── dataset/
│   └── Crop_recommendation.csv    # Conjunto de dados original (2.200 amostras)
├── modelo/
│   ├── modelo_arvore_culturas.pkl # Modelo de Árvore de Decisão serializado
│   └── label_encoder.pkl          # Mapeamento codificado das 22 culturas
├── app.py                         # Aplicação interativa em Streamlit (Deploy)
├── treinamento.py                 # Script autônomo de treinamento e exportação
├── Graos&Cia.ipynb                # Notebook Jupyter/Colab com análises e gráficos
├── requirements.txt               # Lista de dependências e bibliotecas
└── README.md                      # Documentação de execução e reprodução
```

---

## Instruções de Instalação e Execução Local

Siga os passos abaixo para configurar o ambiente e reproduzir o projeto localmente:

### 1. Clonar ou Baixar o Projeto
```bash
git clone https://https://github.com/MarcoAntonioCh/Graos-Cia_IA.git
cd Graos-Cia_IA
```

### 2. Criar e Ativar o Ambiente Virtual (`.venv`)

* **No Linux / macOS:**
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  ```

* **No Windows (Prompt ou PowerShell):**
  ```powershell
  python -m venv .venv
  .venv\Scripts\activate
  ```

### 3. Atualizar o Gerenciador de Pacotes e Instalar Dependências
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. (Opcional) Executar o Script de Treinamento
Para retreinar os modelos a partir do CSV e regenerar os arquivos na pasta `modelo/`:
```bash
python treinamento.py
```

### 5. Executar a Aplicação Web Localmente
Inicie o servidor do Streamlit:
```bash
streamlit run app.py
```
O navegador abrirá automaticamente no endereço: `http://localhost:8501`.

---

## 🧪 Casos de Teste Sugeridos para Verificação

Para validar o funcionamento da aplicação, utilize os cenários pré-estabelecidos:

1. **Entrada Válida 1 (Arroz):**
   * $N=90, P=42, K=43, \text{Temp}=20.8, \text{Umid}=82.0, \text{pH}=6.5, \text{Chuva}=202.9$
   * **Saída esperada:** `RICE` (Confiança: 100%).
2. **Entrada Válida 2 (Melancia):**
   * $N=107, P=21, K=54, \text{Temp}=26.5, \text{Umid}=80.5, \text{pH}=7.0, \text{Chuva}=48.3$
   * **Saída esperada:** `WATERMELON` (Confiança: 100%).
3. **Entrada Válida 3 (Feijão Preto):**
   * $N=22, P=72, K=19, \text{Temp}=27.6, \text{Umid}=55.7, \text{pH}=7.8, \text{Chuva}=137.9$
   * **Saída esperada:** `BLACKGRAM` (Confiança: 100%).
4. **Entrada Inválida / Incompleta:**
   * Deixar campos vazios ou inserir valores fora do escopo físico ($\text{pH}=16.0$ ou $\text{Chuva}=-20.0$).
   * **Saída esperada:** Alertas visuais em vermelho bloqueando a inferência indevida.
5. **Caso Limítrofe / Desafiador (Zona de Transição):**
   * $N=86.0, P=40.0, K=39.0, \text{Temp}=25.72, \text{Umid}=88.17, \text{pH}=6.21, \text{Chuva}=175.61$
   * **Saída esperada:** `JUTE` (~82.11% de confiança) e alerta informando probabilidade concorrente para `RICE` (~17.89%).

---

## ⚠️ Limitações da Solução

* O modelo abrange estritamente o catálogo fechado de 22 culturas pré-definidas.
* Não considera interferências biológicas (pragas, fungos), incidência solar diária, declividade do solo e variáveis econômicas sazonais.
* O sistema deve ser utilizado como ferramenta de apoio à decisão analítica, não dispensando vistorias de profissionais da Agronomia.
