import os
import joblib
import numpy as np
import streamlit as st

st.set_page_config(
    page_title="Grãos & Cia - Recomendação de Culturas",
    page_icon="🌱",
    layout="wide"
)

@st.cache_resource
def carregar_artefatos():
    path_modelo = "modelo/modelo_arvore_culturas.pkl"
    path_encoder = "modelo/label_encoder.pkl"

    if not os.path.exists(path_modelo) or not os.path.exists(path_encoder):
        return None, None

    modelo = joblib.load(path_modelo)
    encoder = joblib.load(path_encoder)
    return modelo, encoder

modelo, encoder = carregar_artefatos()

#Verificação caso os arquivos .pkl não existam
if modelo is None or encoder is None:
    st.error("Arquivos de modelo não encontrados na pasta 'modelo/'. Execute o treinamento primeiro.")
    st.stop()

st.title("Grãos & Cia: Recomendação de Culturas")
st.markdown("""
**Disciplina:** Tópicos Avançados em Inteligência Artificial \n
**Professor:** Prof. Dr. Márcio Andrey Teixeira \n
**Alunos:** *Ana Letícia Penariol Fabri*, *Luiza Degan de Rizzo* e *Marco Antônio Chaves do Nascimento*
""")
st.markdown("""
Esta aplicação utiliza **Aprendizado de Máquina (Árvore de Decisão)** para recomendar o melhor cultivo agrícola com base nos parâmetros químicos do solo e variáveis meteorológicas.
""")
st.divider()
st.subheader("Parâmetros do Solo e Clima")
st.caption("Insira os dados da sua propriedade respeitando as unidades de medida indicadas:")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("#### Nutrientes (Solo)")
    n = st.number_input(
        "Nitrogênio (N) [kg/ha]", 
        min_value=-10.0, max_value=250.0, value=None, placeholder="Ex: 90.0", step=1.0,
        help="Teor de Nitrogênio disponível no solo (faixa típica: 0 a 140 kg/ha)"
    )
    p = st.number_input(
        "Fósforo (P) [kg/ha]",
          min_value=-10.0, max_value=250.0, value=None, placeholder="Ex: 42.0", step=1.0,
          help="Teor de Fósforo disponível no solo (faixa típica: 5 a 145 kg/ha)"
    )
    k = st.number_input(
        "Potássio (K) [kg/ha]",
        min_value=-10.0, max_value=300.0, value=None, placeholder="Ex: 43.0", step=1.0,
        help="Teor de Potássio disponível no solo (faixa típica: 5 a 205 kg/ha)"      
    )

with col2:
    st.markdown("#### Clima")
    temp = st.number_input(
        "Temperatura Média [°C]", 
        min_value=-20.0, max_value=60.0, value=None, placeholder="Ex: 20.8", step=0.5,
        help="Temperatura média local em graus Celsius (faixa típica: 8 a 45 °C)"    
    )
    umidade = st.number_input(
        "Umidade Relativa [%]", 
        min_value=-10.0, max_value=150.0, value=None, placeholder="Ex: 82.0", step=1.0,
        help="Umidade relativa do ar em percentual (faixa física: 0 a 100%)"    
    )

with col3:
    st.markdown("#### Água & Solo")
    ph = st.number_input(
        "pH do Solo (0 a 14)", 
        min_value=-2.0, max_value=16.0, value=None, placeholder="Ex: 6.5", step=0.1,
        help="Acidez ou alcalinidade do solo (faixa agronômica viável: 3.5 a 9.9)"    
    )
    chuva = st.number_input(
        "Precipitação [mm]", 
        min_value=-50.0, max_value=500.0, value=None, placeholder="Ex: 202.9", step=1.0,
        help="Volume de chuvas esperado no ciclo agrícola (faixa típica: 20 a 300 mm)"    
    )

st.markdown("---")
if st.button("Recomendar Melhor Cultivo", use_container_width=True):

    if any(v is None for v in [n, p, k, temp, umidade, ph, chuva]):
        st.error("**Entrada Incompleta:** Por favor, preencha todos os campos antes de prever.")
    else:
        erros = []
        if n < 0 or p < 0 or k < 0:
            erros.append("Os nutrientes (N, P, K) não podem ter valores negativos.")
        if ph < 0.0 or ph > 14.0:
            erros.append("O pH do solo deve estar na escala física entre 0.0 e 14.0.")
        if umidade < 0.0 or umidade > 100.0:
            erros.append("A umidade relativa do ar deve estar entre 0% e 100%.")
        if chuva < 0.0:
            erros.append("A precipitação de chuva não pode ser negativa.")
        if temp < -10.0 or temp > 55.0:
            erros.append("A temperatura informada está fora de limites viáveis para agricultura.")
        if erros:
            for erro in erros:
                st.error(f"**Entrada Inválida:** {erro}")
        else:
            entrada = np.array([[n, p, k, temp, umidade, ph, chuva]])
            
            pred_num = modelo.predict(entrada)[0]
            cultura_recomendada = encoder.inverse_transform([pred_num])[0]
            
            probabilidades = modelo.predict_proba(entrada)[0]
            confianca = max(probabilidades) * 100
            
            st.success(f"### Cultura Recomendada: **{cultura_recomendada.upper()}**")
            st.metric(label="Grau de Confiança do Modelo", value=f"{confianca:.2f}%")
            
            if confianca < 100.0:
                st.info(" **Zona de Transição Identificada:** O solo/clima analisado apresenta características favoráveis a mais de uma cultura. Verifique abaixo as opções avaliadas:")
                for idx in np.where(probabilidades > 0)[0]:
                    nome = encoder.inverse_transform([idx])[0]
                    st.write(f"- **{nome.capitalize()}**: {probabilidades[idx]*100:.2f}%")

st.markdown("---")
with st.expander("Sobre a Aplicação e Limitações do Modelo"):
    st.markdown("""
    * **Espaço de Decisão:** O modelo foi treinado em um catálogo fechado de 22 culturas agronômicas padrão.
    * **Variáveis Consideradas:** A recomendação baseia-se unicamente em $N, P, K$, temperatura, umidade, pH e chuva. Fatores como pragas, declividade do terreno, sazonalidade de mercado e radiação solar não são contemplados.
    * **Propósito:** Esta ferramenta atua como sistema de suporte à decisão analítica, não substituindo laudos técnicos de engenheiros agrônomos.
    """)