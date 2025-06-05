import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from io import StringIO

# Configuração da página
st.set_page_config(layout="wide", page_title="Análise Interativa de Regressão e ANOVA")

# Estilo CSS para melhorar a aparência (opcional)
st.markdown("""
<style>
    .reportview-container {
        background: #f0f2f6;
    }
    .sidebar .sidebar-content {
        background: #f0f2f6;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        padding: 10px 24px;
    }
    .stSelectbox, .stMultiselect {
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# --- Funções Auxiliares ---
@st.cache_data # Cache para otimizar o carregamento de dados
def load_data(uploaded_file):
    if uploaded_file is not None:
        try:
            # Tenta ler como CSV, que é o formato esperado
            df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Erro ao carregar o arquivo: {e}")
            return None
    else: # Carrega o arquivo AmesHousing.csv por padrão se nenhum arquivo for enviado
        try:
            df = pd.read_csv("AmesHousing.csv")
        except FileNotFoundError:
            st.error("Arquivo AmesHousing.csv não encontrado. Por favor, faça o upload ou coloque-o na pasta do app.")
            return None
    
    # Padroniza nomes das colunas
    df.columns = df.columns.str.strip().str.replace(" ", "_").str.lower()
    return df

def perform_anova(df_anova, categorical_var, target_var):
    st.subheader(f"Análise ANOVA: {target_var} por {categorical_var}")

    df_subset = df_anova[[categorical_var, target_var]].copy()
    df_subset.dropna(inplace=True)

    if df_subset.empty or df_subset[categorical_var].nunique() < 2:
        st.warning("Dados insuficientes ou poucas categorias para realizar ANOVA.")
        return

    # Boxplot
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(x=categorical_var, y=target_var, data=df_subset, ax=ax, palette="viridis")
    ax.set_title(f'Preço de Venda por {categorical_var}')
    ax.set_xlabel(categorical_var)
    ax.set_ylabel(target_var)
    st.pyplot(fig)

    # Grupos para ANOVA
    groups = [group[target_var].values for name, group in df_subset.groupby(categorical_var)]

    # ANOVA
    if len(groups) >= 2: # ANOVA requer pelo menos 2 grupos
        f_statistic, p_value_anova = stats.f_oneway(*groups)
        st.markdown("### Resultados da ANOVA (One-Way)")
        st.write(f"**Estatística F:** {f_statistic:.4f}")
        st.write(f"**Valor-p:** {p_value_anova:.4g}")
        if p_value_anova < 0.05:
            st.success("Há diferenças estatisticamente significativas entre as médias dos grupos.")
        else:
            st.info("Não há evidências de diferenças estatisticamente significativas entre as médias dos grupos.")

        # Verificação de Pressupostos
        st.markdown("### Verificação dos Pressupostos da ANOVA")
        
        # Modelo OLS para resíduos
        try:
            formula = f'`{target_var}` ~ C(`{categorical_var}`)' # Usar crases para nomes de colunas com caracteres especiais
            model_ols = sm.OLS.from_formula(formula, data=df_subset).fit()
            residuals = model_ols.resid

            # Normalidade dos Resíduos (Shapiro-Wilk)
            if len(residuals) >=3: # Shapiro-Wilk requer pelo menos 3 amostras
                shapiro_stat, shapiro_p = stats.shapiro(residuals)
                st.write(f"**Teste de Normalidade dos Resíduos (Shapiro-Wilk) - Valor-p:** {shapiro_p:.4g}")
                if shapiro_p < 0.05:
                    st.warning("Os resíduos não seguem uma distribuição normal.")
                else:
                    st.success("Os resíduos parecem seguir uma distribuição normal.")
            else:
                st.warning("Não foi possível realizar o teste de Shapiro-Wilk (poucos dados nos resíduos).")


            # Homocedasticidade (Levene)
            levene_stat, levene_p = stats.levene(*groups)
            st.write(f"**Teste de Homocedasticidade (Levene) - Valor-p:** {levene_p:.4g}")
            if levene_p < 0.05:
                st.warning("As variâncias dos grupos não são homogêneas (heterocedasticidade).")
            else:
                st.success("As variâncias dos grupos parecem ser homogêneas.")

            # Teste de Kruskal-Wallis (alternativa não paramétrica)
            if shapiro_p < 0.05 or levene_p < 0.05:
                st.markdown("### Teste de Kruskal-Wallis (Alternativa Não Paramétrica)")
                kruskal_stat, kruskal_p = stats.kruskal(*groups)
                st.write(f"**Estatística H (Kruskal-Wallis):** {kruskal_stat:.4f}")
                st.write(f"**Valor-p:** {kruskal_p:.4g}")
                if kruskal_p < 0.05:
                    st.success("O teste de Kruskal-Wallis indica diferenças significativas entre os grupos.")
                else:
                    st.info("O teste de Kruskal-Wallis não indica diferenças significativas entre os grupos.")
        except Exception as e:
            st.error(f"Erro ao verificar pressupostos ou rodar Kruskal-Wallis: {e}")
    else:
        st.warning("ANOVA requer pelo menos dois grupos para comparação.")


def perform_regression(df_reg, explanatory_vars, target_var, apply_log_target, apply_log_continuous):
    st.subheader("Análise de Regressão Linear Múltipla")

    cols_to_use = explanatory_vars + [target_var]
    df_model = df_reg[cols_to_use].copy()
    df_model.dropna(inplace=True)

    if df_model.shape[0] < len(explanatory_vars) + 2: # Verifica se há dados suficientes
        st.warning("Dados insuficientes após remoção de NaNs para construir o modelo.")
        return

    # Identificar variáveis categóricas e contínuas entre as explicativas
    categorical_explanatory = [col for col in explanatory_vars if df_model[col].dtype == 'object' or df_model[col].nunique() < 20] # Heurística para categóricas
    continuous_explanatory = [col for col in explanatory_vars if col not in categorical_explanatory]

    # Aplicar log na variável alvo
    y = df_model[target_var].astype(float)
    if apply_log_target:
        if (y <= 0).any():
            st.warning(f"A variável alvo '{target_var}' contém valores não positivos. Log não aplicado.")
        else:
            y = np.log(y)
            st.info(f"Transformação logarítmica aplicada à variável alvo: '{target_var}'.")
    
    # Preparar X
    X_df = df_model[explanatory_vars].copy()

    # Aplicar log nas variáveis explicativas contínuas selecionadas
    if apply_log_continuous:
        for col in continuous_explanatory:
            if (X_df[col] <= 0).any():
                st.warning(f"Variável explicativa '{col}' contém valores não positivos. Log não aplicado para esta coluna.")
            else:
                X_df[f'log_{col}'] = np.log(X_df[col])
                st.info(f"Transformação logarítmica aplicada à variável explicativa: '{col}'.")
        
        # Remover colunas originais que foram transformadas, exceto se forem dummies
        cols_to_drop_for_log = [col for col in continuous_explanatory if f'log_{col}' in X_df.columns]
        X_df.drop(columns=cols_to_drop_for_log, inplace=True)


    # Criar dummies para variáveis categóricas
    if categorical_explanatory:
        X_df = pd.get_dummies(X_df, columns=categorical_explanatory, drop_first=True, dtype=float)
    
    X_df = X_df.astype(float)
    X_with_const = sm.add_constant(X_df)

    # Ajustar modelo
    try:
        model = sm.OLS(y, X_with_const).fit()
        st.markdown("### Sumário do Modelo de Regressão (OLS Results)")
        st.text(model.summary().as_text())

        # Análise de Resíduos
        st.markdown("### Análise de Resíduos")
        y_pred = model.predict(X_with_const)
        residuals = model.resid

        # Gráfico de Resíduos vs. Preditos
        fig, ax = plt.subplots(figsize=(10,6))
        sns.scatterplot(x=y_pred, y=residuals, ax=ax, color="royalblue", alpha=0.7)
        ax.axhline(0, color='red', linestyle='--')
        ax.set_xlabel('Valores Preditos')
        ax.set_ylabel('Resíduos')
        ax.set_title('Resíduos vs. Valores Preditos')
        st.pyplot(fig)
        
        # Testes de Pressupostos sobre os resíduos
        if len(residuals) >= 3: # Shapiro-Wilk requer pelo menos 3 amostras
            shapiro_stat, shapiro_p = stats.shapiro(residuals)
            st.write(f"**Teste de Normalidade dos Resíduos (Shapiro-Wilk) - Valor-p:** {shapiro_p:.4g}")
            if shapiro_p < 0.05:
                st.warning("Os resíduos não parecem seguir uma distribuição normal.")
            else:
                st.success("Os resíduos parecem seguir uma distribuição normal.")
        else:
            st.warning("Não foi possível realizar o teste de Shapiro-Wilk nos resíduos (poucos dados).")


        if X_with_const.shape[0] > X_with_const.shape[1]: # Breusch-Pagan precisa de mais observações que regressores
            try:
                bp_test_lm, bp_p_value, bp_f_stat, bp_f_p_value = sm.stats.het_breuschpagan(residuals, X_with_const)
                st.write(f"**Teste de Homocedasticidade (Breusch-Pagan) - Valor-p (LM-stat):** {bp_p_value:.4g}")
                if bp_p_value < 0.05:
                    st.warning("Há evidência de heterocedasticidade (variâncias dos resíduos não são constantes).")
                else:
                    st.success("Não há evidência de heterocedasticidade.")
            except Exception as e_bp:
                 st.warning(f"Não foi possível realizar o teste de Breusch-Pagan: {e_bp}")
        else:
            st.warning("Não foi possível realizar o teste de Breusch-Pagan (dados insuficientes ou colinearidade perfeita).")


        # VIF
        st.markdown("### Verificação de Multicolinearidade (VIF)")
        if X_with_const.shape[1] > 1: # VIF requer mais de uma variável (além da constante)
            vif_data = pd.DataFrame()
            vif_data["Variável"] = X_with_const.columns
            try:
                vif_data["VIF"] = [variance_inflation_factor(X_with_const.values, i) for i in range(X_with_const.shape[1])]
                st.dataframe(vif_data.sort_values(by="VIF", ascending=False))
                if (vif_data["VIF"] > 10).any():
                    st.warning("Algumas variáveis apresentam VIF > 10, indicando possível multicolinearidade.")
                elif (vif_data["VIF"] > 5).any():
                    st.warning("Algumas variáveis apresentam VIF > 5, indicando potencial multicolinearidade moderada.")
                else:
                    st.success("Não foram encontrados problemas graves de multicolinearidade (VIF < 5 para todas as variáveis explicativas).")

            except Exception as e_vif:
                 st.warning(f"Não foi possível calcular o VIF para todas as variáveis (pode indicar colinearidade perfeita): {e_vif}")
        else:
            st.info("VIF não aplicável com menos de duas variáveis explicativas.")


        # Métricas de Desempenho
        st.markdown("### Métricas de Desempenho do Modelo")
        
        y_actual_for_metrics = y # Já está em log se apply_log_target for True
        y_pred_for_metrics = y_pred # Predições também estão na escala de y

        if apply_log_target: # Se log foi aplicado no alvo, reverter para escala original para métricas interpretáveis
            y_actual_for_metrics = np.exp(y)
            y_pred_for_metrics = np.exp(y_pred)
            st.info("Métricas RMSE e MAE calculadas na escala original da variável alvo (após exp()). R² é da escala ajustada (log).")
            r2_adjusted_scale = model.rsquared # R² do modelo na escala log
            r2_original_scale = r2_score(y_actual_for_metrics, y_pred_for_metrics) # R² na escala original
            st.write(f"**R² (na escala do modelo, {'logarítmica' if apply_log_target else 'original'}):** {r2_adjusted_scale:.4f}")
            st.write(f"**R² (na escala original da variável alvo):** {r2_original_scale:.4f} (Comparativo)")
        else:
            r2 = model.rsquared
            st.write(f"**R²:** {r2:.4f}")

        rmse = np.sqrt(mean_squared_error(y_actual_for_metrics, y_pred_for_metrics))
        mae = mean_absolute_error(y_actual_for_metrics, y_pred_for_metrics)
        st.write(f"**RMSE (Root Mean Squared Error):** {rmse:.2f}")
        st.write(f"**MAE (Mean Absolute Error):** {mae:.2f}")
        
        st.markdown("---")
        st.markdown("#### Interpretação Geral dos Coeficientes")
        st.write("Os coeficientes no sumário do modelo indicam a mudança média na variável alvo para um aumento de uma unidade na variável explicativa, mantendo as outras constantes.")
        if apply_log_target and any(f'log_{col}' in X_with_const.columns for col in continuous_explanatory if apply_log_continuous):
            st.write("Para **variáveis explicativas contínuas transformadas com log (log_)**: Um aumento de 1% na variável explicativa está associado a uma mudança de (coeficiente * 100)% na variável alvo (se esta também estiver em log). Se a variável alvo não estiver em log, um aumento de 1% na explicativa leva a uma mudança de (coeficiente / 100) unidades na variável alvo.")
        elif apply_log_target:
             st.write("Para **variáveis explicativas contínuas (não log)**: Um aumento de uma unidade na variável explicativa está associado a uma mudança de (coeficiente * 100)% na variável alvo (que está em log).")
        elif any(f'log_{col}' in X_with_const.columns for col in continuous_explanatory if apply_log_continuous):
            st.write("Para **variáveis explicativas contínuas transformadas com log (log_)**: Um aumento de 1% na variável explicativa está associado a uma mudança de (coeficiente / 100) unidades na variável alvo (que não está em log).")

        st.write("Para **variáveis dummy (categóricas)**: O coeficiente representa a diferença média na variável alvo (ou log da variável alvo) entre a categoria representada pela dummy e a categoria base (omitida), mantendo as outras constantes.")


    except Exception as e:
        st.error(f"Erro ao ajustar o modelo de regressão: {e}")
        st.error("Verifique se as variáveis selecionadas são apropriadas e se há dados suficientes.")
        st.error("Problemas comuns incluem colinearidade perfeita ou variáveis com variância zero.")

# --- Interface Principal do Streamlit ---
st.title("📊 Aplicativo Interativo de Análise de Dados")
st.markdown("Realize análises de ANOVA e Regressão Linear de forma interativa.")

# Upload de arquivo
uploaded_file = st.sidebar.file_uploader("Carregue seu arquivo CSV", type=["csv"])
df = load_data(uploaded_file)

if df is not None:
    st.sidebar.success(f"Dados carregados com sucesso! ({df.shape[0]} linhas, {df.shape[1]} colunas)")
    
    analysis_type = st.sidebar.selectbox("Escolha o tipo de análise:", ["ANOVA", "Regressão Linear"])

    if analysis_type == "ANOVA":
        st.sidebar.header("Opções para ANOVA")
        
        numerical_cols_anova = df.select_dtypes(include=np.number).columns.tolist()
        categorical_cols_anova = df.select_dtypes(include='object').columns.tolist()
        
        if not categorical_cols_anova:
            st.error("Nenhuma coluna categórica encontrada no dataset para ANOVA.")
        elif not numerical_cols_anova:
            st.error("Nenhuma coluna numérica encontrada no dataset para ANOVA.")
        else:
            default_cat_anova = 'kitchen_qual' if 'kitchen_qual' in categorical_cols_anova else categorical_cols_anova[0]
            default_num_anova = 'saleprice' if 'saleprice' in numerical_cols_anova else numerical_cols_anova[0]

            categorical_var_anova = st.sidebar.selectbox("Selecione a variável categórica (grupos):", categorical_cols_anova, index=categorical_cols_anova.index(default_cat_anova) if default_cat_anova in categorical_cols_anova else 0)
            target_var_anova = st.sidebar.selectbox("Selecione a variável numérica (alvo):", numerical_cols_anova, index=numerical_cols_anova.index(default_num_anova) if default_num_anova in numerical_cols_anova else 0)

            if st.sidebar.button("Executar ANOVA", key="run_anova"):
                perform_anova(df, categorical_var_anova, target_var_anova)

    elif analysis_type == "Regressão Linear":
        st.sidebar.header("Opções para Regressão Linear")
        
        all_cols_reg = df.columns.tolist()
        numerical_cols_reg = df.select_dtypes(include=np.number).columns.tolist()

        if not numerical_cols_reg:
             st.error("Nenhuma coluna numérica encontrada no dataset para ser a variável alvo da regressão.")
        else:
            default_target_reg = 'saleprice' if 'saleprice' in numerical_cols_reg else numerical_cols_reg[0]
            target_var_reg = st.sidebar.selectbox("Selecione a variável alvo (dependente):", numerical_cols_reg, index=numerical_cols_reg.index(default_target_reg) if default_target_reg in numerical_cols_reg else 0)
            
            available_explanatory = [col for col in all_cols_reg if col != target_var_reg]
            
            # Variáveis explicativas padrão baseadas no notebook, se existirem
            default_explanatory_vars = ['gr_liv_area', 'overall_qual', 'garage_finish', 'kitchen_qual', 'exter_qual', 'central_air']
            default_explanatory_vars_present = [var for var in default_explanatory_vars if var in available_explanatory]
            
            explanatory_vars_reg = st.sidebar.multiselect("Selecione as variáveis explicativas (independentes):", available_explanatory, default=default_explanatory_vars_present)

            apply_log_target_reg = st.sidebar.checkbox("Aplicar log na variável alvo?", value=True)
            
            continuous_for_log_selection = [col for col in explanatory_vars_reg if df[col].dtype in [np.number] and df[col].nunique() > 20] # Heurística para contínuas
            
            apply_log_continuous_reg = []
            if continuous_for_log_selection:
                 apply_log_continuous_reg_selection = st.sidebar.multiselect("Aplicar log em quais variáveis explicativas contínuas?", continuous_for_log_selection, default=[v for v in ['gr_liv_area', 'overall_qual'] if v in continuous_for_log_selection])
                 # O checkbox abaixo foi removido para dar controle mais granular com multiselect
                 # apply_log_continuous_reg = st.sidebar.checkbox("Aplicar log nas variáveis explicativas contínuas selecionadas?", value=True)
                 if apply_log_continuous_reg_selection: # Se o usuário selecionou alguma
                    apply_log_continuous_reg = apply_log_continuous_reg_selection


            if st.sidebar.button("Executar Regressão", key="run_regression"):
                if not explanatory_vars_reg:
                    st.warning("Por favor, selecione pelo menos uma variável explicativa.")
                else:
                    # Passa a lista de colunas contínuas que devem ter log aplicado
                    continuous_to_log = [col for col in explanatory_vars_reg if col in apply_log_continuous_reg]
                    perform_regression(df, explanatory_vars_reg, target_var_reg, apply_log_target_reg, continuous_to_log)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("Desenvolvido com base no notebook fornecido.")

else:
    st.info("Aguardando o carregamento do arquivo de dados CSV...")
