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
st.set_page_config(layout="wide", page_title="Análise Interativa de Dados", initial_sidebar_state="expanded")

# --- Funções de Sanitização e Carregamento de Dados ---
def sanitize_column_names(df_cols):
    new_cols = []
    for col in df_cols:
        new_col = str(col).strip().replace(" ", "_").replace("-", "_").replace(".", "_").replace("'", "").replace("`", "").lower()
        if new_col and new_col[0].isdigit():
            new_col = "col_" + new_col # Prefixa se começar com dígito
        new_cols.append(new_col)
    return new_cols

@st.cache_data
def load_data(uploaded_file_obj):
    df_loaded = None
    if uploaded_file_obj is not None:
        try:
            df_loaded = pd.read_csv(uploaded_file_obj)
        except Exception as e:
            st.error(f"Erro ao carregar o arquivo: {e}")
            return None
    else:
        try:
            df_loaded = pd.read_csv("AmesHousing.csv")
        except FileNotFoundError:
            st.sidebar.warning("Arquivo 'AmesHousing.csv' não encontrado. Por favor, faça o upload de um arquivo CSV ou coloque 'AmesHousing.csv' na pasta do app.")
            return None
        except Exception as e:
            st.error(f"Erro ao carregar o arquivo padrão 'AmesHousing.csv': {e}")
            return None
    
    if df_loaded is not None:
        df_loaded.columns = sanitize_column_names(df_loaded.columns)
    return df_loaded

# --- Funções de Análise ---
def perform_anova(df_anova, cat_var, num_var):
    st.header(f"📊 Análise ANOVA: '{num_var}' por '{cat_var}'")
    
    df_subset_anova = df_anova[[cat_var, num_var]].copy()
    df_subset_anova.dropna(inplace=True)

    if df_subset_anova.empty or df_subset_anova[cat_var].nunique() < 2:
        st.warning("Dados insuficientes ou poucas categorias para realizar ANOVA.")
        return

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📦 Boxplot")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(x=cat_var, y=num_var, data=df_subset_anova, ax=ax, palette="viridis")
        ax.set_title(f"Distribuição de '{num_var}' por '{cat_var}'", fontsize=16)
        ax.set_xlabel(cat_var, fontsize=12)
        ax.set_ylabel(num_var, fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(fig)

    groups_anova = [group[num_var].values for name, group in df_subset_anova.groupby(cat_var)]

    if len(groups_anova) >= 2:
        f_stat_anova, p_val_anova = stats.f_oneway(*groups_anova)
        with col2:
            st.subheader("📝 Resultados da ANOVA (One-Way)")
            st.metric(label="Estatística F", value=f"{f_stat_anova:.4f}")
            st.metric(label="Valor-p", value=f"{p_val_anova:.4g}")
            if p_val_anova < 0.05:
                st.success("✔️ Há diferenças estatisticamente significativas entre as médias dos grupos.")
            else:
                st.info("ℹ️ Não há evidências de diferenças estatisticamente significativas.")
        
        with st.expander("🔍 Verificação Detalhada dos Pressupostos da ANOVA"):
            try:
                # Garantir que os nomes das colunas na fórmula sejam seguros
                safe_num_var = f"`{num_var}`"
                safe_cat_var = f"`{cat_var}`"
                formula_anova = f"{safe_num_var} ~ C({safe_cat_var})"
                
                model_ols_anova = sm.OLS.from_formula(formula_anova, data=df_subset_anova).fit()
                residuals_anova = model_ols_anova.resid

                if len(residuals_anova) >= 3:
                    shapiro_stat, shapiro_p_anova = stats.shapiro(residuals_anova)
                    st.write(f"**Teste de Normalidade dos Resíduos (Shapiro-Wilk)**")
                    st.write(f"  - Estatística: {shapiro_stat:.4f}, Valor-p: {shapiro_p_anova:.4g}")
                    if shapiro_p_anova < 0.05:
                        st.warning("   Os resíduos não seguem uma distribuição normal.")
                    else:
                        st.success("   Os resíduos parecem seguir uma distribuição normal.")
                else:
                    st.warning("Não foi possível realizar Shapiro-Wilk (resíduos insuficientes).")

                levene_stat, levene_p_anova = stats.levene(*groups_anova)
                st.write(f"**Teste de Homocedasticidade (Levene)**")
                st.write(f"  - Estatística: {levene_stat:.4f}, Valor-p: {levene_p_anova:.4g}")
                if levene_p_anova < 0.05:
                    st.warning("   As variâncias dos grupos não são homogêneas.")
                else:
                    st.success("   As variâncias dos grupos parecem ser homogêneas.")

                if shapiro_p_anova < 0.05 or levene_p_anova < 0.05:
                    st.markdown("**Teste de Kruskal-Wallis (Alternativa Não Paramétrica)**")
                    kruskal_stat, kruskal_p_anova = stats.kruskal(*groups_anova)
                    st.write(f"  - Estatística H: {kruskal_stat:.4f}, Valor-p: {kruskal_p_anova:.4g}")
                    if kruskal_p_anova < 0.05:
                        st.success("   ✔️ Kruskal-Wallis indica diferenças significativas.")
                    else:
                        st.info("   ℹ️ Kruskal-Wallis não indica diferenças significativas.")
            except Exception as e_anova_assumptions:
                st.error(f"Erro na verificação de pressupostos: {e_anova_assumptions}")
    else:
        st.warning("ANOVA requer pelo menos dois grupos.")


def perform_regression(df_reg, expl_vars, target_var_reg, log_target, log_continuous_vars):
    st.header(f"📈 Análise de Regressão Linear: '{target_var_reg}'")

    cols_to_use_reg = expl_vars + [target_var_reg]
    df_model_reg = df_reg[cols_to_use_reg].copy()
    df_model_reg.dropna(inplace=True)

    if df_model_reg.shape[0] < len(expl_vars) + 2:
        st.warning("Dados insuficientes após remoção de NaNs.")
        return

    y_reg = df_model_reg[target_var_reg].astype(float)
    if log_target:
        if (y_reg <= 0).any():
            st.warning(f"'{target_var_reg}' contém valores não positivos. Log não aplicado.")
        else:
            y_reg = np.log(y_reg)
    
    X_reg_df = df_model_reg[expl_vars].copy()
    
    for col in log_continuous_vars: # log_continuous_vars já é a lista das selecionadas
        if col in X_reg_df.columns:
            if (X_reg_df[col] <= 0).any():
                st.warning(f"'{col}' contém valores não positivos. Log não aplicado para esta coluna.")
            else:
                X_reg_df[f"log_{col}"] = np.log(X_reg_df[col])
    
    cols_to_drop_log = [col for col in log_continuous_vars if f"log_{col}" in X_reg_df.columns]
    X_reg_df.drop(columns=cols_to_drop_log, inplace=True)
    
    cat_expl_vars = [col for col in X_reg_df.columns if X_reg_df[col].dtype == 'object' or (X_reg_df[col].dtype != 'object' and X_reg_df[col].nunique() < 20 and col not in log_continuous_vars and f"log_{col}" not in X_reg_df.columns)] # Evitar dummificar colunas já logaritmizadas
    
    if cat_expl_vars:
        X_reg_df = pd.get_dummies(X_reg_df, columns=cat_expl_vars, drop_first=True, dtype=float)
    
    X_reg_df = X_reg_df.astype(float)
    X_reg_with_const = sm.add_constant(X_reg_df)

    try:
        model_reg = sm.OLS(y_reg, X_reg_with_const).fit()
        st.subheader("📄 Sumário do Modelo (OLS Results)")
        st.text(model_reg.summary().as_text())

        y_pred_reg = model_reg.predict(X_reg_with_const)
        
        with st.expander("🔍 Análise Detalhada dos Resíduos e Diagnósticos"):
            residuals_reg = model_reg.resid
            fig_res, ax_res = plt.subplots(figsize=(10,6))
            sns.scatterplot(x=y_pred_reg, y=residuals_reg, ax=ax_res, color="royalblue", alpha=0.6)
            ax_res.axhline(0, color='red', linestyle='--')
            ax_res.set_xlabel('Valores Preditos')
            ax_res.set_ylabel('Resíduos')
            ax_res.set_title('Resíduos vs. Valores Preditos', fontsize=16)
            st.pyplot(fig_res)
            
            if len(residuals_reg) >= 3:
                shapiro_stat_reg, shapiro_p_reg = stats.shapiro(residuals_reg)
                st.write(f"**Teste de Normalidade (Shapiro-Wilk) - Valor-p:** {shapiro_p_reg:.4g}")
                if shapiro_p_reg < 0.05: st.warning("   Resíduos não normais.")
                else: st.success("   Resíduos normais.")
            
            if X_reg_with_const.shape[0] > X_reg_with_const.shape[1]:
                try:
                    _, bp_p_val_reg, _, _ = sm.stats.het_breuschpagan(residuals_reg, X_reg_with_const)
                    st.write(f"**Teste de Homocedasticidade (Breusch-Pagan) - Valor-p:** {bp_p_val_reg:.4g}")
                    if bp_p_val_reg < 0.05: st.warning("   Heterocedasticidade detectada.")
                    else: st.success("   Homocedasticidade (não rejeitada).")
                except Exception: st.warning("Não foi possível rodar Breusch-Pagan.")
            
            if X_reg_with_const.shape[1] > 1:
                vif_df = pd.DataFrame()
                vif_df["Variável"] = X_reg_with_const.columns
                try:
                    vif_df["VIF"] = [variance_inflation_factor(X_reg_with_const.values, i) for i in range(X_reg_with_const.shape[1])]
                    st.write("**VIF (Fator de Inflação da Variância):**")
                    st.dataframe(vif_df.sort_values(by="VIF", ascending=False))
                    if (vif_df["VIF"] > 5).any(): st.warning("   VIF > 5 para algumas variáveis.")
                except Exception: st.warning("Não foi possível calcular VIF (possível colinearidade perfeita).")

        st.subheader("⚙️ Métricas de Desempenho")
        y_actual_metrics, y_pred_metrics = (np.exp(y_reg), np.exp(y_pred_reg)) if log_target else (y_reg, y_pred_reg)
        
        r2_display = model_reg.rsquared
        r2_original_scale_display = r2_score(y_actual_metrics, y_pred_metrics) if log_target else r2_display
        
        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric(label=f"R² ({'log' if log_target else 'original'})", value=f"{r2_display:.4f}")
        if log_target:
            col_m1.metric(label="R² (escala original)", value=f"{r2_original_scale_display:.4f}")
        col_m2.metric(label="RMSE", value=f"{np.sqrt(mean_squared_error(y_actual_metrics, y_pred_metrics)):.2f}")
        col_m3.metric(label="MAE", value=f"{mean_absolute_error(y_actual_metrics, y_pred_metrics):.2f}")

    except Exception as e_reg:
        st.error(f"Erro ao ajustar regressão: {e_reg}")

# --- Estilo CSS Customizado ---
st.markdown("""
<style>
    body {font-family: 'Roboto', sans-serif;}
    .reportview-container .main .block-container {padding: 2rem;}
    .stTabs [data-baseweb="tab-list"] {gap: 24px;}
    .stTabs [data-baseweb="tab"] {height: 50px; background-color: #f0f2f6; border-radius: 8px 8px 0 0; padding: 10px 20px; transition: background-color 0.3s ease;}
    .stTabs [aria-selected="true"] {background-color: #1f77b4; color: white; box-shadow: 0 4px 8px rgba(0,0,0,0.1);}
    .stButton>button {background-color: #1f77b4; color: white; border-radius: 8px; padding: 12px 28px; border: none; box-shadow: 0 2px 4px rgba(0,0,0,0.1); transition: background-color 0.3s ease;}
    .stButton>button:hover {background-color: #165a87;}
    .stSelectbox, .stMultiselect, .stFileUploader, .stTextInput > div > div > input {border-radius: 8px; border: 1px solid #ccc; padding: 8px;}
    h1, h2, h3 {color: #2c3e50;}
    .stDataFrame {border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);}
    .stMetric {background-color: #ffffff; border-left: 5px solid #1f77b4; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);}
    .sidebar .sidebar-content {background-color: #f8f9fa;}
    .stExpander {border: 1px solid #ddd; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.05);}
    .stExpander header {background-color: #e9ecef; border-radius: 8px 8px 0 0;}
</style>
""", unsafe_allow_html=True)


# --- Interface Principal ---
st.title("🔬 Análise Estatística Interativa de Dados Imobiliários")

# Sidebar para upload e configurações globais
with st.sidebar:
    st.header("⚙️ Configurações")
    uploaded_file_obj = st.file_uploader("Carregue seu arquivo CSV", type=["csv"], help="O arquivo CSV deve ter nomes de colunas na primeira linha.")
    
df_main = load_data(uploaded_file_obj)

if df_main is not None:
    st.sidebar.success(f"🎉 Dados carregados! ({df_main.shape[0]} linhas, {df_main.shape[1]} colunas)")
    
    tab1, tab2, tab3 = st.tabs(["📊 Visão Geral dos Dados", "⚖️ Análise ANOVA", "📈 Análise de Regressão"])

    with tab1:
        st.header("📋 Amostra dos Dados")
        st.dataframe(df_main.head(10))
        
        st.header("ℹ️ Informações Gerais do Dataset")
        buffer = StringIO()
        df_main.info(buf=buffer)
        s = buffer.getvalue()
        st.text(s)

        st.header("📉 Estatísticas Descritivas (Colunas Numéricas)")
        st.dataframe(df_main.describe(include=np.number).transpose())
        
        st.header("📊 Estatísticas Descritivas (Colunas Categóricas)")
        st.dataframe(df_main.describe(include='object').transpose())


    with tab2: # ANOVA
        st.sidebar.subheader("🔧 Opções para ANOVA")
        num_cols_anova_tab = df_main.select_dtypes(include=np.number).columns.tolist()
        cat_cols_anova_tab = [col for col in df_main.columns if df_main[col].nunique() < 30 and df_main[col].dtype == 'object' or df_main[col].dtype.name == 'category'] # Heurística melhorada
        
        if not cat_cols_anova_tab:
            st.warning("Nenhuma coluna categórica adequada (com menos de 30 categorias únicas) encontrada.")
        elif not num_cols_anova_tab:
            st.warning("Nenhuma coluna numérica encontrada.")
        else:
            default_cat_anova_tab = 'kitchen_qual' if 'kitchen_qual' in cat_cols_anova_tab else cat_cols_anova_tab[0]
            default_num_anova_tab = 'saleprice' if 'saleprice' in num_cols_anova_tab else num_cols_anova_tab[0]

            cat_var_anova_tab = st.sidebar.selectbox("Variável Categórica (Grupos):", cat_cols_anova_tab, index=cat_cols_anova_tab.index(default_cat_anova_tab) if default_cat_anova_tab in cat_cols_anova_tab else 0, key="anova_cat_var")
            num_var_anova_tab = st.sidebar.selectbox("Variável Numérica (Alvo):", num_cols_anova_tab, index=num_cols_anova_tab.index(default_num_anova_tab) if default_num_anova_tab in num_cols_anova_tab else 0, key="anova_num_var")

            if st.sidebar.button("Executar Análise ANOVA", key="run_anova_tab", help="Clique para realizar a Análise de Variância."):
                perform_anova(df_main, cat_var_anova_tab, num_var_anova_tab)

    with tab3: # Regressão
        st.sidebar.subheader("🛠️ Opções para Regressão Linear")
        all_cols_reg_tab = df_main.columns.tolist()
        num_cols_reg_tab = df_main.select_dtypes(include=np.number).columns.tolist()

        if not num_cols_reg_tab:
            st.warning("Nenhuma coluna numérica para ser variável alvo.")
        else:
            default_target_reg_tab = 'saleprice' if 'saleprice' in num_cols_reg_tab else num_cols_reg_tab[0]
            target_var_reg_tab = st.sidebar.selectbox("Variável Alvo (Dependente):", num_cols_reg_tab, index=num_cols_reg_tab.index(default_target_reg_tab) if default_target_reg_tab in num_cols_reg_tab else 0, key="reg_target_var")
            
            available_expl_tab = [col for col in all_cols_reg_tab if col != target_var_reg_tab]
            default_expl_vars_tab = ['gr_liv_area', 'overall_qual', 'garage_finish', 'kitchen_qual', 'exter_qual', 'central_air']
            default_expl_vars_present_tab = [var for var in default_expl_vars_tab if var in available_expl_tab]
            
            expl_vars_reg_tab = st.sidebar.multiselect("Variáveis Explicativas (Independentes):", available_expl_tab, default=default_expl_vars_present_tab, key="reg_expl_vars")

            log_target_reg_tab = st.sidebar.checkbox("Aplicar log na variável alvo?", value=True, key="reg_log_target")
            
            continuous_for_log_selection_tab = [col for col in expl_vars_reg_tab if df_main[col].dtype in [np.number] and df_main[col].nunique() > 20]
            log_continuous_vars_reg_tab = []
            if continuous_for_log_selection_tab:
                 log_continuous_vars_reg_tab = st.sidebar.multiselect("Aplicar log em quais explicativas contínuas?", continuous_for_log_selection_tab, default=[v for v in ['gr_liv_area', 'overall_qual'] if v in continuous_for_log_selection_tab], key="reg_log_continuous")

            if st.sidebar.button("Executar Análise de Regressão", key="run_regression_tab", help="Clique para realizar a Regressão Linear Múltipla."):
                if not expl_vars_reg_tab:
                    st.warning("Selecione ao menos uma variável explicativa.")
                else:
                    perform_regression(df_main, expl_vars_reg_tab, target_var_reg_tab, log_target_reg_tab, log_continuous_vars_reg_tab)
else:
    st.info("✨ Bem-vindo! Por favor, carregue um arquivo CSV usando a barra lateral para começar a análise.")

st.sidebar.markdown("---")
st.sidebar.markdown("Desenvolvido com Streamlit")
st.sidebar.markdown("Baseado no notebook fornecido.")

