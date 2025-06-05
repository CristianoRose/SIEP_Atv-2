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
st.set_page_config(layout="wide", page_title="Análise Interativa de Dados de Chocolate", initial_sidebar_state="expanded")

# --- Funções de Sanitização e Carregamento de Dados ---
def sanitize_column_names(df_cols):
    new_cols = []
    for col in df_cols:
        new_col = str(col).strip().replace(" ", "_").replace("-", "_").replace(".", "_") \
                  .replace("'", "").replace("`", "").replace("(", "").replace(")", "") \
                  .replace("\n", "_").lower() # Adicionado replace para quebras de linha
        if new_col and new_col[0].isdigit():
            new_col = "col_" + new_col # Prefixa se começar com dígito
        new_cols.append(new_col)
    return new_cols

@st.cache_data
def load_data(uploaded_file_obj):
    df_loaded = None
    default_file = "flavors_of_cacao.csv" # Novo default

    if uploaded_file_obj is not None:
        try:
            df_loaded = pd.read_csv(uploaded_file_obj)
        except Exception as e:
            st.error(f"Erro ao carregar o arquivo enviado: {e}")
            return None
    else:
        try:
            df_loaded = pd.read_csv(default_file)
        except FileNotFoundError:
            st.sidebar.warning(f"Arquivo padrão '{default_file}' não encontrado. Por favor, faça o upload de um arquivo CSV ou coloque '{default_file}' na pasta do app.")
            return None
        except Exception as e:
            st.error(f"Erro ao carregar o arquivo padrão '{default_file}': {e}")
            return None
    
    if df_loaded is not None:
        df_loaded.columns = sanitize_column_names(df_loaded.columns)
        
        # Limpeza específica para o dataset de cacau
        if 'cocoa_percent' in df_loaded.columns:
            try:
                df_loaded['cocoa_percent'] = df_loaded['cocoa_percent'].astype(str).str.rstrip('%').astype('float')
            except ValueError:
                st.warning("Não foi possível converter 'cocoa_percent' para numérico. Verifique o formato.")
        
        if 'bean_type' in df_loaded.columns: # Limpeza para \xa0
             df_loaded['bean_type'] = df_loaded['bean_type'].replace(u'\xa0', np.nan)

    return df_loaded

# --- Funções de Análise ---
def perform_anova(df_anova, cat_var, num_var):
    if not cat_var or not num_var:
        st.warning("Por favor, selecione uma variável categórica e uma numérica para a ANOVA.")
        return

    st.header(f"📊 Análise ANOVA: '{num_var}' por '{cat_var}'")
    
    df_subset_anova = df_anova[[cat_var, num_var]].copy()
    df_subset_anova[num_var] = pd.to_numeric(df_subset_anova[num_var], errors='coerce') # Garantir numérico
    df_subset_anova.dropna(inplace=True)

    if df_subset_anova.empty or df_subset_anova[cat_var].nunique() < 2:
        st.warning("Dados insuficientes ou poucas categorias para realizar ANOVA após limpeza/filtragem.")
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
                safe_num_var = f"`{num_var}`"
                safe_cat_var = f"`{cat_var}`"
                formula_anova = f"{safe_num_var} ~ C({safe_cat_var})"
                
                model_ols_anova = sm.OLS.from_formula(formula_anova, data=df_subset_anova).fit()
                residuals_anova = model_ols_anova.resid

                shapiro_p_anova_val = 1.0 # Default para caso de erro
                if len(residuals_anova) >= 3:
                    shapiro_stat, shapiro_p_anova_val = stats.shapiro(residuals_anova)
                    st.write(f"**Teste de Normalidade dos Resíduos (Shapiro-Wilk)**")
                    st.write(f"  - Estatística: {shapiro_stat:.4f}, Valor-p: {shapiro_p_anova_val:.4g}")
                    if shapiro_p_anova_val < 0.05:
                        st.warning("   Os resíduos não seguem uma distribuição normal.")
                    else:
                        st.success("   Os resíduos parecem seguir uma distribuição normal.")
                else:
                    st.warning("Não foi possível realizar Shapiro-Wilk (resíduos insuficientes).")

                levene_p_anova_val = 1.0 # Default
                if len(groups_anova) > 0 and all(len(g) > 0 for g in groups_anova):
                    levene_stat, levene_p_anova_val = stats.levene(*groups_anova)
                    st.write(f"**Teste de Homocedasticidade (Levene)**")
                    st.write(f"  - Estatística: {levene_stat:.4f}, Valor-p: {levene_p_anova_val:.4g}")
                    if levene_p_anova_val < 0.05:
                        st.warning("   As variâncias dos grupos não são homogêneas.")
                    else:
                        st.success("   As variâncias dos grupos parecem ser homogêneas.")
                else:
                    st.warning("Não foi possível realizar teste de Levene (grupos vazios ou insuficientes).")


                if shapiro_p_anova_val < 0.05 or levene_p_anova_val < 0.05:
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
    if not expl_vars or not target_var_reg:
        st.warning("Por favor, selecione a variável alvo e pelo menos uma variável explicativa.")
        return

    st.header(f"📈 Análise de Regressão Linear: '{target_var_reg}'")

    cols_to_use_reg = list(set(expl_vars + [target_var_reg])) # Usar set para evitar duplicatas
    df_model_reg = df_reg[cols_to_use_reg].copy()
    
    # Converter colunas numéricas para tipo numérico, tratando erros
    for col in expl_vars + [target_var_reg]:
        if col in df_model_reg.columns:
            df_model_reg[col] = pd.to_numeric(df_model_reg[col], errors='coerce')

    df_model_reg.dropna(inplace=True)


    if df_model_reg.shape[0] < len(expl_vars) + 2:
        st.warning("Dados insuficientes após remoção de NaNs para construir o modelo.")
        return

    y_reg = df_model_reg[target_var_reg].astype(float)
    if log_target:
        if (y_reg <= 0).any():
            st.warning(f"A variável alvo '{target_var_reg}' contém valores não positivos. Log não aplicado.")
            y_was_logged = False
        else:
            y_reg = np.log(y_reg)
            y_was_logged = True # Flag para saber se o log foi aplicado no alvo
    else:
        y_was_logged = False # Flag
    
    X_reg_df = df_model_reg[expl_vars].copy()
    
    log_transformed_expl_cols = []
    for col in log_continuous_vars: 
        if col in X_reg_df.columns:
            if (X_reg_df[col] <= 0).any():
                st.warning(f"Variável explicativa '{col}' contém valores não positivos. Log não aplicado para esta coluna.")
            else:
                X_reg_df[f"log_{col}"] = np.log(X_reg_df[col])
                log_transformed_expl_cols.append(col) # Colunas originais que foram transformadas
    
    # Remover colunas originais que foram transformadas para log
    X_reg_df.drop(columns=log_transformed_expl_cols, inplace=True, errors='ignore')
        
    # Identificar categóricas *depois* de transformar contínuas e antes de dummificar
    # Garantir que as colunas 'log_...' não sejam tratadas como categóricas erroneamente
    current_expl_cols = X_reg_df.columns.tolist()
    cat_expl_vars = [
        col for col in current_expl_cols 
        if X_reg_df[col].dtype == 'object' or 
           (X_reg_df[col].dtype != 'object' and X_reg_df[col].nunique() < 20 and not col.startswith("log_"))
    ]
        
    if cat_expl_vars:
        X_reg_df = pd.get_dummies(X_reg_df, columns=cat_expl_vars, drop_first=True, dtype=float)
    
    X_reg_df = X_reg_df.astype(float)
    X_reg_with_const = sm.add_constant(X_reg_df, has_constant='add')

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
            
            if X_reg_with_const.shape[0] > X_reg_with_const.shape[1] and X_reg_with_const.shape[1] > 0 : # Checar se há regressores
                try:
                    _, bp_p_val_reg, _, _ = sm.stats.het_breuschpagan(residuals_reg, X_reg_with_const)
                    st.write(f"**Teste de Homocedasticidade (Breusch-Pagan) - Valor-p:** {bp_p_val_reg:.4g}")
                    if bp_p_val_reg < 0.05: st.warning("   Heterocedasticidade detectada.")
                    else: st.success("   Homocedasticidade (não rejeitada).")
                except Exception: st.warning("Não foi possível rodar Breusch-Pagan (verifique colinearidade ou dados).")
            
            if X_reg_with_const.shape[1] > 1: # VIF requer mais de uma variável (incluindo constante)
                vif_df = pd.DataFrame()
                vif_df["Variável"] = X_reg_with_const.columns
                try:
                    vif_df["VIF"] = [variance_inflation_factor(X_reg_with_const.values, i) for i in range(X_reg_with_const.shape[1])]
                    st.write("**VIF (Fator de Inflação da Variância):**")
                    st.dataframe(vif_df[vif_df["Variável"] != "const"].sort_values(by="VIF", ascending=False)) # Exclui constante do display
                    if (vif_df.loc[vif_df["Variável"] != "const", "VIF"] > 5).any(): st.warning("   VIF > 5 para algumas variáveis explicativas.")
                except Exception: st.warning("Não foi possível calcular VIF (possível colinearidade perfeita).")

        st.subheader("⚙️ Métricas de Desempenho")
        y_actual_metrics, y_pred_metrics = (np.exp(y_reg), np.exp(y_pred_reg)) if y_was_logged else (y_reg, y_pred_reg)
        
        r2_display = model_reg.rsquared
        r2_original_scale_display = r2_score(y_actual_metrics, y_pred_metrics) if y_was_logged else r2_display
        
        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric(label=f"R² ({'log da alvo' if y_was_logged else 'alvo original'})", value=f"{r2_display:.4f}")
        if y_was_logged: # Mostrar R2 na escala original apenas se o alvo foi logado
            col_m1.metric(label="R² (escala original do alvo)", value=f"{r2_original_scale_display:.4f}")
        col_m2.metric(label="RMSE", value=f"{np.sqrt(mean_squared_error(y_actual_metrics, y_pred_metrics)):.2f}")
        col_m3.metric(label="MAE", value=f"{mean_absolute_error(y_actual_metrics, y_pred_metrics):.2f}")

    except Exception as e_reg:
        st.error(f"Erro ao ajustar regressão: {e_reg}")
        st.error("Verifique se as variáveis selecionadas são válidas e se há dados suficientes após a limpeza.")


# --- Estilo CSS Customizado ---
st.markdown("""
<style>
    body {font-family: 'Roboto', sans-serif;}
    .reportview-container .main .block-container {padding-top: 2rem; padding-bottom: 2rem;}
    .stTabs [data-baseweb="tab-list"] {gap: 24px;}
    .stTabs [data-baseweb="tab"] {height: 50px; background-color: #f0f2f6; border-radius: 8px 8px 0 0; padding: 10px 20px; transition: background-color 0.3s ease;}
    .stTabs [aria-selected="true"] {background-color: #1f77b4; color: white; box-shadow: 0 4px 8px rgba(0,0,0,0.1);}
    .stButton>button {background-color: #1f77b4; color: white; border-radius: 8px; padding: 10px 24px; border: none; box-shadow: 0 2px 4px rgba(0,0,0,0.1); transition: background-color 0.3s ease;}
    .stButton>button:hover {background-color: #165a87;}
    .stSelectbox, .stMultiselect, .stFileUploader, .stTextInput > div > div > input {border-radius: 8px; border: 1px solid #ccc; padding: 8px;}
    h1, h2, h3 {color: #2c3e50;}
    .stDataFrame {border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);}
    .stMetric {background-color: #ffffff; border-left: 5px solid #1f77b4; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);}
    .sidebar .sidebar-content {background-color: #f8f9fa; padding: 1rem;}
    .stExpander {border: 1px solid #ddd; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.05);}
    .stExpander header {background-color: #e9ecef; border-radius: 8px 8px 0 0;}
</style>
""", unsafe_allow_html=True)


# --- Interface Principal ---
st.title("🍫 Análise Interativa de Avaliações de Chocolate")

# Sidebar para upload e configurações globais
with st.sidebar:
    st.header("⚙️ Configurações")
    uploaded_file_obj = st.file_uploader("Carregue seu arquivo CSV (Opcional)", type=["csv"], help="Se nenhum arquivo for enviado, 'flavors_of_cacao.csv' será usado.")
    
df_main = load_data(uploaded_file_obj)

if df_main is not None:
    st.sidebar.success(f"🎉 Dados carregados! ({df_main.shape[0]} linhas, {df_main.shape[1]} colunas)")
    
    tab_overview, tab_anova, tab_regression = st.tabs(["📊 Visão Geral", "⚖️ Análise ANOVA", "📈 Análise de Regressão"])

    with tab_overview:
        st.header("📋 Amostra dos Dados")
        st.dataframe(df_main.head(10))
        
        with st.expander("ℹ️ Informações Gerais do Dataset (df.info)"):
            buffer = StringIO()
            df_main.info(buf=buffer)
            s = buffer.getvalue()
            st.text(s)

        with st.expander("📉 Estatísticas Descritivas (Colunas Numéricas)"):
            st.dataframe(df_main.describe(include=np.number).transpose())
        
        with st.expander("📊 Estatísticas Descritivas (Colunas Categóricas)"):
            st.dataframe(df_main.describe(include='object').transpose())


    with tab_anova: # ANOVA
        st.sidebar.subheader("🔧 Opções para ANOVA")
        num_cols_anova_tab = df_main.select_dtypes(include=np.number).columns.tolist()
        # Heurística para colunas categóricas: object/category ou numéricas com poucas categorias únicas
        cat_cols_anova_tab = [col for col in df_main.columns if 
                              (df_main[col].dtype == 'object' or df_main[col].dtype.name == 'category' or 
                               (df_main[col].dtype in [np.int64, np.float64] and df_main[col].nunique() < 30))]
        
        if not cat_cols_anova_tab:
            st.warning("Nenhuma coluna categórica adequada encontrada.")
        elif not num_cols_anova_tab:
            st.warning("Nenhuma coluna numérica encontrada.")
        else:
            # Defaults para o dataset de cacau
            default_cat_anova_tab = 'company_location' if 'company_location' in cat_cols_anova_tab else cat_cols_anova_tab[0]
            default_num_anova_tab = 'rating' if 'rating' in num_cols_anova_tab else num_cols_anova_tab[0]

            cat_var_anova_tab = st.sidebar.selectbox("Variável Categórica (Grupos):", sorted(list(set(cat_cols_anova_tab))), 
                                                     index=sorted(list(set(cat_cols_anova_tab))).index(default_cat_anova_tab) if default_cat_anova_tab in cat_cols_anova_tab else 0, 
                                                     key="anova_cat_var_cacao")
            num_var_anova_tab = st.sidebar.selectbox("Variável Numérica (Alvo):", sorted(list(set(num_cols_anova_tab))), 
                                                     index=sorted(list(set(num_cols_anova_tab))).index(default_num_anova_tab) if default_num_anova_tab in num_cols_anova_tab else 0, 
                                                     key="anova_num_var_cacao")

            if st.sidebar.button("Executar Análise ANOVA", key="run_anova_tab_cacao", help="Clique para realizar a Análise de Variância."):
                perform_anova(df_main, cat_var_anova_tab, num_var_anova_tab)

    with tab_regression: # Regressão
        st.sidebar.subheader("🛠️ Opções para Regressão Linear")
        all_cols_reg_tab = df_main.columns.tolist()
        num_cols_reg_tab = df_main.select_dtypes(include=np.number).columns.tolist()

        if not num_cols_reg_tab:
            st.warning("Nenhuma coluna numérica para ser variável alvo.")
        else:
            default_target_reg_tab = 'rating' if 'rating' in num_cols_reg_tab else num_cols_reg_tab[0]
            target_var_reg_tab = st.sidebar.selectbox("Variável Alvo (Dependente):", sorted(list(set(num_cols_reg_tab))), 
                                                      index=sorted(list(set(num_cols_reg_tab))).index(default_target_reg_tab) if default_target_reg_tab in num_cols_reg_tab else 0, 
                                                      key="reg_target_var_cacao")
            
            available_expl_tab = [col for col in all_cols_reg_tab if col != target_var_reg_tab]
            # Defaults para o dataset de cacau
            default_expl_vars_tab = ['cocoa_percent', 'company_location', 'bean_type', 'review_date']
            default_expl_vars_present_tab = [var for var in default_expl_vars_tab if var in available_expl_tab]
            
            expl_vars_reg_tab = st.sidebar.multiselect("Variáveis Explicativas (Independentes):", sorted(list(set(available_expl_tab))), 
                                                       default=default_expl_vars_present_tab, 
                                                       key="reg_expl_vars_cacao")

            log_target_reg_tab = st.sidebar.checkbox("Aplicar log na variável alvo?", value=False, key="reg_log_target_cacao") # Default False para rating
            
            # Identificar contínuas entre as explicativas selecionadas
            continuous_for_log_selection_tab = [
                col for col in expl_vars_reg_tab 
                if df_main[col].dtype in [np.number] and df_main[col].nunique() > 20
            ]
            log_continuous_vars_reg_tab = []
            if continuous_for_log_selection_tab:
                 log_continuous_vars_reg_tab = st.sidebar.multiselect("Aplicar log em quais explicativas contínuas?", 
                                                                      sorted(list(set(continuous_for_log_selection_tab))), 
                                                                      default=[v for v in ['cocoa_percent'] if v in continuous_for_log_selection_tab], 
                                                                      key="reg_log_continuous_cacao")

            if st.sidebar.button("Executar Análise de Regressão", key="run_regression_tab_cacao", help="Clique para realizar a Regressão Linear Múltipla."):
                if not expl_vars_reg_tab:
                    st.warning("Selecione ao menos uma variável explicativa.")
                else:
                    perform_regression(df_main, expl_vars_reg_tab, target_var_reg_tab, log_target_reg_tab, log_continuous_vars_reg_tab)
else:
    st.info("✨ Bem-vindo! Por favor, carregue um arquivo CSV usando a barra lateral para começar a análise, ou use o dataset padrão 'flavors_of_cacao.csv'.")

st.sidebar.markdown("---")
st.sidebar.caption("Desenvolvido com Streamlit")

