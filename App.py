import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
import scipy.stats as stats
from statsmodels.formula.api import ols
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt
import seaborn as sns

# Funcao para carregar os dados
@st.cache_data
def load_data(uploaded_file):
    try:
        df = pd.read_csv(uploaded_file)
        df.columns = df.columns.str.strip().str.replace(" ", "_").str.lower()
        return df
    except Exception as e:
        st.error(f"Erro ao carregar o arquivo: {e}")
        return None

# Funcao para limpar/preparar colunas para ANOVA
def prepare_anova_data(df, selected_var, target_var='saleprice'):
    df_anova = df[[selected_var, target_var]].copy()
    df_anova.dropna(subset=[selected_var, target_var], inplace=True)
    return df_anova

# Funcao para verificar pressupostos da ANOVA
def check_anova_assumptions(df_anova, selected_var, target_var='saleprice'):
    st.subheader(f"Verificação de Pressupostos da ANOVA para '{selected_var}'")
    
    # Ajustar modelo OLS para obter resíduos
    try:
        model_ols = ols(f'{target_var} ~ C({selected_var})', data=df_anova).fit()
        residuals = model_ols.resid
    except Exception as e:
        st.warning(f"Não foi possível ajustar o modelo OLS para {selected_var} para verificar os pressupostos: {e}")
        return False, False # Normalidade, Homocedasticidade

    # 1. Normalidade dos Resíduos (Shapiro-Wilk)
    if len(residuals) >= 3: # Shapiro-Wilk requer pelo menos 3 amostras
        shapiro_stat, shapiro_p = stats.shapiro(residuals)
        st.write(f"**Normalidade dos Resíduos (Shapiro-Wilk) para '{selected_var}':**")
        st.write(f"  - Estatística do teste: {shapiro_stat:.4f}")
        st.write(f"  - Valor-p: {shapiro_p:.4f}")
        normality_assumed = shapiro_p > 0.05
        if normality_assumed:
            st.success("Os resíduos parecem seguir uma distribuição normal.")
        else:
            st.warning("Os resíduos não parecem seguir uma distribuição normal (p <= 0.05).")
            # Visualização da distribuição dos resíduos
            fig_res_dist, ax_res_dist = plt.subplots()
            sns.histplot(residuals, kde=True, ax=ax_res_dist)
            ax_res_dist.set_title(f'Distribuição dos Resíduos ({selected_var})')
            st.pyplot(fig_res_dist)
            
            fig_qq, ax_qq = plt.subplots()
            sm.qqplot(residuals, line='s', ax=ax_qq)
            ax_qq.set_title(f'Q-Q Plot dos Resíduos ({selected_var})')
            st.pyplot(fig_qq)

    else:
        st.warning(f"Não há dados suficientes para o teste de Shapiro-Wilk para '{selected_var}' (resíduos < 3).")
        normality_assumed = False # Assume não normalidade por falta de dados para teste

    # 2. Homocedasticidade (Teste de Levene)
    groups = [group[target_var].values for name, group in df_anova.groupby(selected_var) if len(group[target_var]) > 1] # Levene requer grupos com mais de 1 amostra
    if len(groups) > 1: # Levene requer pelo menos 2 grupos
        try:
            levene_stat, levene_p = stats.levene(*groups)
            st.write(f"**Homocedasticidade (Teste de Levene) para '{selected_var}':**")
            st.write(f"  - Estatística do teste: {levene_stat:.4f}")
            st.write(f"  - Valor-p: {levene_p:.4f}")
            homoscedasticity_assumed = levene_p > 0.05
            if homoscedasticity_assumed:
                st.success("As variâncias dos grupos parecem ser homogêneas.")
            else:
                st.warning("As variâncias dos grupos não parecem ser homogêneas (p <= 0.05).")
        except ValueError as e:
            st.warning(f"Erro ao realizar o teste de Levene para '{selected_var}': {e}. Pode ser devido a grupos com uma única amostra ou variância zero.")
            homoscedasticity_assumed = False
    else:
        st.warning(f"Não há grupos suficientes para o teste de Levene para '{selected_var}'.")
        homoscedasticity_assumed = False # Assume não homocedasticidade

    return normality_assumed, homoscedasticity_assumed

# Funcao para realizar ANOVA ou Kruskal-Wallis
def perform_anova_or_kruskal(df_anova, selected_var, normality_assumed, homoscedasticity_assumed, target_var='saleprice'):
    st.subheader(f"Resultados da Análise para '{selected_var}'")
    
    groups = [group[target_var].values for name, group in df_anova.groupby(selected_var)]
    
    if normality_assumed and homoscedasticity_assumed:
        st.write("Os pressupostos da ANOVA foram atendidos. Realizando ANOVA...")
        try:
            f_stat, p_value = stats.f_oneway(*groups)
            st.write(f"**ANOVA para '{selected_var}':**")
            st.write(f"  - Estatística F: {f_stat:.4f}")
            st.write(f"  - Valor-p: {p_value:.4g}") # Usar .4g para notação científica se necessário
            if p_value < 0.05:
                st.success(f"Há uma diferença estatisticamente significativa nos preços médios de venda entre as categorias de '{selected_var}'.")
            else:
                st.info(f"Não há uma diferença estatisticamente significativa nos preços médios de venda entre as categorias de '{selected_var}'.")
        except ValueError as e:
            st.error(f"Erro ao realizar ANOVA para {selected_var}: {e}. Isso pode acontecer se um grupo tiver apenas uma observação.")

    else:
        st.warning("Os pressupostos da ANOVA não foram atendidos. Realizando Teste de Kruskal-Wallis (alternativa não paramétrica)...")
        try:
            h_stat, p_value = stats.kruskal(*groups)
            st.write(f"**Teste de Kruskal-Wallis para '{selected_var}':**")
            st.write(f"  - Estatística H: {h_stat:.4f}")
            st.write(f"  - Valor-p: {p_value:.4g}") # Usar .4g
            if p_value < 0.05:
                st.success(f"Há uma diferença estatisticamente significativa nas distribuições de preços de venda entre as categorias de '{selected_var}'.")
            else:
                st.info(f"Não há uma diferença estatisticamente significativa nas distribuições de preços de venda entre as categorias de '{selected_var}'.")
        except ValueError as e:
             st.error(f"Erro ao realizar Kruskal-Wallis para {selected_var}: {e}. Isso pode acontecer se um grupo tiver apenas uma observação.")


# --- Configuracao da Pagina ---
st.set_page_config(layout="wide", page_title="Precificação Imobiliária - ANOVA e Regressão")

# --- Titulo ---
st.title("Precificação Imobiliária com ANOVA e Regressão Linear")
st.markdown("""
Este aplicativo realiza uma análise exploratória e modelagem preditiva no Ames Housing Dataset.
Ele é baseado na Tarefa 2 de Precificação Imobiliária.
""")

# --- Upload do Arquivo ---
uploaded_file = st.sidebar.file_uploader("Carregue o arquivo AmesHousing.csv", type=["csv"])

if uploaded_file:
    df_original = load_data(uploaded_file)

    if df_original is not None:
        st.sidebar.success("Dados carregados com sucesso!")
        
        # --- Opcoes de Analise na Sidebar ---
        analysis_type = st.sidebar.radio("Escolha o tipo de análise:", 
                                         ("Visualização dos Dados", "Análise Exploratória com ANOVA", "Modelagem Preditiva com Regressão Linear"))

        # --- 0. Visualizacao dos Dados ---
        if analysis_type == "Visualização dos Dados":
            st.header("Visualização dos Dados")
            st.write("### Primeiras linhas do dataset:")
            st.dataframe(df_original.head())

            st.write("### Informações gerais do dataset:")
            st.text(df_original.info(buf=None)) # Para exibir no Streamlit

            st.write("### Estatísticas descritivas (variáveis numéricas):")
            st.dataframe(df_original.describe())
            
            st.write("### Visualização de algumas variáveis:")
            numeric_cols_for_plot = df_original.select_dtypes(include=np.number).columns.tolist()
            categorical_cols_for_plot = df_original.select_dtypes(include='object').columns.tolist()

            if numeric_cols_for_plot:
                num_var_plot = st.selectbox("Selecione uma variável numérica para plotar (Histograma):", numeric_cols_for_plot)
                if num_var_plot:
                    fig_num, ax_num = plt.subplots()
                    sns.histplot(df_original[num_var_plot], kde=True, ax=ax_num)
                    ax_num.set_title(f'Distribuição de {num_var_plot}')
                    st.pyplot(fig_num)
            
            if categorical_cols_for_plot:
                cat_var_plot = st.selectbox("Selecione uma variável categórica para plotar (Gráfico de Barras):", categorical_cols_for_plot)
                if cat_var_plot:
                    fig_cat, ax_cat = plt.subplots()
                    sns.countplot(y=df_original[cat_var_plot], ax=ax_cat, order = df_original[cat_var_plot].value_counts().index)
                    ax_cat.set_title(f'Contagem de {cat_var_plot}')
                    plt.xticks(rotation=45, ha='right')
                    st.pyplot(fig_cat)


        # --- 1. Analise Exploratoria com ANOVA ---
        elif analysis_type == "Análise Exploratória com ANOVA":
            st.header("I. Análise Exploratória e Comparativa com ANOVA")
            
            categorical_vars = df_original.select_dtypes(include='object').columns.tolist()
            if not categorical_vars:
                st.warning("Nenhuma variável categórica encontrada no dataset para análise ANOVA.")
            else:
                selected_vars_anova = st.multiselect("a) Escolha duas a três variáveis categóricas:", 
                                                     categorical_vars, 
                                                     default=categorical_vars[:2] if len(categorical_vars) >= 2 else categorical_vars)

                if selected_vars_anova:
                    for var_anova in selected_vars_anova:
                        st.markdown("---")
                        st.subheader(f"Análise para a variável: '{var_anova}'")
                        
                        df_anova_current = prepare_anova_data(df_original, var_anova)

                        if df_anova_current[var_anova].nunique() < 2:
                            st.warning(f"A variável '{var_anova}' possui menos de 2 categorias após a remoção de NaNs. Não é possível realizar ANOVA.")
                            continue
                        if df_anova_current.empty:
                            st.warning(f"Não há dados suficientes para a variável '{var_anova}' após a remoção de NaNs.")
                            continue


                        # b) Boxplot para visualização
                        st.write(f"**Boxplot: Preço de Venda por '{var_anova}'**")
                        fig_box, ax_box = plt.subplots(figsize=(10, 6))
                        sns.boxplot(x=var_anova, y='saleprice', data=df_anova_current, ax=ax_box)
                        ax_box.set_title(f'Preço de Venda por {var_anova}')
                        ax_box.set_xlabel(var_anova)
                        ax_box.set_ylabel('Preço de Venda')
                        plt.xticks(rotation=45, ha='right')
                        st.pyplot(fig_box)

                        # c) Verificar pressupostos
                        normality, homoscedasticity = check_anova_assumptions(df_anova_current, var_anova)
                        
                        # d) Aplicar ANOVA ou Kruskal-Wallis
                        perform_anova_or_kruskal(df_anova_current, var_anova, normality, homoscedasticity)

                        # e) Interpretação (texto genérico, adaptar conforme o notebook)
                        st.markdown(f"""
                        **Interpretação para '{var_anova}':**
                        - Se a ANOVA/Kruskal-Wallis indicou diferença significativa (valor-p < 0.05):
                            - A característica '{var_anova}' parece impactar o preço médio de venda.
                            - Corretores e investidores devem considerar as diferentes categorias de '{var_anova}' ao definir preços ou estratégias de investimento. Por exemplo, categorias com preços médios mais altos podem indicar maior valorização.
                        - Se não houve diferença significativa:
                            - A característica '{var_anova}', isoladamente, pode não ser um fator determinante para grandes variações no preço médio de venda neste conjunto de dados.
                        - *Observação: Esta é uma interpretação geral. A análise detalhada dos grupos específicos (via testes post-hoc, não implementados aqui) seria necessária para conclusões mais precisas sobre quais categorias diferem entre si.*
                        """)
                else:
                    st.info("Por favor, selecione pelo menos uma variável categórica para a análise ANOVA.")


        # --- 2. Modelagem Preditiva com Regressao Linear ---
        elif analysis_type == "Modelagem Preditiva com Regressão Linear":
            st.header("II. Modelagem Preditiva com Regressão Linear")

            all_cols = df_original.columns.tolist()
            potential_continuous = df_original.select_dtypes(include=np.number).columns.tolist()
            potential_categorical = df_original.select_dtypes(include='object').columns.tolist()
            
            # Remover 'saleprice' das features
            if 'saleprice' in potential_continuous:
                potential_continuous.remove('saleprice')
            if 'saleprice' in all_cols:
                 all_cols.remove('saleprice')


            st.write("a) Escolha quatro a seis variáveis explicativas (pelo menos uma contínua e uma categórica):")
            
            # Default selections baseadas no notebook, se existirem e forem válidas
            default_continuous = [col for col in ['gr_liv_area', 'overall_qual'] if col in potential_continuous]
            default_categorical = [col for col in ['garage_finish', 'kitchen_qual', 'exter_qual', 'central_air'] if col in potential_categorical]
            
            selected_continuous_reg = st.multiselect("Selecione variáveis contínuas:", 
                                                     potential_continuous, 
                                                     default=default_continuous)
            selected_categorical_reg = st.multiselect("Selecione variáveis categóricas:", 
                                                      potential_categorical, 
                                                      default=default_categorical)

            selected_features_reg = selected_continuous_reg + selected_categorical_reg

            if not selected_continuous_reg:
                st.warning("Por favor, selecione pelo menos uma variável contínua.")
            elif not selected_categorical_reg:
                st.warning("Por favor, selecione pelo menos uma variável categórica.")
            elif len(selected_features_reg) < 4 or len(selected_features_reg) > 6:
                st.warning("Por favor, selecione entre 4 e 6 variáveis explicativas no total.")
            else:
                st.success(f"Variáveis selecionadas: {', '.join(selected_features_reg)}")

                df_modelo = df_original[['saleprice'] + selected_features_reg].copy()
                df_modelo.dropna(inplace=True)

                if df_modelo.shape[0] < 2: # Necessário para a maioria das operações estatísticas
                    st.error("Não há dados suficientes após a remoção de NaNs para as variáveis selecionadas. Por favor, escolha outras variáveis.")
                else:
                    # b) Modelagem (sem log e com log)
                    model_type = st.radio("b) Escolha o tipo de modelo:", ("Linear (sem transformação)", "Log-Log (transformação logarítmica)"))

                    if model_type == "Linear (sem transformação)":
                        y = df_modelo['saleprice']
                        X_cat_dummies = pd.get_dummies(df_modelo[selected_categorical_reg], drop_first=True, dtype=float)
                        X_cont = df_modelo[selected_continuous_reg]
                        X = pd.concat([X_cont, X_cat_dummies], axis=1)
                        X = sm.add_constant(X)
                        
                        # Assegurar que todas as colunas em X são numéricas
                        for col in X.columns:
                            if X[col].dtype == 'object' or X[col].dtype.name == 'category':
                                try:
                                    X[col] = pd.to_numeric(X[col])
                                except ValueError:
                                    st.error(f"Não foi possível converter a coluna '{col}' para numérica. Verifique os dados.")
                                    X = X.drop(columns=[col]) # Remover coluna problemática

                        if X.shape[0] < X.shape[1] +1 : # Regra de ouro: n > p+1
                            st.error(f"Não há observações suficientes ({X.shape[0]}) para o número de preditores ({X.shape[1]}) após o tratamento de dummies. Por favor, selecione menos variáveis ou variáveis com menos categorias.")
                        else:
                            try:
                                modelo_fit = sm.OLS(y, X.astype(float)).fit()
                                st.subheader("Sumário do Modelo Linear (sem transformação):")
                                st.text(modelo_fit.summary())
                                y_pred = modelo_fit.predict(X)
                                residuals = modelo_fit.resid
                            except Exception as e:
                                st.error(f"Erro ao ajustar o modelo OLS: {e}")
                                modelo_fit = None
                                residuals = None
                                y_pred = None
                    
                    else: # Log-Log
                        df_log_modelo = df_modelo.copy()
                        # Aplicar log na variável dependente e nas explicativas contínuas
                        df_log_modelo['log_saleprice'] = np.log(df_log_modelo['saleprice'].replace(0, np.nan)).dropna() # Evitar log(0)
                        
                        for col in selected_continuous_reg:
                            # Checar se há valores não positivos antes de aplicar log
                            if (df_log_modelo[col] <= 0).any():
                                st.warning(f"A variável contínua '{col}' contém valores não positivos. O log não será aplicado a esta variável ou será aplicado após tratamento (e.g., adicionar uma constante pequena se apropriado, ou remover/substituir zeros/negativos). Para este app, estamos removendo linhas com valores não positivos para '{col}'.")
                                df_log_modelo = df_log_modelo[df_log_modelo[col] > 0] # Remove linhas com valores não positivos
                            if df_log_modelo[col].count() > 0: # Se ainda há dados após remoção
                                df_log_modelo[f'log_{col}'] = np.log(df_log_modelo[col])
                            else:
                                st.error(f"Após remover valores não positivos, a variável '{col}' não possui dados. Não é possível aplicar log.")
                                continue # Pula para a próxima variável
                        
                        df_log_modelo.dropna(inplace=True) # Remover NaNs gerados pelo log(0) ou log(negativo)
                        
                        if 'log_saleprice' not in df_log_modelo.columns or df_log_modelo.empty:
                            st.error("Não foi possível transformar 'saleprice' para log ou não há dados suficientes após a transformação. Verifique se há valores não positivos em 'saleprice'.")
                            modelo_fit = None
                            residuals = None
                            y_pred = None
                        else:
                            y = df_log_modelo['log_saleprice']
                            
                            X_cat_dummies_log = pd.get_dummies(df_log_modelo[selected_categorical_reg], drop_first=True, dtype=float)
                            
                            selected_log_continuous_cols = [f'log_{col}' for col in selected_continuous_reg if f'log_{col}' in df_log_modelo.columns]
                            
                            if not selected_log_continuous_cols: # Se nenhuma contínua pode ser logada
                                X_cont_log = pd.DataFrame(index=df_log_modelo.index) # DataFrame vazio se não houver colunas contínuas logadas
                            else:
                                X_cont_log = df_log_modelo[selected_log_continuous_cols]

                            X = pd.concat([X_cont_log, X_cat_dummies_log], axis=1)
                            X = sm.add_constant(X)

                            # Assegurar que todas as colunas em X são numéricas
                            for col in X.columns:
                                if X[col].dtype == 'object' or X[col].dtype.name == 'category':
                                    try:
                                        X[col] = pd.to_numeric(X[col])
                                    except ValueError:
                                        st.error(f"Não foi possível converter a coluna '{col}' para numérica no modelo log-log. Verifique os dados.")
                                        X = X.drop(columns=[col])
                            
                            if X.shape[0] < X.shape[1] +1 or y.empty:
                                st.error(f"Não há observações suficientes ({X.shape[0]} para X, {y.shape[0]} para y) para o número de preditores ({X.shape[1]}) no modelo log-log. Por favor, selecione outras variáveis ou verifique se há muitos valores não positivos nas variáveis contínuas/target.")
                                modelo_fit = None
                                residuals = None
                                y_pred = None
                            else:
                                try:
                                    modelo_fit = sm.OLS(y, X.astype(float)).fit()
                                    st.subheader("Sumário do Modelo Log-Log:")
                                    st.text(modelo_fit.summary())
                                    y_pred = modelo_fit.predict(X) # Predições na escala log
                                    residuals = modelo_fit.resid # Resíduos na escala log
                                except Exception as e:
                                    st.error(f"Erro ao ajustar o modelo OLS (Log-Log): {e}")
                                    modelo_fit = None
                                    residuals = None
                                    y_pred = None

                    # c) Diagnóstico dos pressupostos (se o modelo foi ajustado)
                    if modelo_fit is not None and residuals is not None and y_pred is not None:
                        st.subheader("c) Diagnóstico dos Pressupostos do Modelo:")
                        
                        # Linearidade (Resíduos vs. Preditos)
                        st.write("**Linearidade:** (Gráfico de Resíduos vs. Valores Preditos)")
                        fig_res, ax_res = plt.subplots()
                        ax_res.scatter(y_pred, residuals)
                        ax_res.axhline(0, color='red', linestyle='--')
                        ax_res.set_xlabel('Valores Preditos')
                        ax_res.set_ylabel('Resíduos')
                        ax_res.set_title('Resíduos vs. Valores Preditos')
                        st.pyplot(fig_res)
                        st.markdown("Idealmente, os pontos devem se distribuir aleatoriamente em torno da linha horizontal vermelha, sem padrões óbvios.")

                        # Normalidade dos Resíduos (Shapiro-Wilk e Q-Q Plot)
                        st.write("**Normalidade dos Resíduos:**")
                        if len(residuals) >=3:
                            shapiro_stat_reg, shapiro_p_reg = stats.shapiro(residuals)
                            st.write(f"  - Teste de Shapiro-Wilk (p-valor): {shapiro_p_reg:.4f}")
                            if shapiro_p_reg > 0.05:
                                st.success("Os resíduos parecem seguir uma distribuição normal.")
                            else:
                                st.warning("Os resíduos não parecem seguir uma distribuição normal (p <= 0.05).")
                        else:
                            st.warning("Não há resíduos suficientes para o teste de Shapiro-Wilk.")

                        fig_qq_reg, ax_qq_reg = plt.subplots()
                        sm.qqplot(residuals, line='s', ax=ax_qq_reg)
                        ax_qq_reg.set_title('Q-Q Plot dos Resíduos')
                        st.pyplot(fig_qq_reg)
                        st.markdown("No Q-Q plot, os pontos devem seguir aproximadamente a linha diagonal vermelha para indicar normalidade.")

                        # Homocedasticidade (Breusch-Pagan)
                        st.write("**Homocedasticidade:**")
                        try:
                            # Garantir que não há NaNs em X ou residuals antes do teste
                            X_bp = X.copy()
                            residuals_bp = residuals.copy()
                            
                            # Alinhar índices se necessário (caso X tenha sido modificado e y não)
                            common_index = y.index.intersection(X_bp.index)
                            if not common_index.equals(X_bp.index) or not common_index.equals(residuals_bp.index):
                                X_bp = X_bp.loc[common_index]
                                residuals_bp = residuals_bp.loc[common_index]
                            
                            if X_bp.empty or residuals_bp.empty:
                                 st.warning("Não há dados suficientes após alinhamento para o teste de Breusch-Pagan.")
                            else:
                                # Remover colunas com variância zero de X_bp (exceto 'const')
                                for col in X_bp.columns:
                                    if col != 'const' and X_bp[col].nunique() == 1:
                                        X_bp = X_bp.drop(columns=[col])

                                if X_bp.shape[1] == 0: # Se só sobrou a constante
                                     st.warning("Não há variáveis explicativas com variância para o teste de Breusch-Pagan.")
                                else:
                                    bp_test = sm.stats.het_breuschpagan(residuals_bp, X_bp)
                                    st.write(f"  - Teste de Breusch-Pagan (p-valor): {bp_test[1]:.4f}")
                                    if bp_test[1] > 0.05:
                                        st.success("Os resíduos parecem ser homocedásticos (variância constante).")
                                    else:
                                        st.warning("Os resíduos parecem ser heterocedásticos (variância não constante, p <= 0.05).")
                        except Exception as e_bp:
                            st.warning(f"Não foi possível realizar o teste de Breusch-Pagan: {e_bp}. Verifique se há colunas com variância zero ou outros problemas nos dados.")


                        # Multicolinearidade (VIF) - Excluir a constante para VIF
                        st.write("**Multicolinearidade (VIF):**")
                        X_vif = X.drop('const', axis=1, errors='ignore') # Ignorar erro se 'const' não existir
                        
                        # Remover colunas que foram eliminadas de X_vif se X foi modificado
                        cols_to_drop_for_vif = [col for col in X_vif.columns if col not in X.columns and col != 'const']
                        if cols_to_drop_for_vif:
                            X_vif = X_vif.drop(columns=cols_to_drop_for_vif, errors='ignore')


                        if X_vif.empty or X_vif.shape[1] == 0:
                            st.warning("Não há variáveis explicativas suficientes para calcular o VIF.")
                        else:
                            # Verificar se há colunas com variância zero (após remoção da constante)
                            cols_with_zero_variance = X_vif.columns[X_vif.nunique() == 1].tolist()
                            if cols_with_zero_variance:
                                st.warning(f"As seguintes colunas têm variância zero e serão removidas para o cálculo do VIF: {', '.join(cols_with_zero_variance)}")
                                X_vif_no_zero_var = X_vif.drop(columns=cols_with_zero_variance)
                            else:
                                X_vif_no_zero_var = X_vif

                            if X_vif_no_zero_var.empty or X_vif_no_zero_var.shape[1] == 0:
                                st.warning("Nenhuma variável explicativa com variância restante para calcular o VIF.")
                            else:
                                try:
                                    vif_data = pd.DataFrame()
                                    vif_data["Variável"] = X_vif_no_zero_var.columns
                                    vif_data["VIF"] = [variance_inflation_factor(X_vif_no_zero_var.values, i) for i in range(X_vif_no_zero_var.shape[1])]
                                    st.dataframe(vif_data)
                                    st.markdown("Valores de VIF acima de 5-10 podem indicar multicolinearidade problemática.")
                                except Exception as e_vif:
                                    st.warning(f"Não foi possível calcular o VIF: {e_vif}. Verifique a natureza das suas variáveis (e.g., perfeita colinearidade).")


                        # e) Métricas de desempenho
                        st.subheader("e) Métricas de Desempenho do Modelo Ajustado:")
                        if model_type == "Linear (sem transformação)":
                            r2_val = modelo_fit.rsquared
                            rmse_val = np.sqrt(modelo_fit.mse_resid)
                            mae_val = mean_absolute_error(y, y_pred) # y e y_pred na escala original
                        else: # Log-Log
                            # Reverter para escala original para RMSE e MAE
                            y_original_scale = np.exp(y)       # y estava em log
                            y_pred_original_scale = np.exp(y_pred) # y_pred estava em log
                            
                            # R² é geralmente reportado na escala da modelagem (log) para modelos log-log
                            # ou recalculado na escala original se o objetivo é prever valores originais.
                            # Para consistência com o notebook, vamos usar o R² do modelo log.
                            r2_val = modelo_fit.rsquared 
                            rmse_val = np.sqrt(mean_squared_error(y_original_scale, y_pred_original_scale))
                            mae_val = mean_absolute_error(y_original_scale, y_pred_original_scale)

                        st.write(f"  - R² (R-quadrado): {r2_val:.4f}")
                        st.write(f"  - RMSE (Root Mean Squared Error): {rmse_val:.2f}")
                        st.write(f"  - MAE (Mean Absolute Error): {mae_val:.2f}")
                        st.markdown(f"""
                        **Discussão do Ajuste Global e Capacidade Preditiva:**
                        - O **R²** de {r2_val:.2%} indica que aproximadamente {r2_val:.2%} da variação no preço de venda (ou log do preço de venda, se modelo log-log) é explicada pelas variáveis independentes incluídas no modelo.
                        - O **RMSE** de {rmse_val:,.2f} (em {'' if model_type == "Linear (sem transformação)" else 'unidades de log, ou convertido para escala original se métricas recalculadas'}) representa a raiz do erro quadrático médio das previsões.
                        - O **MAE** de {mae_val:,.2f} (em {'' if model_type == "Linear (sem transformação)" else 'unidades de log, ou convertido para escala original'}) indica o erro absoluto médio das previsões.
                        - *Avalie esses valores no contexto do seu problema. Um R² mais alto e RMSE/MAE mais baixos geralmente indicam um modelo melhor.*
                        """)
                        
                        # f) Interpretação dos coeficientes (texto genérico)
                        st.subheader("f) Interpretação dos Coeficientes Estimados:")
                        st.dataframe(modelo_fit.params.rename("Coeficiente"))
                        if model_type == "Log-Log":
                            st.markdown("""
                            No modelo log-log:
                            - Para variáveis explicativas contínuas (que também foram transformadas em log): um aumento de 1% na variável explicativa está associado a uma variação de X% no preço de venda, onde X é o coeficiente da variável.
                            - Para variáveis dummy (categóricas): a presença da categoria (em comparação com a categoria base omitida) está associada a uma variação de aproximadamente (exp(coeficiente) - 1) * 100% no preço de venda.
                            - Destaque as variáveis com maior impacto (maior valor absoluto do coeficiente, se padronizado, ou maior significância estatística) e se esse impacto é estatisticamente significativo (valor-p baixo).
                            """)
                        else:
                             st.markdown("""
                            No modelo linear:
                            - Para variáveis explicativas contínuas: um aumento de uma unidade na variável explicativa está associado a uma variação de X unidades no preço de venda, onde X é o coeficiente da variável.
                            - Para variáveis dummy (categóricas): a presença da categoria (em comparação com a categoria base omitida) está associada a uma diferença de X unidades no preço de venda.
                            - Destaque as variáveis com maior impacto e se esse impacto é estatisticamente significativo.
                            """)

                        # g) Recomendações práticas (texto genérico)
                        st.subheader("g) Recomendações Práticas:")
                        st.markdown("""
                        Com base no modelo estimado:
                        - **Identifique as características que mais aumentam o valor esperado do imóvel:** Variáveis com coeficientes positivos e significativos. Exemplo: "Casas com [característica X] tendem a ter, em média, um preço Y% (ou Z unidades) maior."
                        - **Identifique as características que reduzem o valor esperado:** Variáveis com coeficientes negativos e significativos.
                        - **Oriente decisões de investimento/reforma:** Focar em melhorias nas características que têm maior impacto positivo.
                        - **Auxilie na precificação:** Usar o modelo para estimar preços de imóveis com base em suas características.
                        *(Lembre-se que correlação não implica causalidade, e o modelo é uma simplificação da realidade.)*
                        """)

else:
    st.sidebar.info("Por favor, carregue o arquivo CSV para iniciar a análise.")

st.sidebar.markdown("---")
st.sidebar.markdown("Desenvolvido com base na Tarefa 2.")

