import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy.stats import shapiro, levene, kruskal, f_oneway
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Configuração da página
st.set_page_config(
    page_title="Análise Imobiliária - ANOVA e Regressão Linear",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Função para carregar os dados
@st.cache_data
def load_data():
    df = pd.read_csv("AmesHousing.csv")
    # Padronizar nomes das colunas
    df.columns = df.columns.str.strip().str.replace(" ", "_").str.lower()
    return df

# Carregar os dados
df = load_data()

# Título principal
st.title("🏠 Dashboard de Análise Imobiliária")
st.markdown("Este dashboard interativo permite realizar análises estatísticas avançadas no conjunto de dados Ames Housing.")
st.markdown("Explore os dados, execute análises ANOVA e modelos de Regressão Linear para entender os fatores que influenciam os preços dos imóveis.")

# Abas principais
tab1, tab2, tab3 = st.tabs(["📊 Visão Geral dos Dados", "📈 Análise ANOVA", "🔍 Regressão Linear"])

# Aba 1: Visão Geral dos Dados
with tab1:
    st.header("Visão Geral dos Dados")
    
    # Exibir informações básicas sobre o dataset
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Primeiras linhas do dataset")
        st.dataframe(df.head())
    
    with col2:
        st.subheader("Informações do dataset")
        buffer = []
        buffer.append(f"**Número de registros:** {df.shape[0]}")
        buffer.append(f"**Número de colunas:** {df.shape[1]}")
        buffer.append(f"**Preço médio:** ${df['saleprice'].mean():,.2f}")
        buffer.append(f"**Preço mínimo:** ${df['saleprice'].min():,.2f}")
        buffer.append(f"**Preço máximo:** ${df['saleprice'].max():,.2f}")
        st.markdown("\n".join(buffer))
    
    # Distribuição de preços
    st.subheader("Distribuição dos Preços de Venda")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.histogram(df, x="saleprice", nbins=50, 
                          title="Histograma de Preços",
                          labels={"saleprice": "Preço de Venda ($)"},
                          color_discrete_sequence=["#3366CC"])
        fig.update_layout(bargap=0.1)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.box(df, y="saleprice", 
                    title="Boxplot de Preços",
                    labels={"saleprice": "Preço de Venda ($)"},
                    color_discrete_sequence=["#3366CC"])
        st.plotly_chart(fig, use_container_width=True)
    
    # Correlação entre variáveis numéricas e preço
    st.subheader("Correlação entre Variáveis Numéricas e Preço")
    
    # Selecionar apenas colunas numéricas
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Calcular correlações com o preço
    correlations = df[numeric_cols].corr()['saleprice'].sort_values(ascending=False)
    
    # Mostrar as 10 variáveis mais correlacionadas com o preço
    top_correlations = correlations.head(11)[1:]  # Excluir saleprice (correlação = 1)
    
    fig = px.bar(
        x=top_correlations.values,
        y=top_correlations.index,
        orientation='h',
        labels={'x': 'Correlação com Preço de Venda', 'y': 'Variável'},
        title='Top 10 Variáveis Mais Correlacionadas com o Preço',
        color=top_correlations.values,
        color_continuous_scale='Blues'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Gráfico de dispersão interativo
    st.subheader("Gráfico de Dispersão: Área x Preço")
    
    # Permitir ao usuário escolher a variável para o eixo X
    x_var = st.selectbox("Escolha a variável para o eixo X:", 
                        options=numeric_cols,
                        index=numeric_cols.index('gr_liv_area') if 'gr_liv_area' in numeric_cols else 0)
    
    # Permitir ao usuário escolher uma variável categórica para colorir os pontos
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    color_var = st.selectbox("Colorir pontos por (opcional):", 
                            options=['Nenhum'] + categorical_cols)
    
    # Criar gráfico de dispersão
    if color_var == 'Nenhum':
        fig = px.scatter(df, x=x_var, y='saleprice', 
                        title=f"{x_var.capitalize().replace('_', ' ')} vs Preço de Venda",
                        labels={x_var: x_var.capitalize().replace('_', ' '), 'saleprice': 'Preço de Venda ($)'},
                        opacity=0.7)
    else:
        fig = px.scatter(df, x=x_var, y='saleprice', color=color_var,
                        title=f"{x_var.capitalize().replace('_', ' ')} vs Preço de Venda, por {color_var.capitalize().replace('_', ' ')}",
                        labels={x_var: x_var.capitalize().replace('_', ' '), 'saleprice': 'Preço de Venda ($)', color_var: color_var.capitalize().replace('_', ' ')},
                        opacity=0.7)
    
    st.plotly_chart(fig, use_container_width=True)

# Aba 2: Análise ANOVA
with tab2:
    st.header("Análise ANOVA")
    st.markdown("""
    A Análise de Variância (ANOVA) permite verificar se há diferenças significativas entre os preços médios de venda 
    em relação às categorias de variáveis selecionadas.
    """)
    
    # Selecionar variáveis categóricas para análise
    categorical_vars = df.select_dtypes(include='object').columns.tolist()
    
    # Adicionar informação sobre número de categorias
    categorical_info = []
    for var in categorical_vars:
        n_categories = df[var].nunique()
        categorical_info.append(f"{var} ({n_categories} categorias)")
    
    # Permitir ao usuário selecionar até 3 variáveis categóricas
    selected_vars = st.multiselect(
        "Selecione até 3 variáveis categóricas para análise:",
        options=categorical_vars,
        default=['kitchen_qual', 'exter_qual'] if 'kitchen_qual' in categorical_vars and 'exter_qual' in categorical_vars else [],
        max_selections=3
    )
    
    if selected_vars:
        # Para cada variável selecionada, realizar análise ANOVA
        for var in selected_vars:
            st.subheader(f"Análise ANOVA para {var}")
            
            # Filtrar dados e remover valores ausentes
            df_var = df[[var, 'saleprice']].dropna()
            
            # Verificar se há pelo menos 2 categorias com dados suficientes
            value_counts = df_var[var].value_counts()
            valid_categories = value_counts[value_counts >= 5].index.tolist()
            
            if len(valid_categories) < 2:
                st.warning(f"A variável {var} não possui categorias suficientes para análise ANOVA (mínimo 2 categorias com pelo menos 5 observações cada).")
                continue
            
            # Filtrar apenas categorias válidas
            df_var = df_var[df_var[var].isin(valid_categories)]
            
            # Criar os grupos para ANOVA
            groups = [group["saleprice"].values for name, group in df_var.groupby(var)]
            
            # Executar ANOVA
            try:
                anova_result = f_oneway(*groups)
                
                # Exibir resultados da ANOVA
                col1, col2 = st.columns(2)
                
                with col1:
                    # Boxplot
                    fig = px.box(df_var, x=var, y="saleprice", 
                                title=f"Preço de Venda por {var}",
                                labels={var: var.capitalize().replace('_', ' '), "saleprice": "Preço de Venda ($)"},
                                color=var)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Estatísticas descritivas
                    stats_df = df_var.groupby(var)['saleprice'].agg(['count', 'mean', 'std', 'min', 'max']).reset_index()
                    stats_df = stats_df.rename(columns={
                        'count': 'Contagem', 
                        'mean': 'Média', 
                        'std': 'Desvio Padrão',
                        'min': 'Mínimo',
                        'max': 'Máximo'
                    })
                    
                    # Formatar valores monetários
                    for col in ['Média', 'Desvio Padrão', 'Mínimo', 'Máximo']:
                        stats_df[col] = stats_df[col].apply(lambda x: f"${x:,.2f}")
                    
                    st.dataframe(stats_df, use_container_width=True)
                
                # Resultados dos testes estatísticos
                st.subheader("Resultados dos Testes Estatísticos")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Resultados da ANOVA
                    st.markdown(f"**Teste ANOVA:**")
                    st.markdown(f"- Estatística F: {anova_result.statistic:.4f}")
                    st.markdown(f"- Valor-p: {anova_result.pvalue:.4f}")
                    
                    if anova_result.pvalue < 0.05:
                        st.success("✅ Há diferenças significativas entre os grupos (p < 0.05)")
                    else:
                        st.info("ℹ️ Não há diferenças significativas entre os grupos (p >= 0.05)")
                
                with col2:
                    # Verificação dos pressupostos
                    st.markdown("**Verificação dos Pressupostos:**")
                    
                    # Ajustar modelo OLS para obter resíduos
                    # Criar variáveis dummies para a variável categórica
                    df_var_dummies = pd.get_dummies(df_var[var], drop_first=True, dtype=float)
                    X = df_var_dummies.astype(float)
                    X = sm.add_constant(X)
                    y = df_var['saleprice'].astype(float)
                    
                    modelo = sm.OLS(y, X).fit()
                    residuos = modelo.resid
                    
                    # Teste de Shapiro-Wilk para normalidade
                    shapiro_test = shapiro(residuos)
                    st.markdown(f"- Shapiro-Wilk (normalidade): p-valor = {shapiro_test.pvalue:.4f}")
                    
                    # Teste de Levene para homocedasticidade
                    levene_test = levene(*groups)
                    st.markdown(f"- Levene (homocedasticidade): p-valor = {levene_test.pvalue:.4f}")
                    
                    # Teste não-paramétrico de Kruskal-Wallis
                    kruskal_test = kruskal(*groups)
                    st.markdown(f"- Kruskal-Wallis (não-paramétrico): p-valor = {kruskal_test.pvalue:.4f}")
                
                # Interpretação dos resultados
                st.subheader("Interpretação dos Resultados")
                
                # Verificar pressupostos
                pressupostos_violados = []
                if shapiro_test.pvalue < 0.05:
                    pressupostos_violados.append("normalidade dos resíduos")
                if levene_test.pvalue < 0.05:
                    pressupostos_violados.append("homocedasticidade")
                
                if pressupostos_violados:
                    st.warning(f"⚠️ Os pressupostos de {' e '.join(pressupostos_violados)} foram violados.")
                    st.markdown("""
                    Quando os pressupostos da ANOVA são violados, é recomendável utilizar testes não-paramétricos como o Kruskal-Wallis.
                    """)
                    
                    # Usar resultado do Kruskal-Wallis
                    if kruskal_test.pvalue < 0.05:
                        st.success(f"✅ O teste de Kruskal-Wallis confirma que há diferenças significativas nos preços entre as categorias de {var} (p < 0.05).")
                    else:
                        st.info(f"ℹ️ O teste de Kruskal-Wallis indica que não há diferenças significativas nos preços entre as categorias de {var} (p >= 0.05).")
                else:
                    st.success("✅ Os pressupostos da ANOVA foram atendidos.")
                    
                    # Usar resultado da ANOVA
                    if anova_result.pvalue < 0.05:
                        st.success(f"✅ A ANOVA confirma que há diferenças significativas nos preços entre as categorias de {var} (p < 0.05).")
                    else:
                        st.info(f"ℹ️ A ANOVA indica que não há diferenças significativas nos preços entre as categorias de {var} (p >= 0.05).")
                
                # Análise das médias
                if anova_result.pvalue < 0.05 or kruskal_test.pvalue < 0.05:
                    # Calcular médias por categoria
                    means = df_var.groupby(var)['saleprice'].mean().sort_values(ascending=False)
                    highest_cat = means.index[0]
                    lowest_cat = means.index[-1]
                    diff_percent = ((means[highest_cat] - means[lowest_cat]) / means[lowest_cat]) * 100
                    
                    st.markdown(f"""
                    **Análise das médias de preço:**
                    
                    - A categoria "{highest_cat}" apresenta o maior preço médio (${means[highest_cat]:,.2f})
                    - A categoria "{lowest_cat}" apresenta o menor preço médio (${means[lowest_cat]:,.2f})
                    - A diferença entre a maior e a menor média é de ${means[highest_cat] - means[lowest_cat]:,.2f} ({diff_percent:.1f}%)
                    
                    **Implicações para o mercado imobiliário:**
                    
                    Imóveis com {var.replace('_', ' ')} classificado como "{highest_cat}" tendem a ter um valor significativamente maior 
                    do que aqueles classificados como "{lowest_cat}". Esta informação pode ser utilizada por:
                    
                    - **Corretores**: Para justificar preços mais altos para imóveis com melhores características de {var.replace('_', ' ')}
                    - **Investidores**: Para identificar oportunidades de valorização ao melhorar o {var.replace('_', ' ')} de imóveis
                    - **Compradores**: Para avaliar se o preço solicitado é justo considerando o {var.replace('_', ' ')} do imóvel
                    """)
                
                st.markdown("---")
            
            except Exception as e:
                st.error(f"Erro ao realizar ANOVA para {var}: {str(e)}")
    else:
        st.info("Selecione pelo menos uma variável categórica para realizar a análise ANOVA.")

# Aba 3: Regressão Linear
with tab3:
    st.header("Modelagem com Regressão Linear")
    st.markdown("""
    A Regressão Linear permite modelar o preço de venda dos imóveis com base em múltiplas variáveis explicativas.
    Selecione as variáveis para incluir no modelo e analise os resultados.
    """)
    
    # Separar variáveis contínuas e categóricas
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    # Remover a variável dependente (saleprice) da lista de variáveis explicativas
    if 'saleprice' in numeric_cols:
        numeric_cols.remove('saleprice')
    
    # Permitir ao usuário selecionar variáveis para o modelo
    st.subheader("Seleção de Variáveis para o Modelo")
    
    col1, col2 = st.columns(2)
    
    with col1:
        selected_numeric = st.multiselect(
            "Selecione variáveis numéricas (2-4 recomendado):",
            options=numeric_cols,
            default=['gr_liv_area', 'overall_qual'] if 'gr_liv_area' in numeric_cols and 'overall_qual' in numeric_cols else []
        )
    
    with col2:
        selected_categorical = st.multiselect(
            "Selecione variáveis categóricas (1-3 recomendado):",
            options=categorical_cols,
            default=['kitchen_qual', 'central_air'] if 'kitchen_qual' in categorical_cols and 'central_air' in categorical_cols else []
        )
    
    # Opções de transformação
    st.subheader("Opções de Transformação")
    
    col1, col2 = st.columns(2)
    
    with col1:
        transform_y = st.checkbox("Aplicar transformação logarítmica na variável dependente (preço)", value=True)
    
    with col2:
        transform_x = st.checkbox("Aplicar transformação logarítmica nas variáveis numéricas", value=False)
    
    # Botão para executar a regressão
    run_regression = st.button("Executar Regressão Linear", type="primary")
    
    if run_regression:
        if not selected_numeric and not selected_categorical:
            st.warning("Selecione pelo menos uma variável explicativa para executar o modelo.")
        else:
            # Preparar os dados para o modelo
            selected_vars = selected_numeric + selected_categorical
            df_model = df[['saleprice'] + selected_vars].dropna()
            
            # Verificar se há dados suficientes
            if len(df_model) < 10:
                st.error("Dados insuficientes para executar o modelo após remover valores ausentes.")
            else:
                st.success(f"Modelo executado com {len(df_model)} observações.")
                
                # Criar variáveis dummies para variáveis categóricas
                if selected_categorical:
                    df_model = pd.get_dummies(df_model, columns=selected_categorical, drop_first=True, dtype=float)
                
                # Aplicar transformações logarítmicas se selecionado
                if transform_y:
                    df_model['log_saleprice'] = np.log(df_model['saleprice'])
                    y_var = 'log_saleprice'
                    y_label = 'Log do Preço de Venda'
                else:
                    y_var = 'saleprice'
                    y_label = 'Preço de Venda'
                
                # Transformar variáveis numéricas se selecionado
                if transform_x and selected_numeric:
                    for var in selected_numeric:
                        # Verificar se a variável tem valores positivos
                        if (df_model[var] > 0).all():
                            df_model[f'log_{var}'] = np.log(df_model[var])
                            # Substituir a variável original pela transformada na lista
                            selected_numeric = [f'log_{var}' if v == var else v for v in selected_numeric]
                
                # Preparar variáveis para o modelo
                X_vars = selected_numeric + [col for col in df_model.columns if any(cat in col for cat in selected_categorical) and col != y_var and col != 'saleprice']
                
                # Verificar se há variáveis explicativas após o processamento
                if not X_vars:
                    st.error("Não há variáveis explicativas válidas após o processamento.")
                else:
                    # Preparar X e y para o modelo
                    X = df_model[X_vars].astype(float)
                    X = sm.add_constant(X)
                    y = df_model[y_var].astype(float)
                    
                    # Ajustar o modelo
                    model = sm.OLS(y, X).fit()
                    
                    # Exibir resultados do modelo
                    st.subheader("Resultados do Modelo de Regressão Linear")
                    
                    # Métricas do modelo
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("R² Ajustado", f"{model.rsquared_adj:.4f}")
                    
                    with col2:
                        # Calcular RMSE
                        y_pred = model.predict(X)
                        rmse = np.sqrt(mean_squared_error(y, y_pred))
                        st.metric("RMSE", f"{rmse:.4f}")
                    
                    with col3:
                        # Calcular MAE
                        mae = mean_absolute_error(y, y_pred)
                        st.metric("MAE", f"{mae:.4f}")
                    
                    # Resumo do modelo
                    st.subheader("Resumo Estatístico do Modelo")
                    
                    # Exibir coeficientes em uma tabela formatada
                    coef_df = pd.DataFrame({
                        'Variável': model.params.index,
                        'Coeficiente': model.params.values,
                        'Erro Padrão': model.bse.values,
                        'Valor-t': model.tvalues.values,
                        'Valor-p': model.pvalues.values
                    })
                    
                    # Formatar p-valores com asteriscos para significância
                    def format_pvalue(p):
                        if p < 0.001:
                            return f"{p:.4f} ***"
                        elif p < 0.01:
                            return f"{p:.4f} **"
                        elif p < 0.05:
                            return f"{p:.4f} *"
                        else:
                            return f"{p:.4f}"
                    
                    coef_df['Valor-p'] = coef_df['Valor-p'].apply(format_pvalue)
                    
                    st.dataframe(coef_df, use_container_width=True)
                    st.caption("Significância: * p<0.05, ** p<0.01, *** p<0.001")
                    
                    # Diagnóstico dos pressupostos
                    st.subheader("Diagnóstico dos Pressupostos do Modelo")
                    
                    # Calcular resíduos
                    residuos = model.resid
                    residuos_padronizados = model.get_influence().resid_studentized_internal
                    
                    # Criar gráficos de diagnóstico
                    fig = make_subplots(
                        rows=2, cols=2,
                        subplot_titles=(
                            "Resíduos vs. Valores Ajustados", 
                            "QQ-Plot dos Resíduos",
                            "Histograma dos Resíduos",
                            "Resíduos Padronizados vs. Valores Ajustados"
                        )
                    )
                    
                    # 1. Resíduos vs. Valores Ajustados
                    fig.add_trace(
                        go.Scatter(
                            x=model.fittedvalues,
                            y=residuos,
                            mode='markers',
                            marker=dict(color='blue', opacity=0.6),
                            name='Resíduos'
                        ),
                        row=1, col=1
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=[min(model.fittedvalues), max(model.fittedvalues)],
                            y=[0, 0],
                            mode='lines',
                            line=dict(color='red', dash='dash'),
                            name='Linha Zero'
                        ),
                        row=1, col=1
                    )
                    
                    # 2. QQ-Plot
                    from scipy.stats import probplot
                    qq_x, qq_y = probplot(residuos, dist="norm", fit=False)
                    fig.add_trace(
                        go.Scatter(
                            x=qq_x,
                            y=qq_y,
                            mode='markers',
                            marker=dict(color='blue', opacity=0.6),
                            name='QQ Plot'
                        ),
                        row=1, col=2
                    )
                    # Linha de referência para QQ-Plot
                    qq_line_x = np.linspace(min(qq_x), max(qq_x), 100)
                    qq_line_y = qq_line_x * np.std(residuos) + np.mean(residuos)
                    fig.add_trace(
                        go.Scatter(
                            x=qq_line_x,
                            y=qq_line_y,
                            mode='lines',
                            line=dict(color='red', dash='dash'),
                            name='Linha de Referência'
                        ),
                        row=1, col=2
                    )
                    
                    # 3. Histograma dos Resíduos
                    fig.add_trace(
                        go.Histogram(
                            x=residuos,
                            nbinsx=30,
                            marker=dict(color='blue', opacity=0.6),
                            name='Histograma'
                        ),
                        row=2, col=1
                    )
                    
                    # 4. Resíduos Padronizados vs. Valores Ajustados
                    fig.add_trace(
                        go.Scatter(
                            x=model.fittedvalues,
                            y=residuos_padronizados,
                            mode='markers',
                            marker=dict(color='blue', opacity=0.6),
                            name='Resíduos Padronizados'
                        ),
                        row=2, col=2
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=[min(model.fittedvalues), max(model.fittedvalues)],
                            y=[0, 0],
                            mode='lines',
                            line=dict(color='red', dash='dash'),
                            name='Linha Zero'
                        ),
                        row=2, col=2
                    )
                    
                    # Atualizar layout
                    fig.update_layout(
                        height=800,
                        showlegend=False,
                        title_text="Gráficos de Diagnóstico do Modelo"
                    )
                    
                    # Atualizar eixos
                    fig.update_xaxes(title_text="Valores Ajustados", row=1, col=1)
                    fig.update_yaxes(title_text="Resíduos", row=1, col=1)
                    
                    fig.update_xaxes(title_text="Quantis Teóricos", row=1, col=2)
                    fig.update_yaxes(title_text="Quantis Amostrais", row=1, col=2)
                    
                    fig.update_xaxes(title_text="Resíduos", row=2, col=1)
                    fig.update_yaxes(title_text="Frequência", row=2, col=1)
                    
                    fig.update_xaxes(title_text="Valores Ajustados", row=2, col=2)
                    fig.update_yaxes(title_text="Resíduos Padronizados", row=2, col=2)
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Testes estatísticos para os pressupostos
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Teste de Normalidade dos Resíduos")
                        shapiro_test = shapiro(residuos)
                        st.markdown(f"**Teste de Shapiro-Wilk:**")
                        st.markdown(f"- Estatística W: {shapiro_test.statistic:.4f}")
                        st.markdown(f"- Valor-p: {shapiro_test.pvalue:.4f}")
                        
                        if shapiro_test.pvalue < 0.05:
                            st.warning("⚠️ Os resíduos não seguem uma distribuição normal (p < 0.05).")
                        else:
                            st.success("✅ Os resíduos seguem uma distribuição normal (p >= 0.05).")
                    
                    with col2:
                        st.subheader("Teste de Homocedasticidade")
                        bp_test = het_breuschpagan(residuos, X)
                        st.markdown(f"**Teste de Breusch-Pagan:**")
                        st.markdown(f"- Estatística LM: {bp_test[0]:.4f}")
                        st.markdown(f"- Valor-p: {bp_test[1]:.4f}")
                        
                        if bp_test[1] < 0.05:
                            st.warning("⚠️ Há evidência de heterocedasticidade (p < 0.05).")
                        else:
                            st.success("✅ Não há evidência de heterocedasticidade (p >= 0.05).")
                    
                    # Multicolinearidade (VIF)
                    st.subheader("Análise de Multicolinearidade (VIF)")
                    
                    # Calcular VIF para cada variável
                    vif_data = pd.DataFrame()
                    vif_data["Variável"] = X.columns
                    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
                    
                    # Formatar VIF
                    def format_vif(vif):
                        if vif > 10:
                            return f"{vif:.2f} ⚠️"
                        elif vif > 5:
                            return f"{vif:.2f} ⚠️"
                        else:
                            return f"{vif:.2f} ✅"
                    
                    vif_data["VIF Formatado"] = vif_data["VIF"].apply(format_vif)
                    
                    # Exibir tabela VIF
                    st.dataframe(vif_data[["Variável", "VIF Formatado"]], use_container_width=True)
                    st.caption("VIF > 10: Multicolinearidade alta, VIF > 5: Multicolinearidade moderada, VIF < 5: Multicolinearidade baixa")
                    
                    # Interpretação dos coeficientes
                    st.subheader("Interpretação dos Coeficientes")
                    
                    # Filtrar apenas coeficientes significativos (p < 0.05)
                    sig_coefs = coef_df[coef_df['Valor-p'].str.contains('\*')]
                    
                    if len(sig_coefs) > 1:  # Ignorar a constante
                        st.markdown("**Variáveis com impacto significativo no preço:**")
                        
                        for _, row in sig_coefs.iterrows():
                            var_name = row['Variável']
                            coef = row['Coeficiente']
                            
                            # Pular a constante
                            if var_name == 'const':
                                continue
                            
                            # Interpretar coeficientes com base no tipo de modelo
                            if transform_y and transform_x and 'log_' in var_name and var_name != 'log_saleprice':
                                # Modelo log-log
                                st.markdown(f"- **{var_name.replace('log_', '').replace('_', ' ').title()}**: Um aumento de 1% nesta variável está associado a uma variação de {coef:.4f}% no preço do imóvel, mantendo as demais variáveis constantes.")
                            
                            elif transform_y and 'log_' not in var_name:
                                # Modelo log-linear
                                percent_change = (np.exp(coef) - 1) * 100
                                if '_' in var_name and any(cat in var_name for cat in selected_categorical):
                                    # Variável dummy
                                    orig_var = var_name.split('_')[0]
                                    category = '_'.join(var_name.split('_')[1:])
                                    st.markdown(f"- **{orig_var.replace('_', ' ').title()} = {category}**: Imóveis com esta característica têm, em média, preços {percent_change:.2f}% {'maiores' if percent_change > 0 else 'menores'} em comparação com a categoria base.")
                                else:
                                    # Variável contínua
                                    st.markdown(f"- **{var_name.replace('_', ' ').title()}**: Um aumento de uma unidade nesta variável está associado a uma variação de {percent_change:.2f}% no preço do imóvel, mantendo as demais variáveis constantes.")
                            
                            elif not transform_y and transform_x and 'log_' in var_name:
                                # Modelo linear-log
                                if var_name != 'const':
                                    st.markdown(f"- **{var_name.replace('log_', '').replace('_', ' ').title()}**: Um aumento de 1% nesta variável está associado a uma variação de ${coef/100:.2f} no preço do imóvel, mantendo as demais variáveis constantes.")
                            
                            else:
                                # Modelo linear
                                if '_' in var_name and any(cat in var_name for cat in selected_categorical):
                                    # Variável dummy
                                    orig_var = var_name.split('_')[0]
                                    category = '_'.join(var_name.split('_')[1:])
                                    st.markdown(f"- **{orig_var.replace('_', ' ').title()} = {category}**: Imóveis com esta característica têm, em média, preços ${coef:,.2f} {'maiores' if coef > 0 else 'menores'} em comparação com a categoria base.")
                                else:
                                    # Variável contínua
                                    st.markdown(f"- **{var_name.replace('_', ' ').title()}**: Um aumento de uma unidade nesta variável está associado a uma variação de ${coef:,.2f} no preço do imóvel, mantendo as demais variáveis constantes.")
                    
                    # Recomendações práticas
                    st.subheader("Recomendações Práticas")
                    
                    # Verificar qualidade do modelo
                    if model.rsquared_adj < 0.3:
                        model_quality = "baixo poder explicativo"
                    elif model.rsquared_adj < 0.6:
                        model_quality = "poder explicativo moderado"
                    else:
                        model_quality = "alto poder explicativo"
                    
                    st.markdown(f"""
                    **Com base no modelo de regressão ajustado (R² Ajustado = {model.rsquared_adj:.4f}, {model_quality}):**
                    
                    1. **Para Investidores e Proprietários:**
                    """)
                    
                    # Identificar as variáveis mais impactantes (maiores coeficientes absolutos)
                    if len(sig_coefs) > 1:
                        # Ignorar a constante
                        sig_coefs_no_const = sig_coefs[sig_coefs['Variável'] != 'const']
                        
                        if not sig_coefs_no_const.empty:
                            # Ordenar por valor absoluto do coeficiente
                            sig_coefs_no_const['Abs_Coef'] = sig_coefs_no_const['Coeficiente'].abs()
                            top_vars = sig_coefs_no_const.sort_values('Abs_Coef', ascending=False).head(3)
                            
                            recommendations = []
                            
                            for _, row in top_vars.iterrows():
                                var_name = row['Variável']
                                coef = row['Coeficiente']
                                
                                # Variáveis categóricas (dummies)
                                if '_' in var_name and any(cat in var_name for cat in selected_categorical):
                                    orig_var = var_name.split('_')[0]
                                    category = '_'.join(var_name.split('_')[1:])
                                    
                                    if coef > 0:
                                        recommendations.append(f"- Priorize imóveis com {orig_var.replace('_', ' ')} = {category}, pois esta característica está associada a um aumento significativo no valor do imóvel.")
                                    else:
                                        recommendations.append(f"- Evite imóveis com {orig_var.replace('_', ' ')} = {category}, ou considere melhorar esta característica, pois está associada a uma redução significativa no valor do imóvel.")
                                
                                # Variáveis contínuas
                                elif 'log_' in var_name:
                                    clean_name = var_name.replace('log_', '').replace('_', ' ')
                                    if coef > 0:
                                        recommendations.append(f"- Priorize imóveis com maior {clean_name}, pois esta característica está fortemente associada a um aumento no valor do imóvel.")
                                    else:
                                        recommendations.append(f"- Considere que imóveis com maior {clean_name} podem não representar o melhor investimento, pois esta característica está associada a uma redução no valor do imóvel.")
                                else:
                                    clean_name = var_name.replace('_', ' ')
                                    if coef > 0:
                                        recommendations.append(f"- Priorize imóveis com maior {clean_name}, pois esta característica está fortemente associada a um aumento no valor do imóvel.")
                                    else:
                                        recommendations.append(f"- Considere que imóveis com maior {clean_name} podem não representar o melhor investimento, pois esta característica está associada a uma redução no valor do imóvel.")
                            
                            for rec in recommendations:
                                st.markdown(rec)
                    
                    st.markdown("""
                    2. **Para Corretores e Vendedores:**
                    """)
                    
                    if len(sig_coefs) > 1:
                        # Gerar recomendações para corretores
                        recommendations = []
                        
                        for _, row in sig_coefs_no_const.sort_values('Abs_Coef', ascending=False).head(3).iterrows():
                            var_name = row['Variável']
                            coef = row['Coeficiente']
                            
                            # Variáveis categóricas (dummies)
                            if '_' in var_name and any(cat in var_name for cat in selected_categorical):
                                orig_var = var_name.split('_')[0]
                                category = '_'.join(var_name.split('_')[1:])
                                
                                if coef > 0:
                                    recommendations.append(f"- Destaque a característica {orig_var.replace('_', ' ')} = {category} em anúncios e apresentações, pois está associada a um aumento significativo no valor do imóvel.")
                                else:
                                    recommendations.append(f"- Considere sugerir melhorias na característica {orig_var.replace('_', ' ')}, pois a categoria {category} está associada a uma redução significativa no valor do imóvel.")
                            
                            # Variáveis contínuas
                            elif 'log_' in var_name:
                                clean_name = var_name.replace('log_', '').replace('_', ' ')
                                if coef > 0:
                                    recommendations.append(f"- Enfatize o valor de {clean_name} em anúncios e apresentações, pois esta característica está fortemente associada a um aumento no valor do imóvel.")
                                else:
                                    recommendations.append(f"- Destaque outros atributos positivos do imóvel para compensar possíveis limitações em {clean_name}, pois esta característica está associada a uma redução no valor do imóvel.")
                            else:
                                clean_name = var_name.replace('_', ' ')
                                if coef > 0:
                                    recommendations.append(f"- Enfatize o valor de {clean_name} em anúncios e apresentações, pois esta característica está fortemente associada a um aumento no valor do imóvel.")
                                else:
                                    recommendations.append(f"- Destaque outros atributos positivos do imóvel para compensar possíveis limitações em {clean_name}, pois esta característica está associada a uma redução no valor do imóvel.")
                        
                        for rec in recommendations:
                            st.markdown(rec)
                    
                    # Avaliação final do modelo
                    st.subheader("Avaliação Final do Modelo")
                    
                    # Verificar pressupostos
                    pressupostos_violados = []
                    if shapiro_test.pvalue < 0.05:
                        pressupostos_violados.append("normalidade dos resíduos")
                    if bp_test[1] < 0.05:
                        pressupostos_violados.append("homocedasticidade")
                    if (vif_data["VIF"] > 10).any():
                        pressupostos_violados.append("ausência de multicolinearidade")
                    
                    if pressupostos_violados:
                        st.warning(f"""
                        ⚠️ **Limitações do Modelo:**
                        
                        O modelo atual viola os pressupostos de {' e '.join(pressupostos_violados)}, o que pode afetar a confiabilidade das estimativas e inferências.
                        
                        **Sugestões de melhoria:**
                        - Considere aplicar transformações adicionais nas variáveis
                        - Remova variáveis com alta multicolinearidade
                        - Explore modelos alternativos, como modelos não-lineares ou técnicas de machine learning
                        """)
                    else:
                        st.success(f"""
                        ✅ **Modelo Robusto:**
                        
                        O modelo atende aos pressupostos estatísticos e apresenta um R² Ajustado de {model.rsquared_adj:.4f}, 
                        indicando que {model.rsquared_adj*100:.1f}% da variação nos preços dos imóveis é explicada pelas variáveis selecionadas.
                        
                        As estimativas dos coeficientes são confiáveis e podem ser utilizadas para tomada de decisão no mercado imobiliário.
                        """)

# Rodapé
st.markdown("---")
st.markdown("""
<div style="text-align: center">
    <p>Dashboard desenvolvido para análise imobiliária com ANOVA e Regressão Linear</p>
    <p>Universidade de Brasília - UnB | Departamento de Engenharia de Produção</p>
</div>
""", unsafe_allow_html=True)
