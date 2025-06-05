import streamlit as st
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Load data
df = pd.read_csv('AmesHousing.csv')

# Streamlit app title and description
st.title("Interactive Housing Data Analysis")
st.write("This app allows you to perform ANOVA and regression analysis on the Ames Housing dataset to explore factors affecting house prices.")

# Sidebar for analysis selection
st.sidebar.header("Analysis Options")
analysis_type = st.sidebar.selectbox("Choose analysis type", ["ANOVA", "Regression"])

# Display dataset preview
st.write("### Dataset Preview")
st.dataframe(df.head())

# ANOVA Analysis
if analysis_type == "ANOVA":
    st.header("ANOVA Analysis")
    st.write("Select categorical variables to test for significant differences in SalePrice.")
    cat_vars = st.multiselect(
        "Select categorical variables",
        df.select_dtypes(include='object').columns,
        help="Choose one or more categorical variables for one-way ANOVA."
    )
    if cat_vars:
        for var in cat_vars:
            st.subheader(f"ANOVA Results for {var}")
            df_anova = df[[var, 'SalePrice']].dropna()
            st.write(f"Using {len(df_anova)} samples after removing missing values")
            formula = f'SalePrice ~ C({var})'
            try:
                model = ols(formula, data=df_anova).fit()
                anova_table = sm.stats.anova_lm(model, typ=2)
                st.write(anova_table)
                # Box plot visualization
                fig, ax = plt.subplots()
                sns.boxplot(x=var, y='SalePrice', data=df_anova, ax=ax)
                ax.set_title(f'SalePrice Distribution by {var}')
                st.pyplot(fig)
            except Exception as e:
                st.error(f"Error performing ANOVA for {var}: {str(e)}")
    else:
        st.info("Please select at least one categorical variable to perform ANOVA.")

# Regression Analysis
elif analysis_type == "Regression":
    st.header("Regression Analysis")
    st.write("Select independent variables to predict SalePrice using linear regression.")
    all_vars = [col for col in df.columns if col != 'SalePrice']
    selected_vars = st.multiselect(
        "Select independent variables",
        all_vars,
        help="Choose variables to include in the regression model."
    )
    log_transform = st.checkbox("Apply log transformation to SalePrice", help="Transforms SalePrice to log scale for a log-linear model.")
    
    if selected_vars:
        df_reg = df[selected_vars + ['SalePrice']].dropna()
        st.write(f"Using {len(df_reg)} samples after removing missing values")
        # Prepare data for regression
        X = df_reg[selected_vars]
        y = df_reg['SalePrice']
        
        if log_transform:
            y = np.log(y)
            st.write("Dependent variable (SalePrice) has been log-transformed.")
        
        # Handle categorical variables with dummy encoding
        cat_vars = X.select_dtypes(include='object').columns
        if cat_vars.any():
            X = pd.get_dummies(X, columns=cat_vars, drop_first=True)
        
        # Add constant for intercept
        X = sm.add_constant(X)
        
        try:
            # Fit regression model
            model = sm.OLS(y, X).fit()
            st.write("### Regression Results")
            st.write(model.summary())
        except Exception as e:
            st.error(f"Error fitting regression model: {str(e)}")
    else:
        st.info("Please select at least one independent variable to perform regression.")

# Footer
st.write("---")
st.write("Built with Streamlit. Deployed on Streamlit Share.")

