#!/usr/bin/env python
# coding: utf-8

# # Activity: Hypothesis testing with Python

# ## **Introduction**
# 

# As you've been learning, analysis of variance (commonly called ANOVA) is a group of statistical techniques that test the difference of means among three or more groups. It's a powerful tool for determining whether population means are different across groups and for answering a wide range of business questions.
# 
# In this activity, you are a data professional working with historical marketing promotion data. You will use the data to run a one-way ANOVA and a post hoc ANOVA test. Then, you will communicate your results to stakeholders. These experiences will help you make more confident recommendations in a professional setting. 
# 
# In your dataset, each row corresponds to an independent marketing promotion, where your business uses TV, social media, radio, and influencer promotions to increase sales. You have previously provided insights about how different promotion types affect sales; now stakeholders want to know if sales are significantly different among various TV and influencer promotion types.
# 
# To address this request, a one-way ANOVA test will enable you to determine if there is a statistically significant difference in sales among groups. This includes:
# * Using plots and descriptive statistics to select a categorical independent variable
# * Creating and fitting a linear regression model with the selected categorical independent variable
# * Checking model assumptions
# * Performing and interpreting a one-way ANOVA test
# * Comparing pairs of groups using an ANOVA post hoc test
# * Interpreting model outputs and communicating the results to nontechnical stakeholders

# ## **Step 1: Imports** 
# 

# Import pandas, pyplot from matplotlib, seaborn, api from statsmodels, ols from statsmodels.formula.api, and pairwise_tukeyhsd from statsmodels.stats.multicomp.

# In[1]:


# Import libraries and packages.

### YOUR CODE HERE ### 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd


# `Pandas` was used to load the dataset `marketing_sales_data.csv` as `data`, now display the first five rows. The variables in the dataset have been adjusted to suit the objectives of this lab. As shown in this cell, the dataset has been automatically loaded in for you. You do not need to download the .csv file, or provide more code, in order to access the dataset and proceed with this lab. Please continue with this activity by completing the following instructions.

# In[2]:


# RUN THIS CELL TO IMPORT YOUR DATA.

### YOUR CODE HERE ### 
data = pd.read_csv('marketing_sales_data.csv')

# Display the first five rows.

### YOUR CODE HERE ### 

data.head(5)


# The features in the data are:
# * TV promotion budget (in Low, Medium, and High categories)
# * Social media promotion budget (in millions of dollars)
# * Radio promotion budget (in millions of dollars)
# * Sales (in millions of dollars)
# * Influencer size (in Mega, Macro, Nano, and Micro categories)

# **Question:** Why is it useful to perform exploratory data analysis before constructing a linear regression model?

# [Write your response here. Double-click (or enter) to edit.]

# ## **Step 2: Data exploration** 
# 

# First, use a boxplot to determine how `Sales` vary based on the `TV` promotion budget category.

# In[3]:


# Create a boxplot with TV and Sales.

### YOUR CODE HERE ### 
sns.boxplot(x = "TV", y = "Sales", data = data);


# <details>
# <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# There is a function in the `seaborn` library that creates a boxplot showing the distribution of a variable across multiple groups.
# 
# </details>

# <details>
# <summary><h4><strong>Hint 2</strong></h4></summary>
# 
# Use the `boxplot()` function from `seaborn`.
# 
# </details>

# <details>
# <summary><h4><strong>Hint 3</strong></h4></summary>
# 
# Use `TV` as the `x` argument, `Sales` as the `y` argument, and `data` as the `data` argument.
# 
# </details>

# **Question:** Is there variation in `Sales` based off the `TV` promotion budget?

# [Write your response here. Double-click (or enter) to edit.]

# Now, use a boxplot to determine how `Sales` vary based on the `Influencer` size category.

# In[4]:


# Create a boxplot with Influencer and Sales.

### YOUR CODE HERE ### 

sns.boxplot(x = "Influencer", y = "Sales", data = data);


# **Question:** Is there variation in `Sales` based off the `Influencer` size?

# [Write your response here. Double-click (or enter) to edit.]

# ### Remove missing data
# 
# You may recall from prior labs that this dataset contains rows with missing values. To correct this, drop these rows. Then, confirm the data contains no missing values.

# In[5]:


# Drop rows that contain missing data and update the DataFrame.

### YOUR CODE HERE ### 

data = data.dropna(axis=0)


# Confirm the data contain no missing values.

### YOUR CODE HERE ### 

data.isnull().sum(axis=0)


# <details>
# <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# There is a `pandas` function that removes missing values.
# 
# </details>

# <details>
# <summary><h4><strong>Hint 2</strong></h4></summary>
# 
# The `dropna()` function removes missing values from an object (e.g., DataFrame).
# 
# </details>

# <details>
# <summary><h4><strong>Hint 3</strong></h4></summary>
# 
# Verify the data is updated properly after the rows containing missing data are dropped.
# 
# </details>

# ## **Step 3: Model building** 
# 

# Fit a linear regression model that predicts `Sales` using one of the independent categorical variables in `data`. Refer to your previous code for defining and fitting a linear regression model.

# In[6]:


# Define the OLS formula.

### YOUR CODE HERE ### 
ols_formula = 'Sales ~ C(TV)'

# Create an OLS model.

### YOUR CODE HERE ### 

OLS = ols(formula = ols_formula, data = data)

# Fit the model.

### YOUR CODE HERE ### 

model = OLS.fit()

# Save the results summary.

### YOUR CODE HERE ### 

model_results = model.summary()

# Display the model results.

### YOUR CODE HERE ### 

model_results


# <details>
# <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# Refer to code you've written to fit linear regression models.
# 
# </details>

# <details>
# <summary><h4><strong>Hint 2</strong></h4></summary>
# 
# Use the `ols()` function from `statsmodels.formula.api`, which creates a model from a formula and DataFrame, to create an OLS model.
# 
# </details>
# 

# <details>
# <summary><h4><strong>Hint 3</strong></h4></summary>
# 
# Use `C()` around the variable name in the ols formula to indicate a variable is categorical.
#     
# Be sure the variable string names exactly match the column names in `data`.
# 
# </details>

# **Question:** Which categorical variable did you choose for the model? Why?

# [Write your response here. Double-click (or enter) to edit.]

# ### Check model assumptions

# Now, check the four linear regression assumptions are upheld for your model.

# **Question:** Is the linearity assumption met?

# [Write your response here. Double-click (or enter) to edit.]

# The independent observation assumption states that each observation in the dataset is independent. As each marketing promotion (row) is independent from one another, the independence assumption is not violated.

# Next, verify that the normality assumption is upheld for the model.

# In[7]:


# Calculate the residuals.

### YOUR CODE HERE ### 
residuals = model.resid

# Create a 1x2 plot figure.
fig, axes = plt.subplots(1, 2, figsize = (8,4))

# Create a histogram with the residuals.

### YOUR CODE HERE ### 

sns.histplot(residuals, ax=axes[0])

# Set the x label of the residual plot.
axes[0].set_xlabel("Residual Value")

# Set the title of the residual plot.
axes[0].set_title("Histogram of Residuals")

# Create a QQ plot of the residuals.

### YOUR CODE HERE ### 

sm.qqplot(residuals, line='s',ax = axes[1])

# Set the title of the QQ plot.
axes[1].set_title("Normal QQ Plot")

# Use matplotlib's tight_layout() function to add space between plots for a cleaner appearance.
plt.tight_layout()

# Show the plot.
plt.show()


# <details>
# <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# Access the residuals from the fit model object.
# 
# </details>

# <details>
# <summary><h4><strong>Hint 2</strong></h4></summary>
# 
# Use `model.resid` to get the residuals from a fit model called `model`.
# 
# </details>

# <details>
# <summary><h4><strong>Hint 3</strong></h4></summary>
# 
# For the histogram, pass the residuals as the first argument in the `seaborn` `histplot()` function.
#     
# For the QQ-plot, pass the residuals as the first argument in the `statsmodels` `qqplot()` function.
# 
# </details>

# **Question:** Is the normality assumption met?

# [Write your response here. Double-click (or enter) to edit.]

# Now, verify the constant variance (homoscedasticity) assumption is met for this model.

# In[8]:


# Create a scatter plot with the fitted values from the model and the residuals.

### YOUR CODE HERE ### 


fig = sns.scatterplot(x = model.fittedvalues, y = model.resid)

# Set the x axis label
fig.set_xlabel("Fitted Values")

# Set the y axis label
fig.set_ylabel("Residuals")

# Set the title
fig.set_title("Fitted Values v. Residuals")

# Add a line at y = 0 to visualize the variance of residuals above and below 0.

### YOUR CODE HERE ### 

fig.axhline(0)

# Show the plot
plt.show()


# <details>
# <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# Access the fitted values from the model object fit earlier.
# 
# </details>

# <details>
# <summary><h4><strong>Hint 2</strong></h4></summary>
# 
# Use `model.fittedvalues` to get the fitted values from the fit model called `model`.
# 
# </details>

# <details>
# <summary><h4><strong>Hint 3</strong></h4></summary>
# 
# 
# Call the `scatterplot()` function from the `seaborn` library and pass in the fitted values and residuals.
#     
# Add a line to a figure using the `axline()` function.
# 
# </details>

# **Question:** Is the constant variance (homoscedasticity) assumption met?

# [Write your response here. Double-click (or enter) to edit.]

# ## **Step 4: Results and evaluation** 

# First, display the OLS regression results.

# In[9]:


# Display the model results summary.

### YOUR CODE HERE ### 
model_results


# **Question:** What is your interpretation of the model's R-squared?

# [Write your response here. Double-click (or enter) to edit.]

# **Question:** What is your intepretation of the coefficient estimates? Are the coefficients statistically significant?

# [Write your response here. Double-click (or enter) to edit.]

# **Question:** Do you think your model could be improved? Why or why not? How?

# [Write your response here. Double-click (or enter) to edit.]

# ### Perform a one-way ANOVA test
# 
# With the model fit, run a one-way ANOVA test to determine whether there is a statistically significant difference in `Sales` among groups. 

# In[10]:


# Create an one-way ANOVA table for the fit model.

### YOUR CODE HERE ### 
sm.stats.anova_lm(model, typ=2)


# <details>
# <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# Review what you've learned about how to perform a one-way ANOVA test.
# 
# </details>

# <details>
# <summary><h4><strong>Hint 2</strong></h4></summary>
# 
# There is a function in `statsmodels.api` (i.e. `sm`) that peforms an ANOVA test for a fit linear model.
# 
# </details>

# <details>
# <summary><h4><strong>Hint 3</strong></h4></summary>
# 
# Use the `anova_lm()` function from `sm.stats`. Specify the type of ANOVA test (for example, one-way or two-way), using the `typ` parameter.
#    
# 
# </details>

# **Question:** What are the null and alternative hypotheses for the ANOVA test?

# [Write your response here. Double-click (or enter) to edit.]

# **Question:** What is your conclusion from the one-way ANOVA test?

# [Write your response here. Double-click (or enter) to edit.]

# **Question:** What did the ANOVA test tell you?

# [Write your response here. Double-click (or enter) to edit.]

# ### Perform an ANOVA post hoc test
# 
# If you have significant results from the one-way ANOVA test, you can apply ANOVA post hoc tests such as the Tukey’s HSD post hoc test. 
# 
# Run the Tukey’s HSD post hoc test to compare if there is a significant difference between each pair of categories for TV.

# In[11]:


# Perform the Tukey's HSD post hoc test.

### YOUR CODE HERE ### 

tukey_oneway = pairwise_tukeyhsd(endog = data["Sales"], groups = data["TV"])

# Display the results
tukey_oneway.summary()


# <details>
# <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# Review what you've learned about how to perform a Tukey's HSD post hoc test.
# 
# </details>

# <details>
# <summary><h4><strong>Hint 2</strong></h4></summary>
# 
# Use the `pairwise_tukeyhsd()` function from `statsmodels.stats.multicomp`.
# 
# </details>

# <details>
# <summary><h4><strong>Hint 3</strong></h4></summary>
# 
# The `endog` argument in `pairwise_tukeyhsd` indicates which variable is being compared across groups (i.e., `Sales`). The `groups` argument in `pairwise_tukeyhsd` tells the function which variable holds the group you’re interested in reviewing.
# 
# </details>

# **Question:** What is your interpretation of the Tukey HSD test?

# [Write your response here. Double-click (or enter) to edit.]

# **Question:** What did the post hoc tell you?**

# [Write your response here. Double-click (or enter) to edit.]

# ## **Considerations**
# 
# **What are some key takeaways that you learned during this lab?**
# 
# [Write your response here. Double-click (or enter) to edit.]
# 
# 
# **What summary would you provide to stakeholders? Consider the statistical significance of key relationships and differences in distribution.**
# 
# [Write your response here. Double-click (or enter) to edit.]
# 

# #### **Reference**
# [Saragih, H.S. *Dummy Marketing and Sales Data*](https://www.kaggle.com/datasets/harrimansaragih/dummy-advertising-and-sales-data)

# **Congratulations!** You've completed this lab. However, you may not notice a green check mark next to this item on Coursera's platform. Please continue your progress regardless of the check mark. Just click on the "save" icon at the top of this notebook to ensure your work has been logged.
