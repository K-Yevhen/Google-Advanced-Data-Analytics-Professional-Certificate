#!/usr/bin/env python
# coding: utf-8

# # **TikTok Project**
# **Course 4 - The Power of Statistics**

# You are a data professional at TikTok. The current project is reaching its midpoint; a project proposal, Python coding work, and exploratory data analysis have all been completed.
# 
# The team has reviewed the results of the exploratory data analysis and the previous executive summary the team prepared. You received an email from Orion Rainier, Data Scientist at TikTok, with your next assignment: determine and conduct the necessary hypothesis tests and statistical analysis for the TikTok classification project.
# 
# A notebook was structured and prepared to help you in this project. Please complete the following questions.

# 
# # **Course 4 End-of-course project: Data exploration and hypothesis testing**
# 
# In this activity, you will explore the data provided and conduct hypothesis testing.
# <br/>
# 
# **The purpose** of this project is to demostrate knowledge of how to prepare, create, and analyze hypothesis tests.
# 
# **The goal** is to apply descriptive and inferential statistics, probability distributions, and hypothesis testing in Python.
# <br/>
# 
# *This activity has three parts:*
# 
# **Part 1:** Imports and data loading
# * What data packages will be necessary for hypothesis testing?
# 
# **Part 2:** Conduct hypothesis testing
# * How will descriptive statistics help you analyze your data?
# 
# * How will you formulate your null hypothesis and alternative hypothesis?
# 
# **Part 3:** Communicate insights with stakeholders
# 
# * What key business insight(s) emerge from your hypothesis test?
# 
# * What business recommendations do you propose based on your results?
# 
# <br/>
# 
# Follow the instructions and answer the questions below to complete the activity. Then, complete an executive summary using the questions listed on the PACE Strategy Document.
# 
# Be sure to complete this activity before moving on. The next course item will provide you with a completed exemplar to compare to your own work.
# 
# 

# # **Data exploration and hypothesis testing**

# <img src="images/Pace.png" width="100" height="100" align=left>
# 
# # **PACE stages**

# Throughout these project notebooks, you'll see references to the problem-solving framework PACE. The following notebook components are labeled with the respective PACE stage: Plan, Analyze, Construct, and Execute.

# <img src="images/Plan.png" width="100" height="100" align=left>
# 
# 
# ## **PACE: Plan**
# 
# Consider the questions in your PACE Strategy Document and those below to craft your response.
# 
# 1. What is your research question for this data project? Later on, you will need to formulate the null and alternative hypotheses as the first step of your hypothesis test. Consider your research question now, at the start of this task.

# ==> ENTER YOUR RESPONSE HERE

# *Complete the following steps to perform statistical analysis of your data:*

# ### **Task 1. Imports and Data Loading**

# Import packages and libraries needed to compute descriptive statistics and conduct a hypothesis test.

# <details>
#   <summary><h4><strong>Hint:</strong></h4></summary>
# 
# Be sure to import `pandas`, `numpy`, `matplotlib.pyplot`, `seaborn`, and `scipy`.
# 
# </details>

# In[1]:


# Import packages for data manipulation
### YOUR CODE HERE ###
import numpy as np
import pandas as pd


# Import packages for data visualization
### YOUR CODE HERE ###
import seaborn as sns
import matplotlib.pyplot as plt

# Import packages for statistical analysis/hypothesis testing
### YOUR CODE HERE ###
from scipy import stats


# Load the dataset.
# 
# **Note:** As shown in this cell, the dataset has been automatically loaded in for you. You do not need to download the .csv file, or provide more code, in order to access the dataset and proceed with this lab. Please continue with this activity by completing the following instructions.

# In[2]:


# Load dataset into dataframe
data = pd.read_csv("tiktok_dataset.csv")


# <img src="images/Analyze.png" width="100" height="100" align=left>
# 
# <img src="images/Construct.png" width="100" height="100" align=left>
# 
# ## **PACE: Analyze and Construct**
# 
# Consider the questions in your PACE Strategy Document and those below to craft your response:
# 1. Data professionals use descriptive statistics for Exploratory Data Analysis. How can computing descriptive statistics help you learn more about your data in this stage of your analysis?
# 

# ==> ENTER YOUR RESPONSE HERE

# ### **Task 2. Data exploration**
# 
# Use descriptive statistics to conduct Exploratory Data Analysis (EDA).
# 
# 

# <details>
#   <summary><h4><strong>Hint:</strong></h4></summary>
# 
# Refer back to *Self Review Descriptive Statistics* for this step-by-step proccess.
# 
# </details>

# Inspect the first five rows of the dataframe.

# In[3]:


# Display first few rows
### YOUR CODE HERE ###
data.head(10) 


# In[4]:


# Generate a table of descriptive statistics about the data
### YOUR CODE HERE ###
data.describe(include='all')


# Check for and handle missing values.

# In[5]:


# Check for missing values
### YOUR CODE HERE ###
data.isna().sum()


# In[6]:


# Drop rows with missing values

### YOUR CODE HERE ###
data = data.dropna(axis=0)


# In[8]:


# Display first few rows after handling missing values

### YOUR CODE HERE ###
data.head(10)


# You are interested in the relationship between `verified_status` and `video_view_count`. One approach is to examine the mean value of `video_view_count` for each group of `verified_status` in the sample data.

# In[9]:


# Compute the mean `video_view_count` for each group in `verified_status`
### YOUR CODE HERE ###
data.groupby('verified_status')["video_view_count"].mean()


# ### **Task 3. Hypothesis testing**
# 
# Before you conduct your hypothesis test, consider the following questions where applicable to complete your code response:
# 
# 1. Recall the difference between the null hypothesis and the alternative hypotheses. What are your hypotheses for this data project?

# **Null hypothesis: It's no connections within verifiend and unverified accounts towards in number of views between TikTok videos.**
# 
# **Alternative hypothesis: is a impact in number of views between verified and unverified accounts within TikTok.**
# 

# 
# 
# Your goal in this step is to conduct a two-sample t-test. Recall the steps for conducting a hypothesis test:
# 
# 
# 1.   State the null hypothesis and the alternative hypothesis
# 2.   Choose a signficance level
# 3.   Find the p-value
# 4.   Reject or fail to reject the null hypothesis
# 
# 

# **$H_0$**: It's no connections within verifiend and unverified accounts towards in number of views between TikTok videos.
# 
# **$H_A$**: is a impact in number of views between verified and unverified accounts within TikTok.
# 

# You choose 5% as the significance level and proceed with a two-sample t-test.

# In[13]:


# Conduct a two-sample t-test to compare means
### YOUR CODE HERE ###
significance_level = 0.05

not_verified = data[data["verified_status"] == "not verified"]["video_view_count"]
verified = data[data["verified_status"] == "verified"]["video_view_count"]

stats.ttest_ind(a=not_verified, b=verified, equal_var=False)


# **Question:** Based on the p-value you got above, do you reject or fail to reject the null hypothesis?
# 

# The p-value is twice smaller than significance level of 5%, so we are rejecting the null hypothesis that is no connections within verifiend and unverified accounts towards in number of views between TikTok videos. And concluding that **IS** a difference in the video view within TikTok between verified and unverified accounts.
# 

# <img src="images/Execute.png" width="100" height="100" align=left>
# 
# ## **PACE: Execute**
# 
# Consider the questions in your PACE Strategy Documentto reflect on the Execute stage.

# ## **Step 4: Communicate insights with stakeholders**

# *Ask yourself the following questions:*
# 
# 1. What business insight(s) can you draw from the result of your hypothesis test?

# Such experimental analysis is showing that has some statistically difference in mean video views between verified accounts and unverified. 
#  
# It can be conditioned by a lot of factors: clickbait videos, spam bots and etc. 

# **Congratulations!** You've completed this lab. However, you may not notice a green check mark next to this item on Coursera's platform. Please continue your progress regardless of the check mark. Just click on the "save" icon at the top of this notebook to ensure your work has been logged.
