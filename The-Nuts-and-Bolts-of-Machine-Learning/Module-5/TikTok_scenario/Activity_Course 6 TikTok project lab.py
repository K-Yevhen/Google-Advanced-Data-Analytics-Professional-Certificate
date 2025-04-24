#!/usr/bin/env python
# coding: utf-8

# # **TikTok Project**
# **Course 6 - The Nuts and bolts of machine learning**

# Recall that you are a data professional at TikTok. Your supervisor was impressed with the work you have done and has requested that you build a machine learning model that can be used to determine whether a video contains a claim or whether it offers an opinion. With a successful prediction model, TikTok can reduce the backlog of user reports and prioritize them more efficiently.
# 
# A notebook was structured and prepared to help you in this project. A notebook was structured and prepared to help you in this project. Please complete the following questions.

# # **Course 6 End-of-course project: Classifying videos using machine learning**
# 
# In this activity, you will practice using machine learning techniques to predict on a binary outcome variable.
# <br/>
# 
# **The purpose** of this model is to increase response time and system efficiency by automating the initial stages of the claims process.
# 
# **The goal** of this model is to predict whether a TikTok video presents a "claim" or presents an "opinion".
# <br/>
# 
# *This activity has three parts:*
# 
# **Part 1:** Ethical considerations
# * Consider the ethical implications of the request
# 
# * Should the objective of the model be adjusted?
# 
# **Part 2:** Feature engineering
# 
# * Perform feature selection, extraction, and transformation to prepare the data for modeling
# 
# **Part 3:** Modeling
# 
# * Build the models, evaluate them, and advise on next steps
# 
# Follow the instructions and answer the questions below to complete the activity. Then, you will complete an Executive Summary using the questions listed on the PACE Strategy Document.
# 
# Be sure to complete this activity before moving on. The next course item will provide you with a completed exemplar to compare to your own work.
# 
# 

# # **Classify videos using machine learning**

# <img src="images/Pace.png" width="100" height="100" align=left>
# 
# # **PACE stages**
# 

# Throughout these project notebooks, you'll see references to the problem-solving framework PACE. The following notebook components are labeled with the respective PACE stage: Plan, Analyze, Construct, and Execute.

# <img src="images/Plan.png" width="100" height="100" align=left>
# 
# 
# ## **PACE: Plan**
# 
# Consider the questions in your PACE Strategy Document to reflect on the Plan stage.
# 
# In this stage, consider the following questions:
# 
# 
# 1.   **What are you being asked to do? What metric should I use to evaluate success of my business/organizational objective?**
# 
# 2.   **What are the ethical implications of the model? What are the consequences of your model making errors?**
#   *   What is the likely effect of the model when it predicts a false negative (i.e., when the model says a video does not contain a claim and it actually does)?
# 
#   *   What is the likely effect of the model when it predicts a false positive (i.e., when the model says a video does contain a claim and it actually does not)?
# 
# 3.   **How would you proceed?**
# 

# **1. What are you being asked to do?
# 
# Business need and modeling objective
# 
# TikTok users can report videos that they believe violate the platform's terms of service. Because there are millions of TikTok videos created and viewed every day, this means that many videos get reported—too many to be individually reviewed by a human moderator.
# 
# Analysis indicates that when authors do violate the terms of service, they're much more likely to be presenting a claim than an opinion. Therefore, it is useful to be able to determine which videos make claims and which videos are opinions.
# 
# TikTok wants to build a machine learning model to help identify claims and opinions. Videos that are labeled opinions will be less likely to go on to be reviewed by a human moderator. Videos that are labeled as claims will be further sorted by a downstream process to determine whether they should get prioritized for review. For example, perhaps videos that are classified as claims would then be ranked by how many times they were reported, then the top x% would be reviewed by a human each day.
# 
# A machine learning model would greatly assist in the effort to present human moderators with videos that are most likely to be in violation of TikTok's terms of service.
# 
# Modeling design and target variable
# 
# The data dictionary shows that there is a column called claim_status. This is a binary value that indicates whether a video is a claim or an opinion. This will be the target variable. In other words, for each video, the model should predict whether the video is a claim or an opinion.
# 
# This is a classification task because the model is predicting a binary class.
# 
# Select an evaluation metric
# 
# To determine which evaluation metric might be best, consider how the model might be wrong. There are two possibilities for bad predictions:
# 
# False positives: When the model predicts a video is a claim when in fact it is an opinion
# False negatives: When the model predicts a video is an opinion when in fact it is a claim
# **2. What are the ethical implications of building the model? In the given scenario, it's better for the model to predict false positives when it makes a mistake, and worse for it to predict false negatives. It's very important to identify videos that break the terms of service, even if that means some opinion videos are misclassified as claims. The worst case for an opinion misclassified as a claim is that the video goes to human review. The worst case for a claim that's misclassified as an opinion is that the video does not get reviewed and it violates the terms of service. A video that violates the terms of service would be considered posted from a "banned" author, as referenced in the data dictionary.
# 
# Because it's more important to minimize false negatives, the model evaluation metric will be recall.
# 
# **3. How would you proceed?

# **Modeling workflow and model selection process**
# 
# Previous work with this data has revealed that there are ~20,000 videos in the sample. This is sufficient to conduct a rigorous model validation workflow, broken into the following steps:
# 
# 1. Split the data into train/validation/test sets (60/20/20)
# 2. Fit models and tune hyperparameters on the training set
# 3. Perform final model selection on the validation set
# 4. Assess the champion model's performance on the test set
# 
# ![](https://raw.githubusercontent.com/adacert/tiktok/main/optimal_model_flow_numbered.svg)
# 

# ### **Task 1. Imports and data loading**
# 
# Start by importing packages needed to build machine learning models to achieve the goal of this project.

# In[1]:


# Import packages for data manipulation
import pandas as pd
import numpy as np

# Import packages for data visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Import packages for data preprocessing
from sklearn.feature_extraction.text import CountVectorizer

# Import packages for data modeling
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, precision_score, \
recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from xgboost import plot_importance


# Now load the data from the provided csv file into a dataframe.
# 
# **Note:** As shown in this cell, the dataset has been automatically loaded in for you. You do not need to download the .csv file, or provide more code, in order to access the dataset and proceed with this lab. Please continue with this activity by completing the following instructions.

# In[2]:


# Load dataset into dataframe
data = pd.read_csv("tiktok_dataset.csv")


# <img src="images/Analyze.png" width="100" height="100" align=left>
# 
# ## **PACE: Analyze**
# 
# Consider the questions in your PACE Strategy Document to reflect on the Analyze stage.

# ### **Task 2: Examine data, summary info, and descriptive stats**

# Inspect the first five rows of the dataframe.

# In[3]:


# Display first few rows
### YOUR CODE HERE ###
data.head()


# Get the number of rows and columns in the dataset.

# In[5]:


# Get number of rows and columns
### YOUR CODE HERE ###
data.shape


# Get the data types of the columns.

# In[6]:


# Get data types of columns
### YOUR CODE HERE ###
data.info()


# Get basic information about the dataset.

# In[7]:


# Get basic information
### YOUR CODE HERE ###
data.describe()


# Generate basic descriptive statistics about the dataset.

# In[4]:


# Generate basic descriptive stats
### YOUR CODE HERE ###
data.isna().sum()


# Check for and handle missing values.

# In[5]:


# Check for missing values
### YOUR CODE HERE ###
data.isna().sum()


# In[6]:


# Drop rows with missing values
### YOUR CODE HERE ###
data = data.dropna(axis=0)


# In[12]:


# Display first few rows after handling missing values
### YOUR CODE HERE ###
data.head(10)


# Check for and handle duplicates.

# In[11]:


# Check for duplicates
### YOUR CODE HERE ###
data.duplicated().sum()


# Check for and handle outliers.

# In[16]:


### YOUR CODE HERE ###

''' Tree-based models are robust to outliers, so there is no need to impute or drop any values based on where they fall in their distribution.'''


# Check class balance.

# In[7]:


# Check class balance
### YOUR CODE HERE ###
data["claim_status"].value_counts(normalize=True)


# <img src="images/Construct.png" width="100" height="100" align=left>
# 
# ## **PACE: Construct**
# Consider the questions in your PACE Strategy Document to reflect on the Construct stage.

# ### **Task 3: Feature engineering**

# Extract the length of each `video_transcription_text` and add this as a column to the dataframe, so that it can be used as a potential feature in the model.

# In[8]:


# Extract the length of each `video_transcription_text` and add this as a column to the dataframe
### YOUR CODE HERE ###
data['text_length'] = data['video_transcription_text'].str.len()
data.head()


# Calculate the average text_length for claims and opinions.

# In[9]:


# Calculate the average text_length for claims and opinions
### YOUR CODE HERE ###
data[['claim_status', 'text_length']].groupby('claim_status').mean()


# Visualize the distribution of `text_length` for claims and opinions.

# In[10]:


# Visualize the distribution of `text_length` for claims and opinions
# Create two histograms in one plot
### YOUR CODE HERE ###
sns.histplot(data=data, stat="count", multiple="dodge", x="text_length",
             kde=False, palette="pastel", hue="claim_status",
             element="bars", legend=True)
plt.xlabel("video_transcription_text length (number of characters)")
plt.ylabel("Count")
plt.title("Distribution of video_transcription_text length for claims and opinions")
plt.show()


# **Feature selection and transformation**

# Encode target and catgorical variables.

# In[11]:


# Create a copy of the X data
### YOUR CODE HERE ###
X = data.copy()
# Drop unnecessary columns
X = X.drop(['#', 'video_id'], axis=1)
# Encode target variable
X['claim_status'] = X['claim_status'].replace({'opinion': 0, 'claim': 1})
# Dummy encode remaining categorical values
X = pd.get_dummies(X,
                   columns=['verified_status', 'author_ban_status'],
                   drop_first=True)
X.head()


# ### **Task 4: Split the data**

# Assign target variable.

# In[12]:


# Isolate target variable
### YOUR CODE HERE ###
y = X['claim_status']


# Isolate the features.

# In[13]:


# Isolate features
X = X.drop(['claim_status'], axis=1)

# Display first few rows of features dataframe
X.head()


# #### **Task 5: Create train/validate/test sets**

# Split data into training and testing sets, 80/20.

# In[14]:


# Split the data into training and testing sets
### YOUR CODE HERE ###
X_tr, X_test, y_tr, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# Split the training set into training and validation sets, 75/25, to result in a final ratio of 60/20/20 for train/validate/test sets.

# In[15]:


# Split the training data into training and validation sets
### YOUR CODE HERE ###
X_train, X_val, y_train, y_val = train_test_split(X_tr, y_tr, test_size=0.25, random_state=0)


# Confirm that the dimensions of the training, validation, and testing sets are in alignment.

# In[16]:


# Get shape of each training, validation, and testing set
### YOUR CODE HERE ###
X_train.shape, X_val.shape, X_test.shape, y_train.shape, y_val.shape, y_test.shape


# ### **Task 6. Build models**
# 

# ### **Build a random forest model**

# Fit a random forest model to the training set. Use cross-validation to tune the hyperparameters and select the model that performs best on recall.

# In[17]:


# Instantiate the random forest classifier
### YOUR CODE HERE ###
rf = RandomForestClassifier(random_state=0)

# Create a dictionary of hyperparameters to tune
cv_params = {'max_depth': [5, 7, None],
             'max_features': [0.3, 0.6],
            #  'max_features': 'auto'
             'max_samples': [0.7],
             'min_samples_leaf': [1,2],
             'min_samples_split': [2,3],
             'n_estimators': [75,100,200],
             }

# Define a list of scoring metrics to capture
scoring = ['accuracy', 'precision', 'recall', 'f1']

# Instantiate the GridSearchCV object
rf_cv = GridSearchCV(rf, cv_params, scoring=scoring, cv=5, refit='recall')


# In[18]:


# Set up a `CountVectorizer` object, which converts a collection of text to a matrix of token counts
count_vec = CountVectorizer(ngram_range=(2, 3),
                            max_features=15,
                            stop_words='english')
count_vec

count_data = count_vec.fit_transform(X_train['video_transcription_text']).toarray()
count_data

# Place the numerical representation of `video_transcription_text` from training set into a dataframe
count_df = pd.DataFrame(data=count_data, columns=count_vec.get_feature_names_out())

# Display first few rows
count_df.head()

# Concatenate `X_train` and `count_df` to form the final dataframe for training data (`X_train_final`)
# Note: Using `.reset_index(drop=True)` to reset the index in X_train after dropping `video_transcription_text`,
# so that the indices align with those in `X_train` and `count_df`
X_train_final = pd.concat([X_train.drop(columns=['video_transcription_text']).reset_index(drop=True), count_df], axis=1)

# Display first few rows
X_train_final.head()

# Extract numerical features from `video_transcription_text` in the testing set
validation_count_data = count_vec.transform(X_val['video_transcription_text']).toarray()
validation_count_data

# Place the numerical representation of `video_transcription_text` from validation set into a dataframe
validation_count_df = pd.DataFrame(data=validation_count_data, columns=count_vec.get_feature_names_out())
validation_count_df.head()

# Concatenate `X_val` and `validation_count_df` to form the final dataframe for training data (`X_val_final`)
# Note: Using `.reset_index(drop=True)` to reset the index in X_val after dropping `video_transcription_text`,
# so that the indices align with those in `validation_count_df`
X_val_final = pd.concat([X_val.drop(columns=['video_transcription_text']).reset_index(drop=True), validation_count_df], axis=1)

# Display first few rows
X_val_final.head()

# Extract numerical features from `video_transcription_text` in the testing set
test_count_data = count_vec.transform(X_test['video_transcription_text']).toarray()

# Place the numerical representation of `video_transcription_text` from test set into a dataframe
test_count_df = pd.DataFrame(data=test_count_data, columns=count_vec.get_feature_names_out())

# Concatenate `X_val` and `validation_count_df` to form the final dataframe for training data (`X_val_final`)
X_test_final = pd.concat([X_test.drop(columns=['video_transcription_text']
                                      ).reset_index(drop=True), test_count_df], axis=1)
X_test_final.head()


# In[20]:


# Examine best recall score
### YOUR CODE HERE ###
rf_cv.fit(X_train_final, y_train)


# In[21]:


# Examine best parameters
### YOUR CODE HERE ###
rf_cv.best_score_
rf_cv.best_params_


# Check the precision score to make sure the model isn't labeling everything as claims. You can do this by using the `cv_results_` attribute of the fit `GridSearchCV` object, which returns a numpy array that can be converted to a pandas dataframe. Then, examine the `mean_test_precision` column of this dataframe at the index containing the results from the best model. This index can be accessed by using the `best_index_` attribute of the fit `GridSearchCV` object.

# In[22]:


# Access the GridSearch results and convert it to a pandas df
rf_results_df = pd.DataFrame(rf_cv.cv_results_)

# Examine the GridSearch results df at column `mean_test_precision` in the best index
rf_results_df['mean_test_precision'][rf_cv.best_index_]


# **Question:** How well is your model performing? Consider average recall score and precision score.

# ### **Build an XGBoost model**

# In[23]:


# Instantiate the XGBoost classifier
xgb = XGBClassifier(objective='binary:logistic', random_state=0)

# Create a dictionary of hyperparameters to tune
cv_params = {'max_depth': [4,8,12],
             'min_child_weight': [3, 5],
             'learning_rate': [0.01, 0.1],
             'n_estimators': [300, 500]
             }

# Define a list of scoring metrics to capture
scoring = ['accuracy', 'precision', 'recall', 'f1']

# Instantiate the GridSearchCV object
xgb_cv = GridSearchCV(xgb, cv_params, scoring=scoring, cv=5, refit='recall')


# In[25]:


# Fit the model to the data
### YOUR CODE HERE ###

xgb_cv.fit(X_train_final, y_train)


# In[26]:


# Examine best recall score
### YOUR CODE HERE ###
xgb_cv.best_score_


# In[27]:


# Examine best parameters
### YOUR CODE HERE ###
xgb_cv.best_params_


# Repeat the steps used for random forest to examine the precision score of the best model identified in the grid search.

# In[28]:


# Access the GridSearch results and convert it to a pandas df
xgb_results_df = pd.DataFrame(xgb_cv.cv_results_)

# Examine the GridSearch results df at column `mean_test_precision` in the best index
xgb_results_df['mean_test_precision'][xgb_cv.best_index_]


# **Question:** How well does your model perform? Consider recall score and precision score.

# <img src="images/Execute.png" width="100" height="100" align=left>
# 
# ## **PACE: Execute**
# Consider the questions in your PACE Strategy Document to reflect on the Execute stage.

# ### **Task 7. Evaluate model**
# 
# Evaluate models against validation criteria.

# #### **Random forest**

# In[29]:


# Use the random forest "best estimator" model to get predictions on the validation set
### YOUR CODE HERE ###
y_pred = rf_cv.best_estimator_.predict(X_val_final)


# Display the predictions on the validation set.

# In[30]:


# Display the predictions on the validation set
### YOUR CODE HERE ###
y_pred


# Display the true labels of the validation set.

# In[31]:


# Display the true labels of the validation set
### YOUR CODE HERE ###
y_val


# Create a confusion matrix to visualize the results of the classification model.

# In[32]:


# Create a confusion matrix to visualize the results of the classification model

# Compute values for confusion matrix
log_cm = confusion_matrix(y_val, y_pred)

# Create display of confusion matrix
log_disp = ConfusionMatrixDisplay(confusion_matrix=log_cm, display_labels=None)

# Plot confusion matrix
log_disp.plot()

# Display plot
plt.show()


# Create a classification report that includes precision, recall, f1-score, and accuracy metrics to evaluate the performance of the model.
# <br> </br>
# 
# **Note:** In other labs there was a custom-written function to extract the accuracy, precision, recall, and F<sub>1</sub> scores from the GridSearchCV report and display them in a table. You can also use scikit-learn's built-in [`classification_report()`](https://scikit-learn.org/stable/modules/model_evaluation.html#classification-report) function to obtain a similar table of results.

# In[34]:


# Create a classification report
# Create classification report for random forest model
### YOUR CODE HERE ###
target_labels = ['opinion', 'claim']
print(classification_report(y_val, y_pred, target_names=target_labels))


# **Question:** What does your classification report show? What does the confusion matrix indicate?

# #### **XGBoost**
# 
# Now, evaluate the XGBoost model on the validation set.

# In[36]:


# Use the best estimator to predict on the validation data
### YOUR CODE HERE ###
y_pred = xgb_cv.best_estimator_.predict(X_val_final)
y_pred


# In[37]:


# Compute values for confusion matrix
log_cm = confusion_matrix(y_val, y_pred)

# Create display of confusion matrix
log_disp = ConfusionMatrixDisplay(confusion_matrix=log_cm, display_labels=None)

# Plot confusion matrix
log_disp.plot()

# Display plot
plt.title('XGBoost - validation set');
plt.show()


# In[38]:


# Create a classification report
### YOUR CODE HERE ###
target_labels = ['opinion', 'claim']
print(classification_report(y_val, y_pred, target_names=target_labels))


# **Question:** Describe your XGBoost model results. How does your XGBoost model compare to your random forest model?

# ### **Use champion model to predict on test data**

# In[39]:


### YOUR CODE HERE ###
y_pred = rf_cv.best_estimator_.predict(X_test_final)


# In[40]:


# Compute values for confusion matrix
log_cm = confusion_matrix(y_test, y_pred)

# Create display of confusion matrix
log_disp = ConfusionMatrixDisplay(confusion_matrix=log_cm, display_labels=None)

# Plot confusion matrix
log_disp.plot()

# Display plot
plt.title('Random forest - test set');
plt.show()


# #### **Feature importances of champion model**
# 

# In[41]:


### YOUR CODE HERE ###
importances = rf_cv.best_estimator_.feature_importances_
rf_importances = pd.Series(importances, index=X_test_final.columns)

fig, ax = plt.subplots()
rf_importances.plot.bar(ax=ax)
ax.set_title('Feature importances')
ax.set_ylabel('Mean decrease in impurity')
fig.tight_layout()


# **Question:** Describe your most predictive features. Were your results surprising?

# ### **Task 8. Conclusion**
# 
# In this step use the results of the models above to formulate a conclusion. Consider the following questions:
# 
# 1. **Would you recommend using this model? Why or why not?**
# 
# 2. **What was your model doing? Can you explain how it was making predictions?**
# 
# 3. **Are there new features that you can engineer that might improve model performance?**
# 
# 4. **What features would you want to have that would likely improve the performance of your model?**
# 
# Remember, sometimes your data simply will not be predictive of your chosen target. This is common. Machine learning is a powerful tool, but it is not magic. If your data does not contain predictive signal, even the most complex algorithm will not be able to deliver consistent and accurate predictions. Do not be afraid to draw this conclusion.
# 

# ==> ENTER YOUR RESPONSES HERE

# **Congratulations!** You've completed this lab. However, you may not notice a green check mark next to this item on Coursera's platform. Please continue your progress regardless of the check mark. Just click on the "save" icon at the top of this notebook to ensure your work has been logged.
