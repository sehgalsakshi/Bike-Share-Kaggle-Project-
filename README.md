# Bike-Share-Kaggle-Project-
Bike share demand kaggle problem: from feature selection, eda to ML models!

<b>Overview</b><br>
We’re given with a <b>bike sharing problem</b>. <br>
Bike sharing platform is used to <b>automate bike rentals</b> from one station to another.<br>
Unlike the regular rental problems, our goal is not to <strike>identify the income generated</strike> but to use <b>bike sharing data as mobility data within city</b> to identify what are days and weather conditions bike riders prefer bike riding on and to predict number of bikers.<br>
Also if we can figure out interesting patterns, we might also <b>identify important city events the bike riders are going for!</b><br>This problem is a <b>Regression Problem</b> since we've to identify count of bike riders given various conditions!

<b>About Dataset</b>

Contains hourly and daily data csv for years 2011-2012 having fields like season, month, day of week, humidity, temperature and windspeed which help us predicting count of bike riders.

## Notebook Sections

### Setting up the environment
Importing required liberaries
### Helper Functions
Functions for:
<ul><li>Separating categorical and numerical columns</li><li>Visualizing Correlation Heatmap</li><li>missing value analysis</li><li>Outliers plotting and removal (by standard deviation and by quantiles)</li><li>Chi square and Root mean square log error calculation</li><li>other functions for plotting kde plots, count plots (separate method for printing percentages) and scatter plots</li></ul>

### Feature Engineering
<ul><li>Created features like hour, date, day of week from date time column</li><li>
No null values are in this data set, hence missing value analysis not required</li><li>
Analysed data types of columns and visualized around 50% of data is categorical.</li></ul>

### Exploratory Data Analysis
<ul><li>Descriptive stats for numerical and categorical values</li><li>
  Correlation Matrix Visualization</li><li>
  Outlier Analysis for numerical variables and for our target variable on the basis of different categories</li><li>
  Chi2 test</li><li>
  Point plots for count in each hour on the basis of different categories (hues)</li><li>
  Using Random Forest for feature importance</li>
  
</ul>

### Feature Scaling
<ul><li>Log transform for target variable</li><li>
  Normalize or Standardize dilemma</li><li>
  Outlier Analysis for numerical variables and for our target variable on the basis of different categories</li><li>
  Chi2 test</li><li>
  Point plots for count in each hour on the basis of different categories (hues)</li><li>
  Using Random Forest for feature importance</li>
  <li>Train/ Test Split</li><li>
  Standardization</li>
</ul>

### Fitting the Models
Models Selected:
<ul><li>Linear Regression Model, the most basic</li><li>
  Ridge and Lasso Regularization</li><li>
  Decision Tree</li><li>
  Random Forest</li><li>
  XGBoost</li>
</ul>

### Observations:
<ul><li>atemp and temp have .99 correlation. Nothing surprising as they're essentially same values in different units</li><li>
  Windspeed has very low predicting power, should be dropped too.</li><li>
  Registered and casual are highly correlated with each other, rather "too" correlated with our target as they're leakage variables.
  There can be only two type of riders, casual and registered, together their count would entirely predict target. Hence we should drop them too.</li><li>
  It’s pretty evident outliers have been contributed the most by season 3. Also average numbers are highest in this season.</li><li>
  Working day or not, mean bikers count is almost the same telling us that biking is not just being used for leisure but for daily activities too.</li><li>
  Same is supported by the fact there is high rise in number of bikers from 7 to 9 AM and 5 to 7 and gradually decreasing on both sides, suggesting these as school or office commute hours.</li><li>
  Outliers in the evening time and on holidays or working or the seasons least favourable for bike rides may suggest an event in the city!</li><li>
  Distribution across all parameters is almost log normal, right skewed.</li><li>
  From chi2, we determined holiday, weekday, workingday, month, season all are dependent and from plotting we saw hour and season describes target the best of them!</li><li>
  We can see box cox has produced better distribution than simply taking log, due to hyper parameter</li><li>
  Our distribution is not perfectly normal and it still does contain some outlier. Let's go with standardization.</li><li>
  Linear Regression Model (with/ without regularization) was underfitting the data because our data was not completely a linear problem.</li><li>
  Decision Tree was overfitting but performance was better than linear models</li><li>
  Both Random Forest and XGBoost performed equally well on test (validation data) But the tie breaker was the train score!!
  </li><li>
  Well I found XGBoost better <strike>not because it scored better for test</strike> but because it performed relatively poorer on training set because that is what we want for model to over fit!</li>
  </ul>

### Major Takeaways/ Key Learnings:
  <ul><li><b>Multicollinearity</b> is dangerous as model won't be able to accurately which of the two variables is predicting it. Thus coefficients/ weights won't be assigned properly. <br>If in a linear model there are two independent variables that are linearly dependent on each other that is they are not truly independent then we cannot predict if the target variable is being affected by which variable</li><li>
    <b>Leakage Variables</b>: Variables that expose information about the target variable.
    When data contains a certain feature that already predicts the target has already occurred.
    <br>
    In our problem registered and nonregistered variables are the leakage variables because for any given row total count of these two variables tells us that this was the number of cyclists for the row hence it is directly predicting the target.<br>
    <a href="https://link.medium.com/Rk1IJWxeK9">More on this here</a></li><li>
  <b>Is feature Scaling Required?</b> Yes!Feature scaling is definitely required because we've observed there's linear relationship.<br>Hence to try on linear models, we need features to be scaled.<br/>There wouldn't have been a need if features would have been in similar ranges</li><li>
  <b>Normalize or Standardize??</b><br>
Normalization when distribution does not follow a Gaussian distribution. This can be useful in algorithms that do not assume any distribution of the data like K-Nearest Neighbors and Neural Networks.<br>
Standardization, helpful where the data follows a Gaussian distribution. However, this does not have to be necessarily true. Also, unlike normalization, standardization does not have a bounding range. So, even if you have outliers in your data, they will not be affected by standardization.<br>
    Woah! Ironical!! Normalization is actually for non normal distribution</li><li>
  <b>When to scale features?</b><br>
  <b>First Split, then Normalize!!!</b><br>Here's why!<br>
You first need to split the data into training and test set (validation set could be useful too).
Don't forget that testing data points represent real-world data. Feature normalization (or data standardization) of the explanatory (or predictor) variables is a technique used to center and normalise the data by subtracting the mean and dividing by the variance. If you take the mean and variance of the whole dataset you'll be introducing future information into the training explanatory variables (i.e. the mean and variance).
Therefore, you should perform feature normalisation over the training data. Then perform normalisation on testing instances as well, but this time using the mean and variance of training explanatory variables. In this way, we can test and evaluate whether our model can generalize well to new, unseen data points.
  </li><li>
  Also we realized it's a good approach to plot predicted variable's distribution over actual to visually analyze the result of our model</li></ul>

### <a href="https://www.kaggle.com/sehgalsakshi/notebookb2ce508d0c"> Head over to my Kaggle Notebook for problem statement, dataset and "My Notebook" (basically everything :P)</a>
