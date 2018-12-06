"""
# This explanation is available at the following link: https://www.analyticsvidhya.com/blog/2016/01/guide-data-exploration/#two
Data Exploration and Preparation

1 - Variable Identification
2 - Univariate Analysis
3 - Bi-variate Analysis
4 - Missing values treatment
5 - Outlier treatment
6 - Variable transformation
7 - Variable creation
"""

"""
1.Variable Identification

segregate varibles in: 
    *Type of varible: predictor or target variable;
    *Data type: Character or numeric;
    *Varible category: categorical or continuos.
"""

"""
2.Univariate Analysis
At this stage, we explore variables one by one. Method to perform uni-variate analysis will
depend on whether the variable type is categorical or continuous.
    Continuous Variables:- In case of continuous variables, we need to understand the central
    tendency and spread of the variable.
    Central tendency - Measure of Dispersion - Visualization Method
    Mean               Range                   Histogram
    Median             Quartile                Box plot
    Mode               IQR
    Min                Standerd Variation
                       Skewness and Kurtosis

    Categorical Variables:- For categorical variables, we’ll use frequency table to understand
    distribution of each category. We can also read as percentage of values under each category.
"""

"""
3.Bi-variate Analysis
Bi-variate Analysis finds out the relationship between two variables. Here, we look for association
and disassociation between variables at a pre-defined significance level. We can perform bi-variate
analysis for any combination of categorical and continuous variables. The combination can be:
Categorical & Categorical, Categorical & Continuous and Continuous & Continuous. Different methods
are used to tackle these combinations during analysis process.

    *Continuous & Continuous: While doing bi-variate analysis between two continuous variables, we
    should look at scatter plot. It is a nifty way to find out the relationship between two variables.
    The pattern of scatter plot indicates the relationship between variables. The relationship can
    be linear or non-linear.
        - Scatter plot shows the relationship between two variable but does not indicates the strength
          of relationship amongst them. To find the strength of the relationship, we use Correlation.
          Correlation varies between -1 and +1.

            -1: perfect negative linear correlation;
            +1:perfect positive linear correlation and 
            0: No correlation

        Correlation can be derived using following formula:
        Correlation = Covariance(X,Y) / SQRT( Var(X)* Var(Y))
    
    *Categorical & Categorical: To find the relationship between two categorical variables,
    we can use following methods:
        - Two-way table: We can start analyzing the relationship by creating a two-way table of count and
          count%. The rows represents the category of one variable and the columns represent the categories
          of the other variable. We show count or count% of observations available in each combination of
          row and column categories.

        - Stacked Column Chart: This method is more of a visual form of Two-way table.

        - Chi-Square Test: This test is used to derive the statistical significance of relationship between
          the variables. Also, it tests whether the evidence in the sample is strong enough to generalize
          that the relationship for a larger population as well. Chi-square is based on the difference between
          the expected and observed frequencies in one or more categories in the two-way table. It returns
          probability for the computed chi-square distribution with the degree of freedom.

          Probability 0, it indicates that both categorical variable are dependent, and 1 that they aren't

          Probability less than 0.05, it indicates that the relationship between the variables is significant
          at 95% confidence. The chi-square test statistic for a test of independence of two categorical
          variables is found by: x² = SUM((O-E)²/E). Where O represents the frequency, while E is the expected
          frequency under the null hypothesis and can be computed by: E = (row total x column total)/ sample size

          Statistical Measures used to analyze the power of relationship are:
            Cramer’s V for Nominal Categorical Variable
            Mantel-Haenszed Chi-Square for ordinal categorical variable.

    *Categorical & Continuous: While exploring relation between categorical and continuous variables, we can
    draw box plots for each level of categorical variables. If levels are small in number, it will not show
    the statistical significance. To look at the statistical significance we can perform Z-test, T-test or ANOVA.

        - Z-Test/ T-Test:- Either test assess whether mean of two groups are statistically different from each
          other or not.

          Z = |medianX1 - medianX2 | / root²( S1²/N1  -  S2²/N2 )
          
          If the probability of Z is small then the difference of two averages is more significant. The T-test is
          very similar to Z-test but it is used when number of observation for both categories is less than 30.
          
        - ANOVA:- It assesses whether the average of more than two groups is statistically different.

Missing value treatment

Why missing values treatment is required?
Missing data in the training data set can reduce the power / fit of a model or can lead to a biased model because
we have not analysed the behavior and relationship with other variables correctly. It can lead to wrong prediction
or classification.

Why my data has missing values?
We looked at the importance of treatment of missing values in a dataset. Now, let’s identify the reasons for
occurrence of these missing values. They may occur at two stages:

    *Data Extraction: It is possible that there are problems with extraction process. In such cases, we should 
     double-check for correct data with data guardians. Some hashing procedures can also be used to make sure data
     extraction is correct. Errors at data extraction stage are typically easy to find and can be corrected easily
     as well.
    *Data collection: These errors occur at time of data collection and are harder to correct. They can be categorized
     in four types:
        - Missing completely at random: This is a case when the probability of missing variable is same for all
          observations. For example: respondents of data collection process decide that they will declare their earning
          after tossing a fair coin. If an head occurs, respondent declares his / her earnings & vice versa. Here each
          observation has equal chance of missing value.
        - Missing at random: This is a case when variable is missing at random and missing ratio varies for different
          values / level of other input variables. For example: We are collecting data for age and female has higher missing
          value compare to male.
        - Missing that depends on unobserved predictors: This is a case when the missing values are not random and are related
          to the unobserved input variable. For example: In a medical study, if a particular diagnostic causes discomfort,
          then there is higher chance of drop out from the study. This missing value is not at random unless we have included
          “discomfort” as an input variable for all patients.
        - Missing that depends on the missing value itself: This is a case when the probability of missing value is directly
          correlated with missing value itself. For example: People with higher or lower income are likely to provide non-response
          to their earning.

Methods to treat missing values
    *Deletion:  It is of two types: List Wise Deletion and Pair Wise Deletion.
        - In list wise deletion, we delete observations where any of the variable is missing. Simplicity is one of the major
          advantage of this method, but this method reduces the power of model because it reduces the sample size.
        - In pair wise deletion, we perform analysis with all cases in which the variables of interest are present. Advantage of
          this method is, it keeps as many cases available for analysis. One of the disadvantage of this method, it uses different
          sample size for different variables.

    #Deletion methods are used when the nature of missing data is “Missing completely at random” else non random missing values
     can bias the model output.

    *Mean/ Mode/ Median Imputation: Imputation is a method to fill in the missing values with estimated ones. The objective is to
     employ known relationships that can be identified in the valid values of the data set to assist in estimating the missing values.
     Mean / Mode / Median imputation is one of the most frequently used methods. It consists of replacing the missing data for a given
     attribute by the mean or median (quantitative attribute) or mode (qualitative attribute) of all known values of that variable.
     It can be of two types:-
        - Generalized Imputation: In this case, we calculate the mean or median for all non missing values of that variable then replace
          missing value with mean or median. Like in above table, variable “Manpower” is missing so we take average of all non missing
          values of “Manpower” (28.33) and then replace missing value with it.
        - Similar case Imputation: In this case, we calculate average for gender “Male” (29.75) and “Female” (25) individually of non
          missing values then replace the missing value based on gender. For “Male“, we will replace missing values of manpower with
          29.75 and for “Female” with 25.

    *Prediction Model:  Prediction model is one of the sophisticated method for handling missing data. Here, we create a predictive model
     to estimate values that will substitute the missing data.  In this case, we divide our data set into two sets: One set with no missing
     values for the variable and another one with missing values. First data set become training data set of the model while second data
     set with missing values is test data set and variable with missing values is treated as target variable. Next, we create a model to
     predict target variable based on other attributes of the training data set and populate missing values of test data set.We can use
     regression, ANOVA, Logistic regression and various modeling technique to perform this.
    # There are 2 drawbacks for this approach:
        - The model estimated values are usually more well-behaved than the true values
        - If there are no relationships with attributes in the data set and the attribute with missing values, then the model will not be
          precise for estimating missing values.

    *KNN Imputation: In this method of imputation, the missing values of an attribute are imputed using the given number of attributes that
     are most similar to the attribute whose values are missing. The similarity of two attributes is determined using a distance function.
     It is also known to have certain advantage & disadvantages.
        - Advantages:
            # k-nearest neighbour can predict both qualitative & quantitative attributes
            # Creation of predictive model for each attribute with missing data is not required
            # Attributes with multiple missing values can be easily treated
            # Correlation structure of the data is taken into consideration
        - Disadvantage:
            # KNN algorithm is very time-consuming in analyzing large database. It searches through all the dataset looking for the most
              similar instances.
            # Choice of k-value is very critical. Higher value of k would include attributes which are significantly different from what we
              need whereas lower value of k implies missing out of significant attributes.
"""

"""
3.Outlier detection and treatment
Outliers tend to make your data skewed and reduces accuracy.

Outlier is a commonly used terminology by analysts and data scientists as it needs close attention else it can result in wildly wrong
estimations. Simply speaking, Outlier is an observation that appears far away and diverges from an overall pattern in a sample.

types of Outliers
Outlier can be of two types: Univariate and Multivariate. Above, we have discussed the example of univariate outlier. These outliers can
be found when we look at distribution of a single variable. Multi-variate outliers are outliers in an n-dimensional space. In order to find
them, you have to look at distributions in multi-dimensions.

causes Outliers
Whenever we come across outliers, the ideal way to tackle them is to find out the reason of having these outliers. The method to deal with
them would then depend on the reason of their occurrence. Causes of outliers can be classified in two broad categories:
    - Artificial (Error) / Non-natural
    - Natural.

    in detail:
      - Data Entry Errors: Human errors such as errors caused during data collection, recording, or entry can cause outliers in data.
      - Measurement Error: It is the most common source of outliers. This is caused when the measurement instrument used turns out to be
        faulty.
      - Experimental Error: Another cause of outliers is experimental error. For example: In a 100m sprint of 7 runners, one runner missed
        out on concentrating on the ‘Go’ call which caused him to start late. 
      - Intentional Outlier: This is commonly found in self-reported measures that involves sensitive data. For example: Teens would
        typically under report the amount of alcohol that they consume.
      - Data Processing Error: Whenever we perform data mining, we extract data from multiple sources. It is possible that some manipulation
        or extraction errors may lead to outliers in the dataset.
      - Sampling error: For instance, we have to measure the height of athletes. By mistake, we include a few basketball players in the
        sample. This inclusion is likely to cause outliers in the dataset.
      - Natural Outlier: When an outlier is not artificial (due to error), it is a natural outlier. For instance: In my last assignment
        with one of the renowned insurance company, I noticed that the performance of top 50 financial advisors was far higher than rest
        of the population. Surprisingly, it was not due to any error. Hence, whenever we perform any data mining activity with advisors,
        we used to treat this segment separately.

The impact of Outliers on a dataset
Outliers can drastically change the results of the data analysis and statistical modeling. There are numerous unfavourable impacts of
outliers in the data set:

  *It increases the error variance and reduces the power of statistical tests
  *If the outliers are non-randomly distributed, they can decrease normality
  *They can bias or influence estimates that may be of substantive interest
  *They can also impact the basic assumption of Regression, ANOVA and other statistical model assumptions.

Outlies Detection
Most commonly used method to detect outliers is visualization. We use various visualization methods, like Box-plot, Histogram, Scatter Plot.
Some analysts also various thumb rules to detect outliers. Like the following:
  *Any value, which is beyond the range of -1.5 x IQR to 1.5 x IQR
  *Use capping methods. Any value which out of range of 5th and 95th percentile can be considered as outlier
  *Data points, three or more standard deviation away from mean are considered outlier
  *Outlier detection is merely a special case of the examination of data for influential data points and it also depends on the business
   understanding
  *Bivariate and multivariate outliers are typically measured using either an index of influence or leverage, or distance. Popular indices
   such as Mahalanobis’ distance and Cook’s D are frequently used to detect outliers.

Removing Outliers
Most of the ways to deal with outliers are similar to the methods of missing values like deleting observations, transforming them, binning
them, treat them as a separate group, imputing values and other statistical methods.
stoped ehre
"""