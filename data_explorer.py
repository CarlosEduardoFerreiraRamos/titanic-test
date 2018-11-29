"""
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
"""