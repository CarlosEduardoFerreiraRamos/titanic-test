"""
Classifers pros and cons        PROS  											CONS

Logistic Regression:      Probabilistic approach, gives informations        The Logistic Regression assuptions
						  about statistical significance features.        

KNN:    				  Simple to understand, fast and efficient			Need to choose the number of neighbours k
						  													  
SVM:                      Performant, not biased by outliers,				Not appropriate for non linear problems, not
						  not sensitive to overfitting. 					the best choice for large number of features

KERNEL SVM:				  High performance on linear problemns, not 		Not the best choice for large number of features,
						  biased by outliers, not sensitive to overfitting 	more complex

NAIVE BAYES:			  Efficient, not biased by outliers, works on  		Based on the assumption taht features have 
                          non linear problems, probabilisti approach 		same statistical relevance

DECISION TREE C.:		  Interpretability, no need for feature scaaling 	Poor results on too small datasets, 
						  works o both linear / non linear problemns 		overfitting can easily occur

RANDOM FOREST C.:		  Powerful and accurate, good performance on  		No interpretability, overfitting can easily 
                          many problems, including non linear		 		occur, need to choose the number o f trees sets

##########################################
Linear models; Logistic Regression and SVM
Non Linear models: K-NN, Naive Bayes, Decision Tree or Random Forest 

Then from a business point of view, you would rather use:

- Logistic Regression or Naive Bayes when you want to rank your predictions by their probability.
For example if you want to rank your customers from the highest probability that they buy a certain product,
to the lowest probability. Eventually that allows you to target your marketing campaigns. And of course for this
type of business problem, you should use Logistic Regression if your problem is linear, and Naive Bayes if your
problem is non linear.

- SVM when you want to predict to which segment your customers belong to. Segments can be any kind of segments,
for example some market segments you identified earlier with clustering.

- Decision Tree when you want to have clear interpretation of your model results,

- Random Forest when you are just looking for high performance with less need for interpretation. 
"""