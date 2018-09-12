"""
Natural Languase Processing - NLP

pos - part of speach - model

Bag of Wards - model
"""
import re;
import nltk;
nltk.download('stupwords');
from nltk.corpus import stopwords;
from nltk stem.porter import PorterStemmer
from sklearn.faeture_extraction.text import CountVectorize

class NLP(object):
	
	def __init__(self, language):
		self.lang = language;
		self.corpus = None;

	def clear(self, dataset, patter, replace = ' ', join = ' '):
	 	corpus = [];
		for e in dataset.values:
			review = re.sub(patter, replace, e);
			review = review.lower();
			review = review.split();
			ps = PorterStemmer();
			review = [ps.stem(word) for word in review if not word in set(stopwords.words(self.lang))];
			review = join.join(review);
			corpus.append(review)

	def create_bag_of_words():
		pass

"""

"""
import pandas as pd;

dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3);

corpus = [];
for e in dataset.values:
	review = re.sub('[^a-zA-Z]',' ', e);
	review = review.lower();
	review = review.split();
	ps = PorterStemmer();
	review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))];
	review = ' '.join(review);
	corpus.append(review);

cv = CountVectorize(max_features= 1500);
X = cv.fit_transform(corpus).toArray();
y = dataset.iloc[:, 1].values

"""
The most used for nlp is Naive bayes nad random tree.
we are gonna use naive bayes
"""
# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


"""
2. Evaluate the performance of each of these models.
Try to beat the Accuracy obtained in the tutorial.
But remember, Accuracy is not enough, so you should
also look at other performance metrics like Precision
(measuring exactness), Recall (measuring completeness)
and the F1 Score (compromise between Precision and Recall).
Please find below these metrics formulas (TP = # True
Positives, TN = # True Negatives, FP = # False Positives,
FN = # False Negatives):

Accuracy = (TP + TN) / (TP + TN + FP + FN)

Precision = TP / (TP + FP)

Recall = TP / (TP + FN)

F1 Score = 2 * Precision * Recall / (Precision + Recall)

3. Try even other classification models that we haven't covered
in Part 3 - Classification. Good ones for NLP include:

CART
C5.0
Maximum Entropy

"""