# Playfield

This project includes work done during my learning in Data Science and in particular Machine Learning. 

Codes are currently organised according to the tools used, such as TensorFlow, Keras, Scikit-learn etc., but its structure will change as new learnings are added. 

Here's a description of current work in the Project:
- tensorflow/
	- tf_intro.ipynb - Introduction to basic low-level TensorFlow API mechanisms (tensors, graphs, sessions) and brief introduction to high-level Estimators
	- tf_iris_model.ipynb - A neural network built from scratch using the TensorFlow's Core API and the Iris dataset
	- tf_linreg.ipynb - Training a model for linear regression in TensorFlow
	- tf_mnist.ipynb - A classification model in TensorFlow on the MNIST dataset (incomplete)
- keras/
	- keras_dense_lstm.py - A keras model using the Dense and LSTM layers for matching an input letter to output numeric class
	- lstm_multivariate.py - An example of using multivariate time series data in a single-layer Long Short Term Memory (LSTM) RNN network
- sklearn/
	- encoding_categorical.ipynb - Data preprocessing with scikit-learn library (OneHotEncoder, LabelBinarizer, LabelEncoder etc.)
	- crossval_scale_hyperparams.ipynb - Brief example with SVM using cross-validation, scaling, hyperparameter tuning (incomplete)
	- sklearn_linreg.ipynb - Linear regression using scikit-learn
	- sklearn_svm.ipynb - Supervised learning using the SVM algorithm on the Iris dataset
	- svm_kmeans.py - Work on classification and clustering respectively using SVM and K-means. Also includes scaling, hyper-parameter search etc.
	- multiple_linreg.py - Multiple linear regression and model improving using Backward Elimination
- data/ 
	Folder to store data for all models
- tf_graph/
	Folder to store TensorBoard data


References:
- TensorFlow website
- Deep Learning with Python (book) - by Francois Chollet
- Machine Learning A-Z™: Hands-On Python & R In Data Science" by Kirill Eremenko and Hadelin de Ponteves
- datacamp.com open tutorials
- cv-tricks.com tutorials
