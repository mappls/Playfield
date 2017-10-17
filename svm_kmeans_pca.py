from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt

digits = datasets.load_digits()
digits_data = digits.data
digits_target = digits.target
number_digits = len(np.unique(digits.target))
digits_images = digits.images


def svm():
    from sklearn import svm

    # gamma, C and kernel here are selected manually, but in order to find the best parameters you can use the method
    # svm_choose_params, below:
    svm_model1 = svm.SVC(gamma=0.001, C=100., kernel='linear')
    svm_model1.fit(X_train, y_train)
    sc1 = svm_model1.score(X_test, y_test)
    print("svm_model1 score: %.3f" % sc1)

    # If you run svm_choose_params, you'll find that the best parameters are gamma=0.001, C=10, kernel='rbf'
    # svm_choose_params()

    # Use the optimal parameters
    svm_model2 = svm.SVC(gamma=0.001, C=10, kernel='rbf').fit(X_train, y_train)
    sc2 = svm_model2.score(X_test, y_test)
    print("svm_model2 score: %.3f" % sc2)

    # Predict
    pred1 = svm_model1.predict(X_test)
    pred2 = svm_model2.predict(X_test)
    print("svm_model1 mislcass. error: %.3f %%" % (sum(pred1 != y_test) * 100 / len(y_test)))
    print("svm_model2 mislcass. error: %.3f %%" % (sum(pred2 != y_test) * 100 / len(y_test)))

    # Visualise the predicted outputs
    # plot_svm_predicted(svm_model2)

    # Evaluate
    from sklearn import metrics

    # Print the classification report of `y_test` and `predicted`
    print(metrics.classification_report(y_test, pred2))

    # Print the confusion matrix of `y_test` and `predicted`
    print(metrics.confusion_matrix(y_test, pred2))

    # Plot the predicted and actual samples
    plot_svm_labels(svm_model2)


def svm_choose_params():
    from sklearn import svm

    # Split the `digits` data into two equal sets
    X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.5, random_state=0)

    # Import GridSearchCV
    from sklearn.grid_search import GridSearchCV

    # Set the parameter candidates
    parameter_candidates = [
        {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
        {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
    ]

    # Create a classifier with the parameter candidates
    clf = GridSearchCV(estimator=svm.SVC(), param_grid=parameter_candidates, n_jobs=-1)

    # Train the classifier on training data
    clf.fit(X_train, y_train)

    # Print out the results
    print('Best score for training data:', clf.best_score_)
    print('Best `C`:', clf.best_estimator_.C)
    print('Best kernel:', clf.best_estimator_.kernel)
    print('Best `gamma`:', clf.best_estimator_.gamma)



def k_means():
    #
    # Cluster the data into 10 clusters
    #

    # Import the `cluster` module
    from sklearn import cluster

    # Create the KMeans model
    # keep 'random_state' same as earlier (in 'train_test_split') for reproducibility
    clf = cluster.KMeans(init='k-means++', n_clusters=10, random_state=42)

    # Fit the training data `X_train`to the model
    clf.fit(X_train)

    # plot_clusters()

    # Predict the labels for `X_test`
    y_pred = clf.predict(X_test)
    print(y_pred[:10])
    print(y_test[:10])
    clf.cluster_centers_.shape

    # plot_kmeans_clusters()

    #
    # Evaluate k-means
    #
    from sklearn import metrics
    print(metrics.confusion_matrix(y_test, y_pred))

    from sklearn.metrics import homogeneity_score, completeness_score, v_measure_score, adjusted_rand_score, \
        adjusted_mutual_info_score, silhouette_score
    print('% 9s' % 'inertia    homo   compl  v-meas     ARI AMI  silhouette')
    print('%i   %.3f   %.3f   %.3f   %.3f   %.3f    %.3f'
              %(clf.inertia_,
          homogeneity_score(y_test, y_pred),
          completeness_score(y_test, y_pred),
          v_measure_score(y_test, y_pred),
          adjusted_rand_score(y_test, y_pred),
          adjusted_mutual_info_score(y_test, y_pred),
          silhouette_score(X_test, y_pred, metric='euclidean')))


#
# Plot initial data of digits
#
def plot_images():
    # Figure size (width, height) in inches
    fig = plt.figure(figsize=(6, 6))

    # Adjust the subplots
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

    # For each of the 64 images
    for i in range(64):
        # Initialize the subplots: add a subplot in the grid of 8 by 8, at the i+1-th position
        ax = fig.add_subplot(8, 8, i + 1, xticks=[], yticks=[])
        # Display an image at the i-th position
        ax.imshow(digits.images[i], cmap=plt.cm.binary, interpolation='nearest')
        # label the image with the target value
        ax.text(0, 7, str(digits.target[i]))

    # Show the plot
    plt.show()


#
# Apply PCA on the data to visualize it in 2 dimensions
#
def plot_pca():
    from sklearn.decomposition import RandomizedPCA
    from sklearn.decomposition import PCA

    # Create a Randomized PCA model that takes two components
    randomized_pca = RandomizedPCA(n_components=2)

    # Fit and transform the data to the model
    reduced_data_rpca = randomized_pca.fit_transform(digits.data)

    # Create a regular PCA model
    pca = PCA(n_components=2)

    # Fit and transform the data to the model
    reduced_data_pca = pca.fit_transform(digits.data)

    # Inspect the shape
    reduced_data_pca.shape

    # Print out the data
    print(reduced_data_rpca)
    print(reduced_data_pca)

    colors = ['black', 'blue', 'purple', 'yellow', 'white', 'red', 'lime', 'cyan', 'orange', 'gray']
    for i in range(len(colors)):
        x = reduced_data_rpca[:, 0][digits.target == i]
        y = reduced_data_rpca[:, 1][digits.target == i]
        plt.scatter(x, y, c=colors[i])
    plt.legend(digits.target_names, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.title("PCA Scatter Plot")
    plt.show()


def plot_clusters():

    from sklearn import cluster
    clf = cluster.KMeans(init='k-means++', n_clusters=10, random_state=42)

    # Figure size in inches
    fig = plt.figure(figsize=(8, 3))

    # Add title
    fig.suptitle('Cluster Center Images', fontsize=14, fontweight='bold')

    # For all labels (0-9)
    for i in range(10):
        # Initialize subplots in a grid of 2X5, at i+1th position
        ax = fig.add_subplot(2, 5, 1 + i)
        # Display images
        ax.imshow(clf.cluster_centers_[i].reshape((8, 8)), cmap=plt.cm.binary)
        # Don't show the axes
        plt.axis('off')

    # Show the plot
    plt.show()


def plot_kmeans_clusters():

    from sklearn import cluster
    clf = cluster.KMeans(init='k-means++', n_clusters=10, random_state=42)

    # Import `Isomap()`
    from sklearn.manifold import Isomap

    # Create an isomap and fit the `digits` data to it
    X_iso = Isomap(n_neighbors=10).fit_transform(X_train)

    # Compute cluster centers and predict cluster index for each sample
    clusters = clf.fit_predict(X_train)

    # Create a plot with subplots in a grid of 1X2
    fig, ax = plt.subplots(1, 2, figsize=(8, 4))

    # Adjust layout
    fig.suptitle('Predicted Versus Training Labels', fontsize=14, fontweight='bold')
    fig.subplots_adjust(top=0.85)

    # Add scatterplots to the subplots
    ax[0].scatter(X_iso[:, 0], X_iso[:, 1], c=clusters)
    ax[0].set_title('Predicted Training Labels')
    ax[1].scatter(X_iso[:, 0], X_iso[:, 1], c=y_train)
    ax[1].set_title('Actual Training Labels')

    # Show the plots
    plt.show()


def plot_svm_predicted(svc_model):
    # Import matplotlib
    import matplotlib.pyplot as plt

    # Assign the predicted values to `predicted`
    predicted = svc_model.predict(X_test)

    # Zip together the `images_test` and `predicted` values in `images_and_predictions`
    images_and_predictions = list(zip(images_test, predicted))

    # For the first 4 elements in `images_and_predictions`
    for index, (image, prediction) in enumerate(images_and_predictions[:4]):
        # Initialize subplots in a grid of 1 by 4 at positions i+1
        plt.subplot(1, 4, index + 1)
        # Don't show axes
        plt.axis('off')
        # Display images in all subplots in the grid
        plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
        # Add a title to the plot
        plt.title('Predicted: ' + str(prediction))

    # Show the plot
    plt.show()


def plot_svm_labels(svc_model):
    # Import `Isomap()`
    from sklearn.manifold import Isomap

    # Create an isomap and fit the `digits` data to it
    X_iso = Isomap(n_neighbors=10).fit_transform(X_train)

    # Compute cluster centers and predict cluster index for each sample
    predicted = svc_model.predict(X_train)

    # Create a plot with subplots in a grid of 1X2
    fig, ax = plt.subplots(1, 2, figsize=(8, 4))

    # Adjust the layout
    fig.subplots_adjust(top=0.85)

    # Add scatterplots to the subplots
    ax[0].scatter(X_iso[:, 0], X_iso[:, 1], c=predicted)
    ax[0].set_title('Predicted labels')
    ax[1].scatter(X_iso[:, 0], X_iso[:, 1], c=y_train)
    ax[1].set_title('Actual Labels')

    # Add title
    fig.suptitle('Predicted versus actual labels', fontsize=14, fontweight='bold')

    # Show the plot
    plt.show()


if __name__ == '__main__':

    # Preprocess data
    #
    # shift the distribution of each attribute to have a mean of zero and a standard deviation of one (unit variance)
    #
    from sklearn.preprocessing import scale

    # Apply `scale()` to the `digits` data
    data = scale(digits.data)

    #
    # Split data into training and test sets
    #
    from sklearn.cross_validation import train_test_split

    # Split the `digits` data into training and test sets
    X_train, X_test, y_train, y_test, images_train, images_test = train_test_split(data, digits.target, digits.images,
                                                                                   test_size=0.25, random_state=42)

    # Use either k_means() or svm()
    svm()

