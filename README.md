# Chest-Xray-Classifier
NN resnet which detects whether or not someone has pneumonia from an Xray of their chest.

## Status - Work in progress, currently fixing errors which i suspect is related to data leakage (as the average loss from training is extremely low) and the classification report, as only one classes is being predicted.
-Update - now fixing recall problem with the normal class.

Dataset -> Chest X-Ray Images (Pneumonia), link on line below
https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
5,863 images

Results:               precision    recall  f1-score   support

                  normal       0.38      1.00      0.55       234
               pneumonia       0.00      0.00      0.00       390

               accuracy                           0.38       624
               macro avg       0.19      0.50      0.27       624
            weighted avg       0.14      0.38      0.20       624


Update - the bug turned about to be a data input issue, where the training data wasn't getting the data it should've been getting due to me inputting the wrong path. Here is the updates results from the classification report:

Results:              precision    recall  f1-score   support

      normal          0.93      0.59      0.73       234
   pneumonia          0.80      0.97      0.88       390

    accuracy                              0.83       624
   macro avg          0.87      0.78      0.80       624
weighted avg          0.85      0.83      0.82       624

From these scores, we can see that the normal class has a recall score of 59% whereas the pneumonia class has a score of 97%, indicating to me that the dataset could be imbalanced. To test this, I am going to calculate the imbalance and weights for the 2 classes. Then I will pass these weight values in a tensor where it will be passed into the cross entropy function to penalize the corresponding loss accordingly, rather than treating those two losses the same. Here are the results after including weights:

Results:               precision    recall  f1-score   support

          normal       0.94      0.58      0.72       234
       pneumonia       0.80      0.98      0.88       390

        accuracy                           0.83       624
       macro avg       0.87      0.78      0.80       624
    weighted avg       0.85      0.83      0.82       62
 
From this, We can observe that these results are actually very similar to the results of the classification report with no weights included, meaning that this did not solve the problem and the data being imbalanced is not the problem.

One thing the recall problem could be is the ResNet hidden layers being frozen. As the ResNet model was trained with millions of random images in various categories, however because of this, the features of different X-ray images may not be picked up effectively in the last layers of the model, especially the output layer.

The fix - I am going to unfreeze the backbone layers

How to run -> Open pneumoniaV2.ipynb as a kaggle notebook, uploading from github, then attach corresponding dataset and run all cells.
