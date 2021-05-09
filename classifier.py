import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from PIL import Image
import PIL.ImageOps

# getiing the data
X = np.load('image.npz')
X = X['arr_0']
y = pd.read_csv("labels.csv")
y = y["labels"]

classes = ["A","B" ,"C" ,"D" ,"E" ,"F" ,"G" ,"H" ,"I" ,"J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]
n_classes = len(classes)

#Splitting the data and scaling it
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=9, train_size=3500, test_size=500)

#scaling the features to binary
scaled_X_train = X_train/255.0
scaled_X_test = X_test/255.0



def get_alphabet(image):

    #opening alphabet image
    im_pil = Image.open(image)
    alph_image = im_pil.convert('L')

    #resizing the picture
    alph_image_resized = alph_image.resize((28,28), Image.ANTIALIAS)

    #setting minimum and max pixel
    pixel_filter = 20
    min_pixel = np.percentile(alph_image_resized, pixel_filter)
    alph_image_resized_inverted_scaled = np.clip(alph_image_resized-min_pixel, 0, 255)
    max_pixel = np.max(alph_image_resized)

    # inverting for proper recognition
    alph_image_resized_inverted_scaled = np.asarray(alph_image_resized_inverted_scaled)/max_pixel

    # converting 2 array and predicting the alphabet
    test_sample = np.array(alph_image_resized_inverted_scaled).reshape(1,660)
    test_pred = clf.predict(test_sample)
    return test_pred[0]