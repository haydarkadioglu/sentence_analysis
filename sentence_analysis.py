# %% [markdown]
# # this model just created with 50.000 sentence
# # if you want more you can change number of for loop in the 5th cell

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
import re
import nltk
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
import myfunctions as func

# %%
df = pd.read_csv('dataset/sentences.csv',  delimiter=',', quoting=2, encoding="utf-8")




# %%
def take_res(count:int)->int:
    labels = df["labels"][count]
    if labels == "bad": return 0
    elif labels == "neutral": return 1
    elif labels == "good": return 2

trainx = []
trainy = []
for i in range(50000):
    trainx.append(func.procces_sentence(df["tweets"][i]))
    trainy.append(take_res(i))
    if i%10000 == 0:
        print(f"proccess sentence number: {i}")



# %%
trainy = np.array(trainy)
trainy
train_x, test_x, y_train, y_test = train_test_split(trainx,trainy, test_size = 0.1)
vectorizer = CountVectorizer( max_features = 5000 )
train_x = vectorizer.fit_transform(train_x)



# %%
train_x.shape, y_train.shape

# %%
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(train_x, y_train)

# %%
test_xx = vectorizer.transform(test_x)
test_xx
test_xx = test_xx.toarray()




# %%
new_sentence = "i love you"

predicted_sentiment = func.predict_sentiment(new_sentence, model, vectorizer)
print(f"Girilen cümlenin sentiment değeri: {predicted_sentiment}")

# %%


#with open("trained_model.pkl", "wb") as file:
 #   joblib.dump(model, file)


