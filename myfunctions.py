import re
import nltk
from nltk.corpus import stopwords


nltk.download("stopwords")
def procces_sentence(review:str)->str:
    sample_review = re.sub(r'http\S+', '', review)
    sample_review = re.sub("[^a-zA-Z]",' ',sample_review)
    sample_review = sample_review.lower()
    sample_review = sample_review.split()
    swords = set(stopwords.words("english"))                     
    sample_review = [w for w in sample_review if w not in swords]        
    sample_review = " ".join(sample_review)
    return sample_review



    
def predict_sentiment(new_sentence:str, model, vectorizer)->str:
    cleaned_sentence = procces_sentence(new_sentence)
    new_sentence_vector = vectorizer.transform([cleaned_sentence])
    new_sentence_vector = new_sentence_vector.toarray()
    prediction = model.predict(new_sentence_vector)
        
    if prediction == 2: sentiment = "positive"
    elif prediction == 1: sentiment = "neutral"
    elif prediction == 0: sentiment = "negative"
    else: sentiment = "undefined"

    return sentiment
