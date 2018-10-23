import nltk
import numpy as np
import random
import string 

#Reading the data
f = open('chatbotText.txt','r',errors = 'ignore')
raw = f.read()
raw = raw.lower() # to convert all words to lowercase

#nltk.download('punkt')
#nltk.download('wordnet')

sent_tokens = nltk.sent_tokenize(raw) #convert to list of sentences
word_tokens = nltk.word_tokenize(raw) #convert to list of words

#sent_tokens[:2]
#word_tokens[:2]

#Pre processing the raw data
lemmer = nltk.stem.WordNetLemmatizer()

def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]

remove_punct_dict = dict((ord(punct),  None) for punct in string.punctuation)

def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict))) #translate remove the puctutions based on the translation to none

#keyword matching
GREETING_INPUT = ("hello","hi","grettings","yo","What's up","hey")
GREETING_RESPONSES = ["Hi","Hey","*nods*","Hi there","Hello","I am glad! You are talking to me"]
def greeting(sentence):
    for word in sentence.split():
        if word.lower() in GREETING_INPUT:
            return random.choice(GREETING_RESPONSES)
    
#Generating response
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def response(user_response):
    robo_response = ''
    TfidfVec = TfidfVectorizer(tokenizer= LemNormalize,stop_words='english')
    tfidf = TfidfVec.fit_transform(sent_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx = vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    
    if(req_tfidf==0):
        robo_response = robo_response+"I am sorry! I don't understand you"
        return robo_response
    else:
        robo_response = robo_response + sent_tokens[idx]
        return robo_response

flag = True
print("ROBO: My name is Robo. I will answer your querries about Chatbots. If you want to exit, type bye!")

while(flag == True):
    user_response = input()
    user_response = user_response.lower()
    if(user_response!='bye'):
        if(user_response == 'thanks' or user_response == 'thank you'):
            flag = False
            print("ROBO: You are welcome..")
        else:
            if(greeting(user_response)!= None):
                print("ROBO: "+greeting(user_response))
            else:
                sent_tokens.append(user_response)
                word_tokens = word_tokens + nltk.word_tokenize(user_response)
                final_words = list(set(word_tokens))
                print("ROBO: ",end="")
                print(response(user_response))
                sent_tokens.remove(user_response)
    else:
        flag = False
        print("ROBO: Bye! Take care..")

    