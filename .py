pip install pandas matplotlib tensorflow
import pandas as pd
data = pd.read_csv("Tweets.csv")
data.columns
review_data = data[['text','airline_sentiment']]
print(review_data.shape) 
review_data.head(10)
review_data = review_data[review_data['airline_sentiment'] != 'neutral']
print(review_data.shape)
review_data.head(5)
review_data["airline_sentiment"].value_counts()
sentiment_label = review_data.airline_sentiment.factorize() #0 represents
positive sentiment and the 1 represents negative sentiment.
sentiment_label
tweet = review_data.text.values
from tensorflow.keras.preprocessing.text import Tokenizer
tokenizer = Tokenizer(num_words=50000)
tokenizer.fit_on_texts(tweet)
encoded_docs = tokenizer.texts_to_sequences(tweet)
from tensorflow.keras.preprocessing.sequence import pad_sequences 
padsequence = pad_sequences(encoded_docs, maxlen=200)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM,Dense, Dropout, SpatialDropout1D
from tensorflow.keras.layers import Embedding
embedding_vector_length = 32
vocab_size=13250
model = Sequential()
model.add(Embedding(vocab_size,embedding_vector_length, input_length=200))
model.add(SpatialDropout1D(0.25))
model.add(LSTM(50, dropout=0.5, recurrent_dropout=0.5))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',
metrics=['accuracy']) 
print(model.summary())
history = model.fit(padsequence,sentiment_label[0],validation_split=0.2,
epochs=5, batch_size=32)
import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'], label='acc')
plt.plot(history.history['val_accuracy'], label='val_acc')
plt.legend()
plt.show()
plt.savefig("Accuracy plot.jpg")
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.legend()
plt.show()
plt.savefig("Loss plt.jpg")
def predict_sentiment(text):
tw = tokenizer.texts_to_sequences([text])
tw = pad_sequences(tw,maxlen=200)
prediction = int(model.predict(tw).round().item())
print("Predicted label: ", sentiment_label[1][prediction])
test_sentence1 = input("Enter the FIRST sentence to TEST: ")
predict_sentiment(test_sentence1)
