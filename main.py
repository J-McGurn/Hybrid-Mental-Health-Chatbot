import nltk
from nltk.stem import WordNetLemmatizer
import json
import random
import numpy as np
from keras.models import load_model
import pickle
import aiml
import os

# Load the trained Keras Model
model = load_model('models/chatbot_model.keras')

# Load words and classes
words = pickle.load(open('pkl/words.pkl', 'rb'))
classes = pickle.load(open('pkl/classes.pkl', 'rb'))

# Load intents JSON
with open('data2/intents2.json', 'r', encoding='utf-8') as file:
    intents = json.load(file)

# Initialize Lemmatizer
lemmatizer = WordNetLemmatizer()

# Initialize AIML Kernel
aiml_kernel = aiml.Kernel()

# Load AIML rules from aiml_rules directory
aiml_rules_path = "aiml_rules"
os.chdir(aiml_rules_path)
aiml_kernel.learn("*.aiml")
os.chdir("..")  # Move back to main directory

# Function to clean and tokenize user input
def clean_up_sentence(sentence):
    sentence = sentence.lower()
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

# Optimized BoW function (Faster)
def bow(sentence, words):
    sentence_words = clean_up_sentence(sentence)
    bag = np.zeros(len(words), dtype=int)  # Efficient binary vector
    for word in sentence_words:
        if word in words:
            bag[words.index(word)] = 1  # Lookup instead of nested loops
    return np.array([bag])  # Reshape to match model input

# Predict class using the trained model
def predict_class(sentence, model):
    p = bow(sentence, words)
    res = model.predict(p)[0]
    
    ERROR_THRESHOLD = 0.4  # Lowered threshold
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)  # Sort by confidence

    if not results:
        return [{"intent": "no-response", "probability": "0.0"}]  # Default fallback

    return [{"intent": classes[r[0]], "probability": str(r[1])} for r in results]

# Get response based on predicted intent
def get_response(ints, intents_json):
    tag = ints[0]['intent']
    for i in intents_json['intents']:
        if i['tag'] == tag:
            return random.choice(i['responses'])
    return "I'm sorry, I don't understand that."

# Hybrid Response (AIML + Keras)
def chatbot_response(msg):
    aiml_response = aiml_kernel.respond(msg)
    if aiml_response and aiml_response.strip():  
        return aiml_response, None  # Return AIML response if matched

    # Use Neural Network if AIML fails
    ints = predict_class(msg, model)
    predicted_class = ints[0]['intent']

    # If model confidence is low, return a default response
    if predicted_class == "no-response":
        return "I'm sorry, I don't understand that. Can you rephrase?", None

    response = get_response(ints, intents)
    return response, predicted_class

# Run the chatbot
if __name__ == "__main__":
    print("Chatbot: Hello! I'm your mental health assistant. Type 'exit' to quit.")

    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            confirm = input("Are you sure you want to exit? (yes/no): ").strip().lower()
            if confirm == "yes":
                print("Chatbot: Goodbye! Take care.")
                break
            else:
                continue

        # Get chatbot response
        response, predicted_class = chatbot_response(user_input)
        print(f"Chatbot: {response}")
        if predicted_class:
            print(f"Predicted Class: {predicted_class}")
