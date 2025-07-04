# Hybrid Mental Health Chatbot

A hybrid AI chatbot combining **rule-based AIML logic** with a **feedforward neural network classifier** to provide safe and context-aware conversational support, with a focus on mental health interactions.

---

## Overview

This project explores the integration of traditional rule-based chatbot systems with modern deep learning techniques. The chatbot is designed to handle structured queries with deterministic AIML responses, while using a neural network classifier to manage flexible, nuanced conversations.

This hybrid architecture offers a balance between **reliability** and **adaptability**, making it especially suitable for **mental health-related interactions**, where accuracy and safety are critical.

---

## Features

- **Hybrid Chatbot Architecture** (Rule-based + AI model)
- **Feedforward Neural Network** using ReLU & Softmax for intent classification
- **AIML Pattern Matching** for high-confidence structured queries
- **Fallback System** to handle uncertain predictions safely
- **NLP Preprocessing**: tokenization, stemming, stop-word removal
- **Data Augmentation** for improved generalization
- Focus on **safe and ethical design** for sensitive interactions

---

## Dataset

**Source:** [Kaggle – Mental Health Conversational Data](https://www.kaggle.com/datasets/elvis23/mental-health-conversational-data)

- Preprocessed to create Bag-of-Words (BoW) vectors.
- Augmented using text transformations to improve model performance.
- Used to train the neural network for intent classification.

---

## Architecture

### AI Model

- **Input Layer:** 128 units, ReLU, L2 regularization
- **Batch Normalization**
- **Dropout:** 0.5
- **Hidden Layer:** 64 units, ReLU, L2 regularization
- **Batch Normalization + Dropout**
- **Output Layer:** Softmax classifier for multi-class intent prediction

### Rule-Based System

- Written in **AIML (Artificial Intelligence Markup Language)**
- Used for predictable queries (e.g., greetings, farewells, basic Q&A)
- Integrated via **PyAIML interpreter**

### Decision Flow

1. If AIML pattern matches input → rule-based response
2. If not matched:
   - Input classified via neural network
   - If prediction confidence ≥ threshold → AI-generated response
   - Else → fallback response for safety

---

### DISCLAIMER:

This project is developed for educational and research purposes only.
It is not intended for use in real-world mental health support or treatment scenarios.
The chatbot does not provide professional medical or psychological advice, and should not be relied upon by users experiencing serious mental health issues.
Always consult a qualified healthcare professional for mental health concerns.
