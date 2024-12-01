import streamlit as st
import tensorflow as tf
import nltk
from nltk.corpus import stopwords
from tensorflow.keras.layers import TextVectorization # type: ignore
import pandas as pd

# Load the pre-trained model
model = tf.keras.models.load_model('model/spam_classifier.keras')

# Re-initialize the TextVectorization layer with the same parameters
max_features = 10000
sequence_length = 128
vectorizer = TextVectorization(max_tokens=max_features, output_mode='int', output_sequence_length=sequence_length)

# Load the training data used to fit the vectorizer
data_path = 'data/Spam_SMS.csv'  # Update with correct path
df = pd.read_csv(data_path)

# NLTK setup
nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Fit the vectorizer on the training data
def preprocess_message(message):
    message = message.lower()
    message = ''.join([char for char in message if char.isalnum() or char.isspace()])
    words = nltk.word_tokenize(message)
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

# Apply preprocessing to the training data
train_messages = df['Message'].apply(preprocess_message)
vectorizer.adapt(train_messages)

# Streamlit UI
st.title("📱 Spam SMS Detection App")
st.write("Detect whether a text message is spam or not using this simple app powered by AI.")

# Input Section
st.header("🚀 Try It Out")
message = st.text_area("Enter the SMS message below:")

# Prediction Button
if st.button("🔍 Predict"):
    if message:
        with st.spinner("Analyzing the message..."):
            # Preprocess and predict
            processed_message = preprocess_message(message)
            message_vector = vectorizer([processed_message])
            prediction = model.predict(message_vector)[0][0]  # Get the prediction probability
            label = 'Spam' if prediction > 0.5 else 'Not Spam'
            confidence = prediction if label == 'Spam' else 1 - prediction
            
        # Display Results
        st.subheader("📊 Prediction Result")
        st.write(f"The message is classified as **{label}**.")
        st.write(f"Confidence: **{confidence:.2%}**")

        # Add conditional formatting
        if label == 'Spam':
            st.error("🚨 Warning: This message might be a spam!")
        else:
            st.success("✅ Safe: This message is likely not a spam.")
    else:
        st.warning("⚠️ Please enter a message before clicking Predict.")

# Footer
st.markdown("---")
st.write("📖 **About:** This app uses a TensorFlow model trained on SMS datasets to classify messages as spam or not spam.")
st.write("💻 **Developer:** Sophanith Som")
