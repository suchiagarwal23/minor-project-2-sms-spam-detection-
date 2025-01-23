import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import base64

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Load pre-trained assets
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))
ps = PorterStemmer()

# Function to preprocess text
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    text = [ps.stem(word) for word in text if word.isalnum() and word not in stopwords.words('english')]
    return " ".join(text)

# Function to set background image
def set_background_image(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded_string.decode()}");
            background-size: cover;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Set background image
set_background_image("premium_photo-1718119451320-7c803e80388b.avif")

# Add custom CSS for styling
st.markdown("""
    <style>
    .custom-label {
        font-size: 22px;
        font-weight: bold;
        color: #353536cc;
    }
    .stApp {
        background-color: rgba(255, 255, 255, 0.8);
    }
    </style>
""", unsafe_allow_html=True)

# App title and description
st.title("🚨 SMS Spam Detection 🚀")
st.markdown('<p style="font-size:25px;">Use this tool to classify messages as <strong>Spam</strong> or <strong>Not Spam</strong> with ease!</p>', unsafe_allow_html=True)
# Input box with custom label
st.markdown('<p class="custom-label">Enter the message:</p>', unsafe_allow_html=True)
input_sms = st.text_area("", height=200, key="input_sms")  # Multiline input

# Predict button
if st.button("Predict"):
    if not input_sms.strip():
        st.warning("⚠️ Please enter a message to classify.")
    else:
        # Step 1: Preprocess
        transformed_sms = transform_text(input_sms)
        # Step 2: Vectorize
        vector_input = tfidf.transform([transformed_sms])
        # Step 3: Predict
        result = model.predict(vector_input)[0]

        # Step 4: Display
        if result == 1:
            st.markdown('<p style="color:red;font-size:22px;">🚨 This message is classified as **Spam**!</p>', unsafe_allow_html=True)
        else:
            st.markdown('<p style="color:green;font-size:22px;">✅ This message is classified as **Not Spam**!</p>', unsafe_allow_html=True)

