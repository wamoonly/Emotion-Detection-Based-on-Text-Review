import re
import io
import csv
import nltk
import time
import copy
import torch
import numpy as np
import pandas as pd
import streamlit as st
from bert import bert_ATE
import plotly.express as px
from dataset import dataset_ATM


from transformers import BertTokenizer
from transformers import pipeline
from nltk import word_tokenize, sent_tokenize, pos_tag
from torch.utils.data import DataLoader, ConcatDataset
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
pretrain_model_name = "bert-base-uncased"
classifier = pipeline("text-classification", model='bhadresh-savani/bert-base-uncased-emotion', return_all_scores=True)


#part of emotion detection
def load_model(model, path):
    model.load_state_dict(torch.load(path), strict=False)
    return model

def extract_aspect_sentences(text, aspects):
    sentences = sent_tokenize(text)
    aspect_sentences = {aspect: [] for aspect in aspects}

    for sentence in sentences:
        words = word_tokenize(sentence)
        pos_tags = pos_tag(words)

        for aspect in aspects:
            aspect_keywords = [word.lower() for word in aspect.split()]

            if any(keyword in [word.lower() for word, pos in pos_tags] for keyword in aspect_keywords):
                aspect_sentences[aspect].append(sentence)

    return aspect_sentences

def predict_model_ATE(sentence, tokenizer):
    word_pieces = []
    tokens = tokenizer.tokenize(sentence)
    word_pieces += tokens

    ids = tokenizer.convert_tokens_to_ids(word_pieces)
    input_tensor = torch.tensor([ids]).to(DEVICE)

    with torch.no_grad():
        outputs = model_ATE(input_tensor, None, None)
        _, predictions = torch.max(outputs, dim=2)
    predictions = predictions[0].tolist()

    return word_pieces, predictions, outputs



# Function to combine ATE and emotion prediction
def ATE_emotion_prediction(text, model_ATE, tokenizer, emotion_classifier):

    terms = []
    word = ""

    # Assuming that predict_model_ATE returns the tokens x, labels y, and other information (_)
    x, y, _ = predict_model_ATE(text, tokenizer)

    for i in range(len(y)):
        if y[i] == 1:
            # If the label is 1, it indicates the start of a new aspect term
            if len(word) != 0:
                # If there's an existing word, append it to the terms list after removing " ##"
                terms.append(word.replace(" ##", ""))
            # Start a new word with the current token
            word = x[i]
        if y[i] == 2:
            word += (" " + x[i])

    if len(word) != 0:
        # print(terms)
        terms.append(word.replace(" ##", ""))

    combined_terms = []
    for term in terms:
        if "#" in term:
            if combined_terms:
                combined_terms[-1] += term
        else:
            combined_terms.append(term)

    # Remove hashtags from the combined terms
    terms_without_hashtag = [term.replace("#", "") for term in combined_terms]
    # Replace specific words with desired format
    for i in range(len(terms_without_hashtag)):
        if  terms_without_hashtag[i] == 'nasilemak':
            terms_without_hashtag[i] = 'nasi lemak'
           


    # Print the original tokens and the extracted aspect terms
    # print("Tokens:", x)
    # print("ATE:", terms_without_hashtag)
    print()

    
    result = extract_aspect_sentences(text, terms_without_hashtag)
    emotion_dict = {}
    emotion_dict = copy.deepcopy(result)
    
    # print(type(result))
    # print(f"Result : {result}")
    for aspect, sentences in result.items():
        # print(f"{aspect} sentences:")
        # print("iteration aspect sentence")
        # Loop over each sentence for the current aspect
        for sentence in sentences:
            # print(f"- {sentence}")
            
            # Assuming you have an emotion_classifier function
            emotion_result = emotion_classifier(sentence)
            emotion_dict[aspect]+= emotion_result
            

            # print(emotion_result)
            # print(print(emotion_dict))
     
        print()  # Print a new line after all sentences for the current aspect
    
    return terms_without_hashtag, emotion_dict

def sorted_round(list_emotion):
    def custom_sort(emotion):
            try:
                return float(emotion['score'])
            except (ValueError, TypeError):
                return 0.0

    sorted_emotion_data = sorted(list_emotion, key=custom_sort, reverse=True)
    rounded_sorted_emotion_data = [{'label': entry['label'], 'score': round((entry['score']),3)} for entry in sorted_emotion_data]

    return rounded_sorted_emotion_data

def process_csv_file(uploaded_file):
    results_df = pd.DataFrame(columns=['Aspect', 'Label1', 'Score1', 'Label2', 'Score2', 'Label3', 'Score3', 'Label4', 'Score4', 'Label5', 'Score5', 'Label6', 'Score6'])

    # Convert the uploaded file to a DataFrame
    df = pd.read_csv(uploaded_file)
    
    for index, row in df.iterrows():
        text = row['Text']
        terms, result_prediction = ATE_emotion_prediction(text, model_ATE, tokenizer, classifier)
        aspect_df = pd.DataFrame()
        for aspect, data in result_prediction.items():
            emotions_list = data[1]
                    
            rounded_sorted_emotion_data = sorted_round(emotions_list)
                    
            # Create a new row for each aspect
            new_row = {'Aspect': aspect}
            for i, entry in enumerate(rounded_sorted_emotion_data):
                new_row[f'Label{i+1}'] = entry['label']
                new_row[f'Score{i+1}'] = entry['score']

            # Append the new row to the DataFrame
            # results_df = results_df.append(new_row, ignore_index=True)
            results_df = pd.concat([results_df, pd.DataFrame([new_row])], ignore_index=True)

    return df,results_df

def process_single_input(text):
    prediction_result = ATE_emotion_prediction(text, model_ATE, tokenizer, classifier)
    # st.write("PR: ",prediction_result[1])
    # print(type(prediction_result))
    # st.write(type(prediction_result))
    new_results_df = pd.DataFrame(columns=['Aspect', 'Label1', 'Score1', 'Label2', 'Score2', 'Label3', 'Score3', 'Label4', 'Score4', 'Label5', 'Score5', 'Label6', 'Score6'])
            
    for aspect, aspect_emotions in prediction_result[1].items():
        print(f"\nAspect: {aspect}")
        if aspect_emotions:
            emotion_list = aspect_emotions[1]  
            rounded_sorted_emotion_data = sorted_round(emotion_list)

            new_rows = {'Aspect': aspect}
            for i, entry in enumerate(rounded_sorted_emotion_data):
                new_rows[f'Label{i+1}'] = entry['label']
                new_rows[f'Score{i+1}'] = entry['score']
            
            # new_results_df = new_results_df.append(new_rows, ignore_index=True)
            new_results_df = pd.concat([new_results_df, pd.DataFrame([new_rows])], ignore_index=True)

    # print("Emotion Prediction DataFrame:")
    # print(results_df)
    return new_results_df

    # print(results_df.head())       
    # results_df.to_csv("LALALa.csv", index=False, encoding='utf-8')


#input format
def check_input_format(text):
    sentences = re.split(r'\.\s*', text)
    for index, sentence in enumerate(sentences[:-1]):  
        if not sentence.endswith(" "):
            return False
    # Check the last sentence if it ends with a dot and does not have a space after it
    if sentences[-1].endswith(".") and not text.endswith(" "):
        return False
    return True

def simulate_popup(message, duration=5):
    # Display a placeholder to show the message
    placeholder = st.empty()
    placeholder.write(message)

    # Sleep for the specified duration
    time.sleep(duration)

    # Clear the placeholder to hide the message
    placeholder.empty()

#input
def get_user_input():
    user_choice = st.selectbox("Select Input Type", ["Single Input", "Multiple Inputs"])

    if user_choice == "Single Input":
        single_input = st.text_area("Enter a single value:")
        message = "Please ensure after each dot have space and your sentence does not have repeated word to get accurate result"

        # Using Markdown to make the text italic and small
        formatted_message = f"*{message}*"  # Italics
        formatted_message = f"<small>{formatted_message}</small>"  # Small

        st.write(formatted_message, unsafe_allow_html=True)
        if st.button("Predict Emotion"):
            try:
           
                st.write(single_input)
                # st.write(type(single_input))
                st.write("Prediction Result:")
                result_single = process_single_input(single_input)
                st.dataframe(result_single)
            except Exception as e:
                simulate_popup('This is a pop-up message!', duration=5)
                st.error(f"An error occurred: {e}")
                st.write("Here's what you can do:")
                st.write("1. Check your input data.")
                st.write("2. Make sure after each dot have space and your sentence does not have repeated word to get accurate result.")
                st.write("3. Contact support if the issue persists (wamo995@gmail.com)")
                    
    
    elif user_choice == "Multiple Inputs":
        uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

        if uploaded_file is not None:
            st.subheader("Prediction Result:")
            df_ori, df_output = process_csv_file(uploaded_file)

            st.write("Preview of the uploaded CSV file:")
            st.dataframe(df_output)
            
            
            st.subheader("Result Summary")
            total_review = len(df_ori)
            total_aspect = len(list(df_output['Aspect'].unique()))

            st.markdown("---")
            left_column, right_column = st.columns(2)
            with left_column:
                st.subheader("Total Review:")
                st.subheader(total_review)
            with right_column:
                st.subheader("Total aspect:")
                st.subheader(total_aspect)
            
            st.markdown("---")
            
            emotion_counts = df_output['Label1'].value_counts()

            # Create a pie chart using Plotly Express
            fig = px.pie(
                emotion_counts,
                names=emotion_counts.index,
                values=emotion_counts.values,
                title='Emotion Occurence in Dataset',
                hole = 0.4
            )

            # Show the plot
            st.plotly_chart(fig)
            

            # Assuming df_output is your DataFrame
            df = df_output.groupby('Aspect')['Label1'].agg(lambda x: x.value_counts().idxmax()).reset_index(name='Label1')

            # Check if df is a Series, convert it to a DataFrame if needed
            if isinstance(df, pd.Series):
                df = df.reset_index()

            # Sidebar filters
            st.sidebar.header('Filter Options')
            selected_aspect = st.sidebar.multiselect('Select Aspect:', ['All'] + list(df_output['Aspect'].unique()))
            selected_label1 = st.sidebar.multiselect('Select Label1:', ['All'] + list(df_output['Label1'].unique()))

            # Apply filters to the DataFrame
            filtered_df = df_output.copy()

            if 'All' not in selected_aspect:
                filtered_df = filtered_df[filtered_df['Aspect'].isin(selected_aspect)]

            if 'All' not in selected_label1:
                filtered_df = filtered_df[filtered_df['Label1'].isin(selected_label1)]

            # Create scatter plot with filtered DataFrame
            fig1 = px.scatter(
                filtered_df,
                x='Aspect',
                y='Label1',
                labels={'x': 'Aspect', 'y': 'Emotion'},
                title='Emotion for Each Aspect',
                height=600,
                width=800
            )

            # Show the plot
            st.plotly_chart(fig1)

            # Calculate the occurrence of each emotion for each unique aspect using Label1
            emotion_counts2 = df_output.groupby(['Aspect', 'Label1']).size().unstack(fill_value=0)
            # Use st.bar_chart
            st.bar_chart(emotion_counts2)
                                            

# Example usage
st.set_page_config(
    page_title="Emotion Detector Based On Text Review",
    page_icon=":grinning:",
    layout="wide"
)


# st.title("Emotion Detector Based on Text Review")
# st.write("Dive into emotion detection with your first review! Uncover the sentiments using AI.")
st.markdown("<h1 style='text-align: center;'>Dive into emotion detection with your review! 😊😢😍</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Uncover the sentiments using AI. Choose your input, and let the emotional insights begin!</p>", unsafe_allow_html=True)
intro_text = """
    <div style="text-align: center;">
        <h2>Welcome to our Nasi Lemak Emotion Detector!</h2>
        <p>Immerse yourself in the world of emotions as we analyze reviews from the iconic Nasi Lemak Wanjo Kg Baru. Our app specializes in detecting sentiments related to the cherished experience of savoring delicious nasi lemak. Dive into the flavorful emotions expressed in each review and uncover the unique sentiments that make Nasi Lemak Wanjo Kg Baru a standout in the world of nasi lemak dining.</p>
    </div>
"""

st.markdown(intro_text, unsafe_allow_html=True)


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
tokenizer = BertTokenizer.from_pretrained(pretrain_model_name)
lr = 2e-5
model_ATE = bert_ATE(pretrain_model_name).to(DEVICE)
optimizer_ATE = torch.optim.Adam(model_ATE.parameters(), lr=lr)
model_ATE = load_model(model_ATE, 'bert_ATE.pkl')
get_user_input()