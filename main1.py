from flask import Flask, request,render_template,jsonify,send_file,redirect,url_for
import os
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import random
import uuid
import numpy as np

app = Flask(__name__)

stop_words = set(stopwords.words('english'))

idx = []
interview = False

def preprocess(answer):
    answer = answer.lower()
    tokens = word_tokenize(answer)
    tokens = [token for token in tokens if token not in stop_words]
    return tokens

def give_similarity(correct_answer,student_answer):
    correct_tokens = preprocess(correct_answer)
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([correct_answer])
    important_keywords = [vectorizer.get_feature_names_out()[i] for i in tfidf_matrix.nonzero()[1]]

    student_tokens = preprocess(student_answer)
    present_keywords = set(important_keywords) & set(student_tokens)

    student_tfidf = vectorizer.transform([student_answer])
    similarity = cosine_similarity(tfidf_matrix, student_tfidf)[0][0]
    return similarity * 100

def generate_unique_random_numbers():
    try:
        random_numbers = random.sample(range(1, 51), 2)
        return random_numbers
    except ValueError:
        return None


def questions_from_topics(file,topics,noQuestion):
    global idx
    df = pd.read_csv(file)
    # Extract unique questions for the user-provided topics
    selected_questions = {}
    for topic in topics:
        topic_questions = df[df['Topic'] == topic]['Question'].unique().tolist()
        random.shuffle(topic_questions)  # Shuffle the questions for each topic
        selected_questions[topic] = topic_questions

    # Calculate the total number of questions required
    total_questions = noQuestion  # Total number of questions needed

    # Randomly select questions until total number of questions equals specified total
    remaining_questions = []
    selected_topics = set()
    for topic, topic_questions in selected_questions.items():
        selected_topics.add(topic)
        remaining_questions.extend([(topic, question) for question in topic_questions])

    if len(remaining_questions) > total_questions:
        remaining_questions = random.sample(remaining_questions, total_questions)

    idx = []
    for topic, question in remaining_questions:
        idx.append(df[(df['Topic'] == topic) & (df['Question'] == question)]['Sr No'].values[0])


    return remaining_questions,idx


@app.route('/')
def main():
    return render_template('index.html')

@app.route('/record', methods=['GET'])
def record():
    return render_template('recoder.html')

@app.route('/dsa', methods=['GET'])
def dsatest():
    global idx
    df = pd.read_csv('data/DataScience.csv')
    idx = generate_unique_random_numbers()
    while idx is None:
        idx = generate_unique_random_numbers()
    questions = list(df['Question'])
    temp_questions = []
    for i in idx:
        temp_questions.append(questions[i])
    return render_template('das.html',questions=temp_questions)

@app.route('/upload', methods=['POST'])
def upload_recordings():
    print('Recording Received . . .')
    from model import pipe
    print('Model Load . . .')
    recordings_dir = 'recordings'
    os.makedirs(recordings_dir, exist_ok=True)
    recordings = request.files

    transcriptions = []

    for recording in recordings.values():
        recording_path = os.path.join(recordings_dir, recording.filename)
        recording.save(recording_path)

        # Read the audio file
        with open(recording_path, 'rb') as audio_file:
            audio_bytes = audio_file.read()

        # Pass the audio bytes to the pipe function
        transcription = pipe(audio_bytes)["text"]
        transcriptions.append(transcription)

        os.remove(recording_path)
    df = pd.read_csv('data/DataScience.csv')
    answers = list(df['Answer'])

    insights = 'Weak Topics are'
    similar_score = 0
    print(idx)
    if 'test_id' in request.form:
        for i in range(len(transcriptions)):
            print('Answer : ',answers[idx[i]-1])
            print('Transcription : ',transcriptions[i])
            similar = give_similarity(answers[idx[i]-1], transcriptions[i]) 
            similar_score += similar
            if similar < 50 :
                print('Topic ===>',df.loc[df['Sr No'] == idx[i], 'Topic'].values[0])
                topic = df.loc[df['Sr No'] == i+1, 'Topic'].values[0]
                insights += ' '+topic
        similar_score = similar_score / len(transcriptions)+1
        testId = request.form['test_id']
        user_id = request.form['user_id']
        questions = ''
        d_q = list(df['Question'])
        for i in idx:
            questions += ','+str(d_q[i-1])
        last_row_no = 0
        if os.path.exists('Test/user_details.csv'):
            df_existing = pd.read_csv('Test/user_details.csv')
            if not df_existing.empty:
                last_row_no = df_existing.index[-1] + 1
            else:
                last_row_no = 1

        # Create a DataFrame with the form data and the new serial number
        data = {'Sr No': [last_row_no + 1], 'Test ID': [testId], 'Name': [user_id],'Score':[similar_score] ,'Questions': [questions[1:]], 'Weak Topics': [insights]}
        new_row = pd.DataFrame(data, index=[0])  # Convert to DataFrame with a single row
        df_combined = pd.concat([df_existing, new_row], ignore_index=True)

        # Append the DataFrame to the CSV file
        df_combined.to_csv('Test/user_details.csv', mode='a', index=False, header=False)
    else:
        for i in range(len(transcriptions)):
            print('Answer : ',answers[idx[i]])
            print('Transcription : ',transcriptions[i])
            similar = give_similarity(answers[idx[i]], transcriptions[i]) 
            similar_score += similar
            if similar < 50 :
                print('Topic ===>',df.loc[df['Sr No'] == idx[i], 'Topic'].values[0])
                topic = df.loc[df['Sr No'] == i+1, 'Topic'].values[0]
                insights += ' '+topic
        similar_score = similar_score / len(transcriptions)+1
    return jsonify({'score':similar_score, 'insights':insights})


@app.route('/dsNotes')
def dsNotes():
    pdf_path = 'data/Notes/DS.pdf'
    return send_file(pdf_path, as_attachment=False, mimetype='application/pdf')

@app.route('/civilNotes')
def civilNotes():
    pdf_path = 'data/Notes/Civil Egg.pdf'
    return send_file(pdf_path, as_attachment=False)

@app.route('/SENotes')
def seNotes():
    pdf_path = 'data/Notes/SD EgG.pdf'
    return send_file(pdf_path, as_attachment=False)

@app.route('/MENotes')
def meNotes():
    pdf_path = 'data/Notes/Mech EGG.pdf'
    return send_file(pdf_path, as_attachment=False)

@app.route('/MarketNotes')
def marketNotes():
    pdf_path = 'data/Notes/Marketing Book.pdf'
    return send_file(pdf_path, as_attachment=False)

@app.route('/AccountsNotes')
def accountNotes():
    pdf_path = 'data/Notes/ACCOUNTANT.pdf'
    return send_file(pdf_path, as_attachment=False)

@app.route('/test',methods=['GET','POST'])
def test():
    if request.method == 'GET':
        return render_template('test.html')
    elif request.method == 'POST':
        test_id = request.form['test_id']
        df = pd.read_csv('Test/test_destails.csv')
        row = df[df['Test ID'] == test_id]
        field = row['Field'].iloc[0]
        topics = row['Topics'].iloc[0]
        topics = topics.split(',')
        noQuestions = row['Total Questions'].iloc[0]
        if field == 'data':
            questions,idx = questions_from_topics('data/DataScience.csv',topics,noQuestions)
            temp_questions = []
            for i in questions:
                temp_questions.append(i[1])
            return render_template('das1.html',questions=temp_questions,Testid = test_id)
        return 'Recieved Test'

@app.route('/createTest',methods=['GET','POST'])
def createTest():
    if request.method == 'GET':
        return render_template('createTest.html')
    elif request.method == 'POST':
        field = request.form['field']
        total_questions = request.form['total-questions']
        topics = ','.join(request.form.getlist('topics[]'))

        # Generate a 10-digit alphanumeric string for the test ID
        test_id = str(uuid.uuid4().hex)[:10]

        # Read existing data from CSV to find the last row number
        last_row_no = 0
        if os.path.exists('Test/test_destails.csv'):
            df_existing = pd.read_csv('Test/test_destails.csv')
            if not df_existing.empty:
                last_row_no = df_existing.index[-1] + 1
            else:
                last_row_no = 1

        # Create a DataFrame with the form data and the new serial number
        data = {'Sr No': [last_row_no + 1], 'Test ID': [test_id], 'Field': [field], 'Total Questions': [total_questions], 'Topics': [topics]}
        new_row = pd.DataFrame(data, index=[0])  # Convert to DataFrame with a single row
        df_combined = pd.concat([df_existing, new_row], ignore_index=True)

        # Append the DataFrame to the CSV file
        df_combined.to_csv('Test/test_destails.csv', mode='a', index=False, header=False)
        

        return jsonify({'Test ID': test_id})

def get_unique_topics(file_name):
    df = pd.read_csv(file_name)
    unique_topics = df['Topic'].unique()
    return unique_topics.tolist()

@app.route('/gettopic', methods=['GET'])
def get_topic():
    field = request.args.get('field')

    if field == 'data':
        topics = get_unique_topics('data/DataScience.csv')
    elif field == 'civil':
        topics = get_unique_topics('data/Civil_datasettopic.csv')
    elif field == 'mechanical':
        topics = get_unique_topics('data/Mech_datasettopic.csv')

    return jsonify({'topics': topics})

@app.route('/admin',methods=['GET','POST'])
def admin():
    if request.method == 'GET':
        return render_template('admin.html')
    elif request.method == 'POST':
        test_id = request.form['test_id']
        print(test_id)
        df = pd.read_csv('Test/user_details.csv')
        filtered_df = df[df['Test ID'] == test_id]
        columns_to_drop = ['Sr No','Test ID','Questions']
        filtered_df.drop(columns=columns_to_drop, inplace=True)
        filtered_df = filtered_df.replace(np.nan, None)
        data_dict = {
            'Name' : filtered_df['Name'].tolist(),
            'Score' : filtered_df['Score'].tolist(),
            # 'Questions' : filtered_df['Questions'].tolist(),
            'Weak Topics' : filtered_df['Weak Topics'].tolist(),
        }
        return jsonify(data_dict)


if __name__ == '__main__':
    app.run(debug=True)