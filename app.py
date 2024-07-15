from flask import Flask, request, render_template, jsonify, redirect, url_for, flash, Response, stream_with_context
import PyPDF2
import io
from werkzeug.utils import secure_filename
import re
import csv
from gensim.models.doc2vec import Doc2Vec
import numpy as np
from numpy.linalg import norm

import warnings
warnings.filterwarnings("default", category=PendingDeprecationWarning)

from docx import Document
from urllib.parse import quote

# text pre processing with regular expression
def preprocess_text(text):
    text = text.lower()
    text2 = text.lower()
    
    # Remove punctuation
    text = re.sub('[^a-z]', ' ', text)
    
    # Remove numerical values 
    text = re.sub(r'\d+', '', text)
    
    # Remove extra whitespaces
    text = ' '.join(text.split())

    return text

# calculating the similarity score between job description and resume data
def model_prediction(input_CV,input_JD):
    model = Doc2Vec.load('cv_job_maching.model')
    v1 = model.infer_vector(input_CV.split())
    v2 = model.infer_vector(input_JD.split())
    similarity = 100*(np.dot(np.array(v1), np.array(v2))) / (norm(np.array(v1)) * norm(np.array(v2)))
    return round(similarity, 2)

app = Flask(__name__)

# extract text from document file
def extract_text_from_docx(docx_file):
    doc = Document(docx_file)
    text_content = '\n'.join([paragraph.text for paragraph in doc.paragraphs])
    return text_content

# extract text from pdf file
def extract_text_from_pdf(pdf_file):
    pdf_content = io.BytesIO(pdf_file.read())
    pdf_reader = PyPDF2.PdfReader(pdf_content)
    text_content = ""
    for page in pdf_reader.pages:
        text_content += page.extract_text()
    return text_content

# sending result to HTML pages 
def extracting_similarity_score(text, document_files):
    final_result = []
    
    for document_file in document_files:
        if document_file.filename.endswith('.pdf'):
            document_text = extract_text_from_pdf(document_file)
            
        elif document_file.filename.endswith('.docx'):
            document_text = extract_text_from_docx(document_file)
        else:
            # Handle unsupported file types or provide an error message
            flash(f"Unsupported file type: {document_file.filename}", 'error')
            return redirect(url_for('index'))

        score = model_prediction(preprocess_text(text), preprocess_text(document_text))
        print("Score: ", score)
        print("Type of Score: ", type(score))
        file_name = document_file.filename
        final_result.append({'score': score, 'file_name': file_name})

    return final_result

# open the home page at first time
@app.route('/')
def index():
    return render_template('index.html')

# route to the result page
@app.route('/process_data_html', methods=['POST'])
def process_html():
    text_data = request.form.get('text_data')
    document_files = request.files.getlist('document_file')
    
    results = extracting_similarity_score(text_data, document_files)

    
    # Generate CSV data as a string to download as a CSV file of result in the result page
    csv_data = ""
    csv_data += "File Name,Score (%)\n"
    for result in results:
        csv_data += f"{result['file_name']},{result['score']}\n"

    print(results)
    return render_template('result_multiple.html', results=results, csv_data=quote(csv_data))

if __name__ == '__main__':
    app.run(debug=True)

