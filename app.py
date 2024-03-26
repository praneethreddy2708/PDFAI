from flask import Flask, request, render_template, redirect, url_for
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
import os

app = Flask(__name__)
load_dotenv()

# Global storage for the knowledge base
knowledge_bases = {}


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'pdf' not in request.files:
            return redirect(request.url)
        pdf = request.files['pdf']
        if pdf.filename == '':
            return redirect(request.url)
        if pdf and pdf.filename.endswith('.pdf'):
            pdf_reader = PdfReader(pdf)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()

            text_splitter = CharacterTextSplitter(
                separator="\n",
                chunk_size=1000,
                chunk_overlap=10,
                length_function=len
            )
            chunks = text_splitter.split_text(text)

            embeddings = OpenAIEmbeddings()
            knowledge_base = FAISS.from_texts(chunks, embeddings)

            # Store knowledge base using a unique identifier, e.g., a session id or similar
            session_id = "unique_session_id"  # This should be replaced with a real session id or other unique identifier
            knowledge_bases[session_id] = knowledge_base

            return render_template('ask_question.html', pdf_uploaded=True, session_id=session_id)
    return render_template('upload.html')


@app.route('/ask', methods=['POST'])
def ask_question():
    session_id = request.form.get('session_id')  # Assume this comes from a hidden input field in 'ask_question.html'
    user_question = request.form['question']
    if user_question and session_id in knowledge_bases:
        knowledge_base = knowledge_bases[session_id]
        docs = knowledge_base.similarity_search(user_question)

        llm = OpenAI()
        chain = load_qa_chain(llm, chain_type="stuff")
        with get_openai_callback() as cb:
            response = chain.run(input_documents=docs, question=user_question)

        return render_template('answer.html', answer=response)
    return render_template('ask_question.html', error="There was a problem processing your question.")


if __name__ == '__main__':
    app.run(debug=True)
