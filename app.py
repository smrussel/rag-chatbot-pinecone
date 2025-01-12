from config import *

import tiktoken
import pinecone
import uuid
import sys
import logging

from flask import Flask, jsonify, render_template
from flask_cors import CORS, cross_origin
from flask import request

from handle_file import handle_file,handle_url
from answer_question import get_answer_from_files

# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s [%(levelname)s] %(message)s",
#     handlers=[
#         logging.FileHandler("debug.log"),
#         logging.StreamHandler(sys.stdout)
#     ]
# )

# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s [%(levelname)s] %(message)s",
#     handlers=[
#         logging.FileHandler("debug.log"),
#         logging.StreamHandler(sys.stdout)
#     ]
# )

# temp_namespace = 'test-namespace'


def load_pinecone_index() -> pinecone.Index:
    """
    Load index from Pinecone, raise error if the index can't be found.
    """
    pinecone.init(
        api_key=PINECONE_API_KEY,
        environment=PINECONE_ENV,
    )
    index_name = PINECONE_INDEX
    if not index_name in pinecone.list_indexes():
        # print(pinecone.list_indexes())
        raise KeyError(f"Index '{index_name}' does not exist.")
    index = pinecone.Index(index_name)

    return index

def create_app():
    pinecone_index = load_pinecone_index()
    tokenizer = tiktoken.get_encoding("gpt2")
    session_id = str(uuid.uuid4().hex)
    app = Flask(__name__)
    app.pinecone_index = pinecone_index
    app.tokenizer = tokenizer
    app.session_id = session_id
    
    # pinecone_index.delete(delete_all=True,namespace='80e8196c94494fe09b53b8c4d2dee64e')
    # pinecone_index.delete(delete_all=True,namespace='fb14e9df251741d7883e51e534456c6f')
    # log session id
    # logging.info(f"session_id: {session_id}")
    print(f"session_id: {session_id}")
    app.config["file_text_dict"] = {}
    CORS(app, supports_credentials=True)

    return app

app = create_app()

@app.route("/")
def home_view():
    return render_template('index.html')


@app.route(f"/process_file", methods=["POST"])
@cross_origin(supports_credentials=True)
def process_file():
    try:
        file = request.files['file']
        temp_namespace = request.form.get('nameSpace')
        logging.info(str(file))
        handle_file(
            file, temp_namespace, app.pinecone_index, app.tokenizer)
        return jsonify({"success": True})
    except Exception as e:
        logging.error(str(e))
        return jsonify({"success": False})


@app.route(f"/process_url", methods=["POST"])
@cross_origin(supports_credentials=True)
def process_url():
    try:
        url = request.form.get('url')
        temp_namespace = request.form.get('nameSpace')
        # print(url)
        handle_url(
            url, temp_namespace, app.pinecone_index, app.tokenizer)
        return jsonify({"success": True})
    except Exception as e:
        logging.error(str(e))
        return jsonify({"success": False})

@app.route(f"/answer_question", methods=["POST"])
@cross_origin(supports_credentials=True)
def answer_question():
    try:
        params = request.get_json()
        question = params["question"]
        temp_namespace = params['nameSpace']
        # print(question)
        answer_question_response = get_answer_from_files(
            question, temp_namespace, app.pinecone_index)
        return answer_question_response
        # return jsonify({"answer": "Sorry something went wrong, Please try again."})
    except Exception as e:
        print(str(e))
        return jsonify({"success": False,"answer": "Sorry something went wrong, Please try again."})

@app.route("/healthcheck", methods=["GET"])
@cross_origin(supports_credentials=True)
def healthcheck():
    return "OK"

if __name__ == "__main__":
    app.run(debug=True, port=SERVER_PORT, threaded=True)
