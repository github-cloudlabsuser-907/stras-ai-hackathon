from flask import Flask, Response, request, jsonify
from utils.helper import LLMHelper
from utils.azureblobstorage import AzureBlobStorageClient

import os
import chardet
import mimetypes
import json
from http import HTTPStatus
from io import BytesIO
from urllib import parse
from docx import Document
from utils.documentparser import DocumentParser
from urllib.parse import unquote
app = Flask(__name__)
api_base = "/api"

@app.route(api_base)
def test_api():
    print("access granted")
    
@app.route(f'{api_base}/explain_question', methods=['POST'])
def explain_question():
    helper = LLMHelper()
    store = AzureBlobStorageClient()
    files = store.get_all_files()

    if len(files) == 0:
        helper.upload_document()

    evol_similarity_result = helper.similarity_search_with_score(request.form["question"], k=5)
    similarity_result_dict = helper.group_content(evol_similarity_result) #contains score
    context = ''
    for sim in similarity_result_dict:
        context += sim['content']
    result = helper.knowledge_search(request.form["question"], context)
    return json.dumps(result)



if __name__ == '__main__':
    app.run(use_reloader=False)