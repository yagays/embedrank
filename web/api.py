import os
import sys
import json
sys.path.append("../src/")

import json
import bottle
from bottle import route, run, request, response, static_file
from gensim.models.doc2vec import Doc2Vec

from embedrank import EmbedRank
from nlp_uitl import tokenize


@route("/")
def hello():
    return "It works!"


@route("/embedrank", method="POST")
def result():
    embedrank = EmbedRank(model=model, tokenize=tokenize, N=int(request.forms.num_keywords))
    result = embedrank.extract_keyword(request.forms.text)

    response.content_type = 'application/json'
    return json.dumps({"keywords": [{"keyword": t[0], "score":str(t[1])} for t in result]},  ensure_ascii=False)


if __name__ == "__main__":
    model = Doc2Vec.load("../model/jawiki.doc2vec.dbow300d.model")
    run(host="0.0.0.0", port=8080, debug=True)
