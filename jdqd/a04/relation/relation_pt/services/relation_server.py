from flask import Flask, request
import json
import jdqd.a04.relation.relation_pt.algor.relation_combine as relation_combine

from jdqd.a04.relation.relation_pt.algor import r_parallel, r_choice, r_further, \
    r_assumption, r_then, r_hypernym, r_condition, r_contrast, r_causality

relations = [r_causality, r_assumption, r_condition, r_contrast, r_then,
             r_further, r_choice, r_parallel, r_hypernym]

app = Flask(__name__)

app.config.update(RESTFUL_JSON=dict(ensure_ascii=False))


@app.route("/relation_extract", methods=['GET', 'POST'])
def extract_sentence_from_req():
    sentence = request.form.get('sentence')
    rst, __, __ = relation_combine.extract_all_relations(sentence, relations)
    return json.dumps(rst)


if __name__ == '__main__':
    app.debug = True
    app.run(host='0.0.0.0', port=12315)
