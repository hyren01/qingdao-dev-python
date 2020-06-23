from flask import Flask, g, request
from neo4j import GraphDatabase, basic_auth
from jdqd.relation_extract2.config.services import Config
from jdqd.relation_extract2.services.relation_combine.relation_combine import extract

app = Flask(__name__)
config = Config()
gdb_driver = GraphDatabase.driver(config.neo4j_uri, auth=basic_auth(config.neo4j_username, config.neo4j_password))


def get_gdb():
    if not hasattr(g, 'neo4j_db'):
        g.neo4j_db = gdb_driver.session()
    return g.neo4j_db


@app.route("/get_event_relations", methods=['POST'])
def get_event_relations():
    try:
        content_id = request.form.get('content_id')
        content = request.form.get('content')
    except KeyError:
        return {"status": "error"}
    else:
        graph_db = get_gdb()
        return extract(graph_db, content, content_id)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=config.http_port)
