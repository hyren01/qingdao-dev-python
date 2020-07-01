from feedwork.utils import logger


def insert_graph_db(graph_db, rst):
    event_tag = "Event"
    relation_type = "causality"
    for i in range(len(rst)):
        n = rst[i]['event_id_pair'][0]
        n1 = rst[i]['event_id_pair'][1]
        results = graph_db.run(
            "MATCH (n:Event{event_id:'" + n + "'}),(n1:Event{event_id:'" + n1 + "'}) "
            "RETURN CASE WHEN (n)-[]-(n1) THEN '1' ELSE '0' END AS result").data()
        logger.info(results)
        if not results:
            insert(graph_db, event_tag, i, relation_type, rst)
        else:
            if results[0]['result'] == '0':
                logger.info(results[0]['result'])
                insert(graph_db, event_tag, i, relation_type, rst)


def insert(graph_db, event_tag, i, relation_type, rst):
    tx = graph_db.begin_transaction()
    cause_event_id = ""
    effect_event_id = ""
    if rst[i]['relation'] == 1:
        cause_event_id = rst[i]['event_id_pair'][0]
        effect_event_id = rst[i]['event_id_pair'][1]
    elif rst[i]['relation'] == 2:
        cause_event_id = rst[i]['event_id_pair'][1]
        effect_event_id = rst[i]['event_id_pair'][0]
    # todo 目前只有因果关系
    relation_attribute = "name: '{name}'".format(name=relation_type)
    relation_attribute = "{" + relation_attribute + "}"
    relation = {"source_event_tag": event_tag, "target_event_tag": event_tag,
                "source_event_id": cause_event_id, "target_event_id": effect_event_id,
                "relation": relation_type, "relation_attribute": relation_attribute}
    if 'source_event_tag' in relation:
        source_event_tag = relation["source_event_tag"]
    else:
        source_event_tag = event_tag
    if 'target_event_tag' in relation:
        target_event_tag = relation["target_event_tag"]
    else:
        target_event_tag = event_tag
    tx.run(f"MATCH (event1:{source_event_tag}),(event2:{target_event_tag}) WHERE "
           f"event1.event_id = '{relation['source_event_id']}' "
           f"AND event2.event_id = '{relation['target_event_id']}' "
           f"CREATE (event1)-[:{relation['relation']} {relation['relation_attribute']}]->(event2)")
    tx.commit()
