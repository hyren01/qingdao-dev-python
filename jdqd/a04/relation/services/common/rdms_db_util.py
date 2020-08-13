import json
from feedwork.database.database_wrapper import DatabaseWrapper
from feedwork.database.database_wrapper import QueryResultType


def insert_event_relations(keyword, left, right, event_pairs, rst, code, sentence_id):
    db = DatabaseWrapper()
    left = str(left).replace("'", "\"")
    right = str(right).replace("'", "\"")
    try:
        if code != 1 and code != 4:
            db.execute(f"insert into event_relations(sentence_id,relation_type ,"
                       f" words, left_sentence, right_sentence, event_source, event_target)"
                       f"values(%s,%s,%s,%s,%s,%s,%s)", (
                           sentence_id, code, keyword, left, right, event_pairs, rst))
        elif code == 4:
            for i in range(len(rst)):
                db.execute(f"insert into event_relations(sentence_id,relation_type,words,"
                           f"left_sentence, right_sentence, event_source, event_target,relation,"
                           f"event_source_id,event_target_id)"
                           f"values(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)", (
                               sentence_id, code, keyword, left, right,
                               rst[i]['event_pair'][0], rst[i]['event_pair'][1], rst[i]['relation'],
                               rst[i]['event_id_pair'][0], rst[i]['event_id_pair'][1]))
        db.commit()
    except Exception as ex:
        db.rollback()
        raise RuntimeError(f"{sentence_id},ex.msg={ex}")
    finally:
        db.close()


def get_event_info(event_id):
    db = DatabaseWrapper()
    try:
        event_info = db.query(f"select shorten_sentence,verb,triggerloc_index from ebm_event_info "
                              f"where event_id='{event_id}'")
        return event_info
    except Exception as ex:
        db.rollback()
        raise RuntimeError(f"{event_id},ex.msg={ex}")
    finally:
        db.close()
