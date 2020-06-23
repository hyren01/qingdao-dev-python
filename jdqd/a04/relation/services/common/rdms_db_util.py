import json
import time
from feedwork.database.database_wrapper import DatabaseWrapper


def insert_event_relations(keyword, left, right, event_pairs, rst, code, sentence_id):
    db = DatabaseWrapper()
    left = str(left).replace("'", "\"")
    right = str(right).replace("'", "\"")
    try:
        if code != 1 and code != 4:
            db.execute(f"insert into event_relations(sentence_id,relation_type ,"
                       f" words, left_sentence, right_sentence, event_source, event_target)"
                       f"values(%s,%s,%s,%s,%s,%s,%s)", (
                           sentence_id, code, json.dumps(keyword, ensure_ascii=False),
                           json.dumps(left, ensure_ascii=False),
                           json.dumps(right, ensure_ascii=False), json.dumps(event_pairs, ensure_ascii=False),
                           json.dumps(rst, ensure_ascii=False)))
        elif code == 4:
            for i in range(len(rst)):
                db.execute(f"insert into event_relations(sentence_id,relation_type,words,"
                           f"left_sentence, right_sentence, event_source, event_target,relation,"
                           f"event_source_id,event_target_id)"
                           f"values(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)", (
                               sentence_id, code, json.dumps(keyword, ensure_ascii=False),
                               json.dumps(left, ensure_ascii=False), json.dumps(right, ensure_ascii=False),
                               rst[i]['event_pair'][0], rst[i]['event_pair'][1], rst[i]['relation'],
                               rst[i]['event_id_pair'][0], rst[i]['event_id_pair'][1]))
        db.commit()
    except Exception as ex:
        db.rollback()
        raise RuntimeError(f"{sentence_id},ex.msg={ex}")
    finally:
        db.close()


# def update(article_id):
#     db = DatabaseWrapper()
#     finish_date = time.strftime("%Y-%m-%d", time.localtime())
#     finish_time = time.strftime("%H:%M:%S", time.localtime())
#     try:
#         db.execute(f"update t_article_msg_zh set is_relation='1',finish_date=%s,finish_time=%s "
#                    f"where article_id=%s",
#                    (finish_date, finish_time, article_id))
#         db.commit()
#     except Exception as ex:
#         db.rollback()
#         raise RuntimeError(f"{article_id},ex.msg={ex}")
#     finally:
#         db.close()
