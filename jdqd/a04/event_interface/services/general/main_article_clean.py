import json
from feedwork.database.database_wrapper import DatabaseWrapper
import feedwork.utils.DateHelper as date_util
from feedwork.database.enum.query_result_type import QueryResultType
from jdqd.a04.event_interface.config.project import Config

config = Config()


def article_clean(article_id, content):
    db = DatabaseWrapper()
    try:
        db.execute(
            "update t_article_msg set is_clean='1',content_cleared=%s,"
            "clean_finish_date=%s,clean_finish_time=%s "
            "where article_id=%s",
            (content, date_util.sys_date("%Y-%m-%d"), date_util.sys_time("%H:%M:%S"), article_id))
        db.commit()
    except Exception as e:
        db.rollback()
        raise RuntimeError(e)
    finally:
        db.close()
