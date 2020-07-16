import time
from jdqd.a04.event_interface.services.common.translate_util import translate_any_2_anyone
from feedwork.database.database_wrapper import DatabaseWrapper
from feedwork.utils import logger
import feedwork.utils.DateHelper as date_util
from langdetect import detect


def article_en(article_id):
    db = DatabaseWrapper()
    try:
        article = db.query(f"select article_id,content_cleared from t_article_msg "
                           f"where is_translated = '0' and is_clean='1' and article_id='{article_id}' ")
        for article_id, content_cleared in zip(article.article_id, article.content_cleared):
            logger.info(f"ID = {article_id}")
            article_detect = detect(content_cleared)
            if article_detect != "en":
                content = translate_any_2_anyone(content_cleared, "en")
            else:
                content = content_cleared
            cur_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            logger.info(f"end---cur_time = {cur_time}")
            if content != "":
                content = str(content).replace("'", "\"")
                db.execute("insert into t_article_msg_en(article_id,content) values(%s,%s)",
                           (article_id, content))
                db.execute(
                    "update t_article_msg set is_translated='1',translated_finish_date=%s,translated_finish_time=%s "
                    "where article_id=%s",
                    (date_util.sys_date("%Y-%m-%d"), date_util.sys_time("%H:%M:%S"), article_id))
                db.commit()
                return "success"
            else:
                logger.info(f"error:{article_id}")
                return "error"
    except Exception as e:
        db.rollback()
        raise RuntimeError(e)
    finally:
        db.close()
