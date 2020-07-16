import json
from urllib.parse import urlencode
from urllib.request import urlopen
from feedwork.database.database_wrapper import DatabaseWrapper
from feedwork.database.enum.query_result_type import QueryResultType

from jdqd.a04.event_interface.services.common.translate_util import translate_any_2_anyone
from feedwork.utils import logger
import feedwork.utils.DateHelper as date_util
from jdqd.a04.event_interface.services.common.http_util import http_post
from jdqd.a04.event_interface.config.project import Config


config = Config()


def article_zdxj(article_id):
    db = DatabaseWrapper()
    try:
        url = config.coref_interface_uri
        article = db.query(f"SELECT article_id,content FROM t_article_msg_en "
                           f"where is_zdxj='0' and article_id='{article_id}'")
        for article_id, content in zip(article.article_id, article.content):
            logger.info(f"ID = {article_id}")
            # 调用指代消解接口
            data = {'content': content}
            result = http_post(data, url)
            result = json.loads(result)
            logger.info(f"ID = {article_id}")
            if result["status"] == "success":
                article_title = db.query(f"SELECT title FROM t_article_msg where article_id='{article_id}'", (),
                                         QueryResultType.PANDAS)
                title = ""
                if len(article_title.title) > 0:
                    title = translate_any_2_anyone(article_title.title[0], "zh")
                    # 数据插入表
                    db.execute("INSERT INTO t_article_msg_zh(article_id, content,title) VALUES(%s,%s,%s)",
                               (article_id, result["coref"], title))
                    db.execute(
                        "update t_article_msg_en set is_zdxj='1',finish_date=%s,finish_time=%s where article_id=%s",
                        (date_util.sys_date("%Y-%m-%d"), date_util.sys_time("%H:%M:%S"), article_id))
                    db.commit()
                return "success"
            else:
                logger.info(f"error article_id:{article_id}")
                return "error"
    except Exception as e:
        db.rollback()
        raise RuntimeError(e)
    finally:
        db.close()
