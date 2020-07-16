import json
from feedwork.database.database_wrapper import DatabaseWrapper
import feedwork.utils.DateHelper as date_util
from feedwork.utils import logger
from jdqd.a04.event_interface.services.common.http_util import http_post
from jdqd.a04.event_interface.config.project import Config


config = Config()


db = DatabaseWrapper()
try:
    # 1cd63e541ba6fe91c1b0483516f7dff0
    df = db.query("select article_id,content from t_article_msg_zh where is_relation='0' ")
    for aid, content in zip(df.article_id, df.content):
        logger.info(f"ID = {aid}")
        cur_time = date_util.sys_date("%Y-%m-%d %H:%M:%S")
        logger.info(f"end---cur_time = {cur_time}")
        data = {"content_id": aid, "content": content}
        res = http_post(data, config.relextract_interface_uri)
        response = json.loads(res)
        if response["status"] == "success":
            db.execute(f"update t_article_msg_zh set is_relation='1',finish_date=%s,finish_time=%s "
                       f"where article_id=%s",
                       (date_util.sys_date("%Y-%m-%d"), date_util.sys_time("%H:%M:%S"), aid))
            db.commit()
        else:
            logger.error(f"error,article_id:{aid}")
except Exception as e:
    db.rollback()
    raise RuntimeError(e)
finally:
    db.close()
