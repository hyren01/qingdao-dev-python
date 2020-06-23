#! /usr/bin/env python
# -*- coding: utf-8 -*-

import re
import langid


def clear_web_data(content):
    """
    清理文本中的HTML、固定格式字符等。
    :param content: string.文本。
    :return string.清洗完的文本
    """
    if content is None or content == '':
        return ''
    content = re.sub('<[^<]+?>', '', content).replace('&nbsp;', '')  # 可以考虑正则穷举HTML标签进行替换
    content = re.sub(r'&#\d+;', '', content).replace('\n', '')
    content = re.sub('【[^【]+?】', '', content).strip()
    language_id = langid.classify(content)[0]
    if language_id in ('ja', 'zh'):  # 日语及中文
        content = __clear_unnecessary_content(content, True)
    else:
        content = __clear_unnecessary_content(content, False)

    return content


def __clear_unnecessary_content(content, is_chinese_or_japanese):
    """
    清理文本中的无关内容。
    :param content: string.文本。
    :param is_chinese_or_japanese: boolean.是否为中文（包括日文）。
    :return string.清洗完的文本
    """
    # 处理中英文下的问题，如：XXXXX。（完）    资料图片：LG化学南京电池厂全景 韩联社/LG化学供图（图片严禁转载复制）
    if is_chinese_or_japanese and (content.endswith("。") is not True):
        sentences = re.split('([。])', content)
        if len(sentences) != 1:
            del sentences[len(sentences) - 1]
        content = "".join(sentences)
    elif is_chinese_or_japanese is not True and content.endswith(".") is not True:
        sentences = re.split('([.])', content)
        if len(sentences) != 1:
            del sentences[len(sentences) - 1]
        content = "".join(sentences)
    else:
        pass
    content = content.strip()
    content = content.replace("■", "").replace("▲", "").replace("©", "").replace("S。", "")
    if is_chinese_or_japanese and content.__contains__("（完）"):
        content = content.split("（完）")
        del content[len(content) - 1]
        content = "".join(content)
    if is_chinese_or_japanese is not True and content.__contains__(" (End) "):
        content = content.split(" (End) ")
        del content[len(content) - 1]
        content = "".join(content)

    return content


if __name__ == '__main__':
    # import json
    # import psycopg2
    # import psycopg2.extras
    # postgres_db = psycopg2.connect(host="139.9.126.19", port=31001, database="ebmdb2", user="jdqd", password="jdqd")
    # cursor = postgres_db.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    # cursor.execute("SELECT article_id, content FROM t_article_msg")
    # result = cursor.fetchall()
    # result = json.loads(json.dumps(result))
    # for row in result:
    #     content = row["content"]
    #     if content is None:
    #         continue
    #     content = clear_web_data(content)
    #     cursor.execute("UPDATE t_article_msg SET content_cleared=%s WHERE article_id=%s", [content, row["article_id"]])
    #     print(content)
    # postgres_db.commit()
    # postgres_db.close()
    content = "< img src=\"/p_img/arirangmeari_1139_1.jpg\"/>< img src=\"/p_img/arirangmeari_1139_2.jpg\"/>"
    content = clear_web_data(content)
    print(content)
