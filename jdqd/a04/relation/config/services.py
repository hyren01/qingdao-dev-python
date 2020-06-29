class Config(object):

    def __init__(self):
        self.http_port = 8084

        self.db_host = "139.9.126.19"
        self.db_port = "31001"
        self.db_name = "ebmdb2"
        self.db_user = "jdqd"
        self.db_passwd = "jdqd"

        self.neo4j_uri = "bolt://172.168.0.115:7687"
        self.neo4j_username = "neo4j"
        self.neo4j_password = "q1w2e3"

        # 事件抽取
        # url_event_extract = "http://172.168.0.115:38082/event_extract"
        self.url_event_extract = "http://127.0.0.1:8083/event_parsed_extract"
        # 提取关键词
        # self.url_relation_keywords = "http://172.168.0.115:12319/relation_keywords"
        # 拆分子句
        # self.url_relation_split = "http://172.168.0.115:12320/relation_split"
        # 事件关系预测
        # self.url_relation_classify = "http://172.168.0.115:12318/relation_classify"
