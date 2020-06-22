from feedwork.database.bean.database_config import DatabaseConfig
from feedwork.database.database_wrapper import DatabaseWrapper
from feedwork.database.enum.query_result_type import QueryResultType


def query_table(table_name):
    # 每次查询建立一次连接
    database_config = DatabaseConfig()
    database_config.name = 'alg'
    db = DatabaseWrapper(database_config)
    try:
        sql = f"SELECT * FROM {table_name}"
        result = db.query(sql, (), result_type=QueryResultType.DB_NATURE)
        result = [list(r.values()) for r in result]
        return result
    except Exception as e:
        raise RuntimeError(f"Query table error! {str(e)}")
    finally:
        # db不可能为None
        db.close()


# TODO 暂时这么写，代码重构要一步步来
def query_table_2pandas(table_name):
    # 每次查询建立一次连接
    database_config = DatabaseConfig()
    database_config.name = 'alg'
    db = DatabaseWrapper(database_config)
    try:
        sql = f"SELECT * FROM {table_name}"
        result = db.query(sql, (), result_type=QueryResultType.PANDAS)
        return result
    except Exception as e:
        raise RuntimeError(f"Query table error! {str(e)}")
    finally:
        # db不可能为None
        db.close()


def query(sql, db: str = 'alg', parameter: tuple = ()):
    """
    从数据库查询数据
    Args:
      sql:
      parameter:
      db: 连接的数据库名称. 'alg' 为算法所需源数据所在库, 如需指定算法生成数据及应用端数据
      所在库, 将此参数指定为其他名称即可, 推荐使用 'mng'

    Returns:
      查询结果
    """
    if not sql:
        raise RuntimeError("The sql must be not none!")
    database_config = DatabaseConfig()
    database_config.name = db
    db = DatabaseWrapper(database_config)
    try:
        result = db.query(sql, parameter, QueryResultType.DB_NATURE)
        result = [list(r.values()) for r in result]
        return result
    except Exception as e:
        raise RuntimeError(f"The query error! {e}")
    finally:
        db.close()


def modify(sql, parameters=(), error=''):
    """
    根据 sql 对数据进行增删改操作
    Args:
      sql:
      parameters:
      error: 操作出错时则日志中输出的错误信息
    """
    if not sql:
        raise RuntimeError("The sql must be not none!")

    database_config = DatabaseConfig()
    database_config.name = 'mng'
    db = DatabaseWrapper(database_config)
    sqls = [sql] if type(sql) == str else sql
    try:
        db.begin_transaction()
        for index, execute_sql in enumerate(sqls):
            if parameters:
                db.execute(execute_sql, parameters[index])
            else:
                db.execute(execute_sql)
        db.commit()
    except Exception as e:
        db.rollback()
        raise RuntimeError(f'{error}: {str(e)} ' if error else error)
    finally:
        db.close()
