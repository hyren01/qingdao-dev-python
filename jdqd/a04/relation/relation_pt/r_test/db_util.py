import psycopg2


def get_conn(database, user, password, host, port):
    conn = psycopg2.connect(database=database, user=user, password=password, host=host, port=port)
    return conn


def query(conn, sql):
    cursor = conn.cursor()
    cursor.execute(sql)
    results = cursor.fetchall()
    return results


def modify(conn, sql):
    cursor = conn.cursor()
    cursor.execute(sql)
    conn.commit()
