import sqlite3
from datasets import Dataset

def get_tables(db: str | sqlite3.Connection):
    if isinstance(db, str):
        con = sqlite3.connect(db)
    elif isinstance(db, sqlite3.Connection):
        con = db
    else:
        raise ValueError(
            f'invalid type of `db`: {type(db)}. '
            f'should be either `str` or `sqlite3.Connection`.'
        )
    cur = con.cursor()
    tables = list(list(zip(*cur.execute(
        'select name from sqlite_master where type="table";'
    ).fetchall()))[0])

    columns = dict()
    for name in tables:
        tbl_info = cur.execute(f"PRAGMA table_info({name});").fetchall()
        _, cols, _, _, _, _ = zip(*tbl_info)
        columns[name] = list(cols)
    return columns

def cursor_generator(cursor, keys):
    for row in cursor:
        yield dict(zip(keys, row))
        
def make_ds(cursor, keys):
    return Dataset.from_dict(dict(zip(keys, zip(*cursor.fetchall()))))

def cur_gen(cur, batch_size=1000, keys=None, list_of_dicts=False):
    while True:
        batch = cur.fetchmany(batch_size)
        if not batch:  # No more rows to fetch
            break
        if keys:
            if list_of_dicts:
                yield [
                    dict(zip(keys, values)) for values in batch 
                ]
            else:
                yield dict(zip(keys, zip(*batch)))
        else:
            yield batch