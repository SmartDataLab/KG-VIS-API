from typing import Optional

from fastapi import FastAPI
from dotenv import load_dotenv  # for python-dotenv method
from py2neo import Graph

load_dotenv("neo.env")  # for python-dotenv method

import os

password = os.environ.get("neo_password")
app = FastAPI()


print(password)
graph = Graph("bolt://localhost:7687", auth=("neo4j", password))


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Optional[str] = None):
    return {"item_id": item_id, "q": q}


@app.get("/neo/{limit}")
def cypher_result(limit: int, q: Optional[str] = None):
    res = graph.run(
        "MATCH p=(:Character)-[:INTERACTS]->(:Character) RETURN p LIMIT %s" % limit
    ).data()
    return {"res": res}
