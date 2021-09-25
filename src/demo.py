"""
demo
"""
import os
from typing import Optional
from fastapi import FastAPI
from dotenv import load_dotenv  # for python-dotenv method
from py2neo import Graph

load_dotenv("neo.env")  # for python-dotenv method


password = os.environ.get("neo_password")
app = FastAPI()

graph = Graph("bolt://localhost:7687", auth=("neo4j", password))


@app.get("/")
def read_root():
    """[summary]

    Returns:
        [type]: [description]
    """
    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q_option: Optional[str] = None):
    """[summary]

    Returns:
        [type]: [description]
    """
    return {"item_id": item_id, "q": q_option}


@app.get("/neo/{limit}")
def cypher_result(limit: int):
    """[summary]

    Returns:
        [type]: [description]
    """
    res = graph.run(
        "MATCH p=(:Character)-[:INTERACTS]->(:Character) RETURN p LIMIT %s" % limit
    ).data()
    return {"res": res}
