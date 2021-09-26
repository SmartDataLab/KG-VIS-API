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


@app.get("/neo/{label}")
def cypher_result(
    label: str,
    limit: Optional[int] = 300,
    id: Optional[int] = 19,
    maxDepth: Optional[int] = 3,
    type_id_str: Optional[str] = "c_id",
):
    """[summary]

    Returns:
        [type]: [description]
    """
    res = graph.run(
        f"MATCH (a:{label}"
        + " {"
        + type_id_str
        + ':"'
        + str(id)
        + '"}'
        + ") WITH id(a) AS startNode CALL gds.alpha.bfs.stream('myGraph', {"
        + f"startNode: startNode, maxDepth: {maxDepth}"
        + "}"
        + f") YIELD path UNWIND [ n in nodes(path) | n ] AS node RETURN node LIMIT {limit}"
    ).data()
    return {"res": res}
