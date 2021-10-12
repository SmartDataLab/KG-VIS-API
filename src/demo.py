"""
demo
"""
import os
from typing import Optional
from fastapi import FastAPI
from dotenv import load_dotenv  # for python-dotenv method
from py2neo import Graph
from fastapi.middleware.cors import CORSMiddleware
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse

load_dotenv("neo.env")  # for python-dotenv method


password = os.environ.get("neo_password")
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
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


def process_output(data: dict):
    """[summary]

    Args:
        data (dict): [description]

    Returns:
        [type]: [description]
    """
    cate2idx = {"c": 0, "f": 1, "a": 2, "k": 3, "o": 4}
    categories = [
        {"name": "CASE"},
        {"name": "FACT"},
        {"name": "ACTION"},
        {"name": "KEYWORD"},
        {"name": "OBJECT"},
    ]
    nodes = []
    links = []
    for one in data:
        for key, value_dict in one.items():
            if key == "node":
                node = value_dict.copy()
                for attr_name, attr_value in value_dict.items():
                    if "id" in attr_name:
                        id_name = attr_name
                        id_value = int(attr_value)
                        node["id"] = id_value + cate2idx[id_name.split("_")[0]] * 1000
                        node["category"] = cate2idx[id_name.split("_")[0]]
                    else:
                        node["name"] = attr_value
                nodes.append(node)
            else:
                link = {"source": "0", "target": "1"}
                links.append(link)
    return {"nodes": nodes, "links": links, "categories": categories}


@app.get("/neo/{label}")
def cypher_result(
    label: str,
    limit: Optional[int] = 300,
    id: Optional[int] = 19,
    maxDepth: Optional[int] = 3,
    type_id_str: Optional[str] = "c_id",
):
    """[summary]

    Args:
        label (str): [description]
        limit (Optional[int], optional): [description]. Defaults to 300.
        id (Optional[int], optional): [description]. Defaults to 19.
        maxDepth (Optional[int], optional): [description]. Defaults to 3.
        type_id_str (Optional[str], optional): [description]. Defaults to "c_id".

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
    js_data = jsonable_encoder(process_output(res))
    return JSONResponse(js_data)
