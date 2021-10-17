# KG-VIS-API

## before commit

- check the pylint and fix it
```bash
cd src && pylint `git ls-files|grep .py$| grep -v ^AugModel.py$ | grep -v ^model.py$ |xargs`
```

## quick start

```bash
uvicorn demo:app --host=0.0.0.0 --port=4002 --reload
```

## format to visual

- cypher output to json format for echarts

```json
{
  "nodes": [
    { "id": "0", "symbolSize": 20.1, "x": 10, "y": 20, "category": 1 },
    { "id": "1", "symbolSize": 20.1, "x": 10, "y": 20, "category": 0 }
  ],
  "links": [
    { "source": "0", "target": "1" },
    { "source": "1", "target": "0" }
  ],
  "categories": [{ "name": "CASE" }, { "name": "ACTION" }]
}
```

这个

```json
{
  "res": [
    { "node": { "c_id": "19", "case": "徐某某盗窃罪一审刑事判决书" } },
    { "node": { "keyword": "罚金", "k_id": "0" } },
    { "node": { "keyword": "有期徒刑", "k_id": "1" } },
    { "node": { "keyword": "从轻处罚", "k_id": "4" } },
    { "node": { "keyword": "并处罚金", "k_id": "6" } },
    { "node": { "keyword": "行政处罚", "k_id": "60" } },
    {
      "node": {
        "fact": "本案于2015年3月30日被行政拘留15日，同年4月9日被刑事拘留",
        "f_id": "488"
      }
    },
    { "node": { "fact": "同年4月21日被逮捕。", "f_id": "489" } }
  ]
}
```

- option for echarts graph

https://echarts.apache.org/en/option.html#series-graph

```json
{
  "nodes": [
    {
      "id": "0",
      "symbol": "rect",
      "symbolSize": 20.1,
      "x": 10,
      "y": 20,
      "category": 1
    },
    {
      "id": "1",
      "symbolSize": 20.1,
      "x": 10,
      "y": 20,
      "category": 0,
      "itemStyle": { "color": "rgb(255,255,255)" }
    }
  ],
  "links": [
    {
      "source": "0",
      "target": "1",
      "value": 10,
      "lineStyle": { "color": "rgb(255,255,255)" }
    },
    { "source": "1", "target": "0", "value": 4, "symbolSize": 20.1 }
  ],
  "categories": [{ "name": "CASE" }, { "name": "ACTION" }]
}
```
