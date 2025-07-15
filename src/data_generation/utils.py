from openai import AzureOpenAI  # openai>=1.0.0
import time
import json
import networkx as nx
import torch
import pandas as pd
from sentence_transformers import SentenceTransformer
import random
import requests
import logging
import torch
import re

# Global variable to hold API cost across multiple LLM calls
api_total_cost = 0.0

clients = {
    # "gpt-4": {
    #     'endpoint': "YOUR AZURE ENDPOINT",
    #     'api_key': "YOUR API KEY",
    #     'api_version': "2024-12-01-preview",
    #     'name': 'gpt-4-1106-preview-nofilter',
    #     'input_price': 2.75 / 10 ** 6,  # input price per Million tokens
    #     'output_price': 11.0 / 10 ** 6, # output price per Million tokens
    #     },
    # "gpt-4o": {
    #     'endpoint': "YOUR AZURE ENDPOINT",
    #     'api_key': "YOUR API KEY",
    #     'api_version': "2024-12-01-preview",
    #     'name': 'gpt-4o-0806-nofilter-global',
    #     'input_price': 2.75 / 10 ** 6, # input price per Million tokens
    #     'output_price': 11.0 / 10 ** 6, # output price per Million tokens
    #     },
    "qwen3-32b": {
            'endpoint': "http://172.31.11.20:11555/v1",
            'api_key': "at-llm",  # Qwen API key
            'api_version': "",
            'name': 'Qwen3-32B',
        }
}

def init_logger(name=''):
    logger = logging.getLogger(__name__)
    # set the logging level to INFO
    logger.setLevel(logging.INFO)
    
    # 确保logs目录存在
    import os
    logs_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'logs'))
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
    
    # save the log to a file
    handler = logging.FileHandler(os.path.join(logs_dir, '{name}-{time}.log'.format(name=name, time=time.strftime("%Y%m%d-%H%M%S"))))
    handler.setLevel(logging.INFO)
    # create a logging format
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(handler)
    return logger

def get_json_from_generated_text(text):
    if not isinstance(text, str):
        print("❌ get_json_from_generated_text 输入不是字符串，实际类型：", type(text))
        return {"Entity": []}
    match = re.search(r'\{[\s\S]*?\}', text)
    if not match:
        print("❌ 未找到合法的 JSON 字符串，原始内容如下：")
        print(repr(text))
        return {"Entity": []}
    json_str = match.group(0)
    # 单引号转双引号
    json_str_fixed = json_str.replace("'", '"')
    # None转null
    json_str_fixed = json_str_fixed.replace('None', 'null')
    # 结尾多余逗号
    json_str_fixed = re.sub(r',\s*}', '}', json_str_fixed)
    json_str_fixed = re.sub(r',\s*]', ']', json_str_fixed)
    # 多次补逗号：在 "value" "key" 之间加逗号，直到没有可补的
    for _ in range(5):
        json_str_fixed_new = re.sub(r'("[^"]+"\s*:\s*"[^"]+")\s+("[^"]+"\s*:)', r'\1, \2', json_str_fixed)
        json_str_fixed_new = re.sub(r'(\d|true|false|null)\s+("[^"]+"\s*:)', r'\1, \2', json_str_fixed_new, flags=re.IGNORECASE)
        if json_str_fixed_new == json_str_fixed:
            break
        json_str_fixed = json_str_fixed_new
    try:
        json_obj = json.loads(json_str_fixed)
        return json_obj
    except Exception as e:
        print("❌ JSON 解析失败，最终修正内容如下：")
        print(repr(json_str_fixed))
        return {"Entity": []}

def build_graph(graph: list) -> nx.Graph:
    G = nx.Graph()
    for triplet in graph:
        h, r, t = triplet
        G.add_edge(h.lower(), t.lower(), relation=r.lower().strip())
    return G

def get_topk_similar_entities(entity, knowledge_graph, emb_model,nodeemb_dict, k=5, filter_threshold = 0.8):
    entity_type = entity["type"]
    # get the entities set from the graph
    node_entities_with_type = knowledge_graph.query('x_type=="{}"'.format(entity_type))['x_name'].unique()
    embeddings_for_node_entities = nodeemb_dict[entity_type]
    # embeddings_for_node_entities = torch.load('{}/{}.pt'.format(type_embeddings_path, entity_type.replace('/','_')), weights_only= False)
    entity_embedding = emb_model.encode(entity["name"])
    # load the embeddings for the type
    
    similarity = emb_model.similarity(entity_embedding, embeddings_for_node_entities)
    val,idx = torch.topk(similarity, k)
    
    topk_similarity = similarity[0][idx]
    top1_similarity = topk_similarity[0][0].item()
    
    # 过滤相似度大于阈值的索引
    filtered_idx = idx[0][similarity[0][idx[0]] > filter_threshold]
    
    
    if len(filtered_idx) == 0:
        return [], top1_similarity
    elif len(filtered_idx) == 1:
        similar_entity = node_entities_with_type[filtered_idx[0]]
        return [similar_entity], top1_similarity
    else:
        return node_entities_with_type[filtered_idx].tolist(), top1_similarity
    
    
def find_all_path_KG(question_entities,result_entities,G):
    path_all = []
    for q_entity in question_entities:
        for a_entity in result_entities:
            path_all+= list(nx.all_shortest_paths(G, q_entity.lower(), a_entity.lower()))
    return path_all

def generate_node_embeddings(knowledge_graph_path = '/path/to/kg.csv', emb_model_name = 'abhinand/MedEmbed-large-v0.1'):
    knowledge_graph = pd.read_csv(knowledge_graph_path, low_memory=False)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🔧 当前生成节点嵌入时使用设备：{device}")
    emb_model = SentenceTransformer(emb_model_name).to(device)
    types = knowledge_graph['x_type'].unique()
    nodeemb_dict = {}
    for t in types:
        print("generating embeddings for type: ", t)
        entities_in_types = knowledge_graph.query('x_type=="{}"'.format(t))['x_name'].unique()
        type_embeddings = emb_model.encode(list(entities_in_types))
        nodeemb_dict[t] = type_embeddings
    # 保存到正确的位置
    import os
    save_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'node_embeddings.pt'))
    torch.save(nodeemb_dict, save_path)
    print(f"节点嵌入已保存到: {save_path}")
    return

def compute_usage(response, engine):
    usage = response.usage.to_dict()
    input = usage["prompt_tokens"]
    reasoning = 0 if "completion_tokens_details" not in usage else usage["completion_tokens_details"]["reasoning_tokens"]
    output = usage["completion_tokens"] - reasoning

    cost = {
        "input": input * clients[engine]['input_price'],
        "output": output * clients[engine]['output_price'],
    }

    cost["total"] = sum(cost.values())

    return cost

import requests

def run_llm(prompt, temperature=0.0, max_tokens=3000, engine="qwen3-32b", max_attempt=10):
    global api_total_cost
    client_config = clients[engine]

    # 自定义 API 方式（Qwen 本地部署）
    if engine == "qwen3-32b":
        messages = [{"role": "user", "content": prompt}]
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {client_config['api_key']}"
        }
        payload = {
            "model": client_config['name'],
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        attempt = 0
        while attempt < max_attempt:
            attempt += 1
            try:
                response = requests.post(
                    f"{client_config['endpoint']}/chat/completions",
                    headers=headers,
                    json=payload
                )
                if response.status_code == 200:
                    return response.json()["choices"][0]["message"]["content"]
                else:
                    print("Error code:", response.status_code, "-", response.text)
                    time.sleep(2)
            except Exception as e:
                print(e)
                time.sleep(2)
        return "openai error, retry"

    # Azure OpenAI 调用方式
    else:
        from openai import AzureOpenAI
        client = AzureOpenAI(
            azure_endpoint=client_config['endpoint'],
            api_key=client_config['api_key'],
            api_version=client_config['api_version']
        )
        messages = [{"role": "user", "content": prompt}]
        try:
            response = client.chat.completions.create(
                model=client_config['name'],
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            result = response.choices[0].message.content
            # api_total_cost += compute_usage(response, engine)["total"]  # 如果有价格信息可加上
            return result
        except Exception as e:
            print(e)
            return "openai error, retry"


def coarse_entity_extraction(text,temperature = 0.0, max_tokens = 3000, engine="qwen3-32b"):
    Extract_prompt = """ You are a helpful, pattern-following medical assistant. 
Given the text in a medical or biomedical context, precisely extract all entities from the text. 

### Output Format
Strictly follow the JSON structure below. 
The type of each entity MUST STRICTLY BELONG to one type from:
1. gene/protein
2. drug 
3. effect/phenotype 
4. disease 
5. biological_process 
6. molecular_function 
7. cellular_component 
8. exposure 
9. pathway 
10. anatomy

IMPORTANT: Your output MUST be strictly valid JSON, and MUST NOT contain any explanation, thinking, or any other text. Only output the JSON.

```json
{{
"Entity": [
    {{"id": "1", "type": "some_type", "name": "entity_name"}},
    {{"id": "2", "type": "some_type", "name": "entity_name"}},
]
}}
```

### Example
text: 
Course in Hospital: Mr. Johnson arrived in the ER from nursing home with a three-day history of worsening shortness of breath, yellow-green sputum, and increased sputum production. 
He was subsequently diagnosed with a COPD exacerbation and was satting at 84% on 4L O2 by nasal prongs. 
Medical presciptions : TAB PARACIP 500 MG two TABLETS PER ORAL THRICE DAILY AFTER FOOD FOR 5 DAYS INJ. AUGMENTIN 1.2 GM, INTRAVENOUS, THREE TIMES A DAY X 4 DAYS

output:
```json
{{
"Entity": [
    {{"id": "1", "type": "effect/phenotype", "name": "shortness of breath"}},
    {{"id": "2", "type": "effect/phenotype", "name": "yellow-green sputum"}},
    {{"id": "3", "type": "disease", "name": "COPD"}},
    {{"id": "4", "type": "effect/phenotype", "name": "nasal prongs"}},
    {{"id": "5", "type": "drug", "name": "PARACIP"}},
    {{"id": "6", "type": "drug", "name": "AUGMENTIN"}}
]
}}
```

### Input
text: 
{text}

output:
""" 
    prompt = Extract_prompt.format(text=text)
    for _ in range(3):  # 最多重试3次
        result = run_llm(prompt,temperature,max_tokens,engine)
        # 只保留第一个 { 到最后一个 } 之间的内容
        start = result.find("{")
        end = result.rfind("}")
        if start != -1 and end != -1 and end > start:
            json_str = result[start:end+1]
            try:
                json.loads(json_str)
                return result
            except Exception:
                continue
    # 兜底，返回空实体
    return '{"Entity": []}'


def most_correlated_enetity_selection(question, query_entity, similar_entities, temperature = 0.0, max_tokens = 3000, engine="qwen3-32b"):
    Reformat_prompt = """ You are a helpful, pattern-following medical assistant.
    Given a medical question and corresponding answer, an query entity which is extracted from the question, and a list of similar entities.
    Select ONE most correlated entity from the list of similar entities based on the question and query entity.
    SELECTED ENTITY MUST BE IN THE SIMILAR ENTITIES.
    IF there is not suitable entity in the similar entities, directly return the NONE.
    
    ### Output Format
    Strictly follow the JSON structure below:
    ```json
    {{
        "selected_entity": {{
            "name": "selected_entity_name",
            "id": a int number, the index of the selected entity in the similar entities list, from 0 to N-1
            "reason": "reason for choosing this entity"
        }}
    }}
    ```
    
    if there is no suitable entity, return:
    ```json
    {{
        "selected_entity": {{
            "name": "NONE",
            "id": "NONE",
            "reason": "reason for not choosing any entity, you need to explain why the entities in the similar entities list are not suitable"
        }}
    }}
    ```
    
    ### Input:
    question: {question}
    query entity: {query_entity}
    similar entities: {similar_entities}
    
    output:
    """
    # convert the list of similar entities to a string
    similar_entities_str = ', '.join("{}.{}".format(idx, ent) for idx, ent in enumerate(similar_entities))
    prompt = Reformat_prompt.format(question = question, query_entity=query_entity, similar_entities=similar_entities_str)
    result = run_llm(prompt,temperature,max_tokens,engine)
    return result


def QA_reformat_based_on_entity(question, answer, entity_list_text, temperature = 0.0, max_tokens = 5000, engine="qwen3-32b"):
    Reformat_prompt = """ You are a helpful, pattern-following medical assistant.
Given a medical question and answer, and all a list of entities.
You need to reformat the question and answer into a pair of description and conclusion.

MUST MAKE SURE the conclusion and description paragraphs contain the entities in the entity list.
You can reallocate information from the question to the description and conclusion paragraphs, to make sure the entities in the entity list are included in the description and conclusion paragraphs.
However, you CAN NOT ADD ANY INFORMATION that is not in the question or answer.

### Output Format
Strictly follow the JSON structure below.

"""

def QA_reformat_with_entity_extraction(question, answer, knowledge_graph, emb_model, nodeemb_dict, temperature = 0.0, max_tokens = 5000, engine="qwen3-32b"):
    print('【DEBUG】knowledge_graph type:', type(knowledge_graph))
    print('【DEBUG】knowledge_graph columns:', getattr(knowledge_graph, 'columns', None))
    QA_text = f"""Question: {question}\nAnswer: {answer}"""
    all_entities = coarse_entity_extraction(QA_text, temperature, max_tokens, engine)
    all_entities = get_json_from_generated_text(all_entities)
    type_set = set(knowledge_graph['x_type'].unique())
    result_entities = []
    for entity in all_entities["Entity"]:
        if entity["type"] not in type_set:
            continue
        similar_entities, top1_similarity = get_topk_similar_entities(entity, knowledge_graph, emb_model, nodeemb_dict, k=10, filter_threshold=0.7)
        if similar_entities == []:
            continue
        selected_entity = None
        for ent in similar_entities:
            if entity["name"].lower() == ent.lower():
                selected_entity = {"name": ent, "id": str(similar_entities.index(ent))}
                break
        if top1_similarity > 0.85 and selected_entity is None:
            selected_entity = {"name": similar_entities[0], "id": str(0)}
        if selected_entity is None:
            selected_entity = most_correlated_enetity_selection(QA_text, entity["name"], similar_entities)
            selected_entity = get_json_from_generated_text(selected_entity)["selected_entity"]
        if selected_entity["name"] != "NONE":
            result_entities.append(similar_entities[int(selected_entity["id"])])
    result_entities = list(set(result_entities))
    entities_text = '\n'.join([f"{idx+1}.{ent}" for idx, ent in enumerate(result_entities)])
    result = QA_reformat_based_on_entity(question, answer, entities_text, temperature, max_tokens, engine)
    return result