import json
import networkx as nx
import torch
import pandas as pd
from sentence_transformers import SentenceTransformer
# from data import QADataset
import yaml
import os
import argparse
from tqdm import tqdm
import multiprocessing
# from data_generation.data.utils import question_parsers
from data import QADataset
import utils

def reasoning_generation(question,          # original question
                            answer,             # original answer
                            kg,
                            emb_model,
                            nodeemb_dict,
                            topK_reasoning_paths = 3,
                            max_path_number_per_group = 50,
                            temperature = 0.0,
                            max_tokens = 5000,
                            engine="qwen3-32b",  # or "gpt-4o"
                            logger=utils.init_logger(name="reasoning_generation"),
                            filter_path=False):
    answer_prompt2 = """ 
    You are an expert in the medical domain.
Given a medical question, a set of reasoning paths, and a provided answer, your task is to reason step by step as if you are independently determining the possible reasoning paths and deriving the correct answer without prior knowledge of the given answer.

1. Explore the question and pretend to generate multiple plausible reasoning paths. If any of the provided paths seem useful, incorporate them naturally as if you discovered them yourself.
2. If none of the given paths seem relevant or correct, ignore them and generate your own reasoning approach based on your expertise.
3. Analyze and evaluate the reasoning paths carefully, expanding on the most relevant ones to construct a logical, well-supported explanation.
4. Do not mention the existence of predefined reasoning paths or the provided answer in your response.
5. Do not assume the given answer is correct. Instead, determine the answer solely based on your reasoning.
6. If your final conclusion contradicts the given answer, acknowledge potential discrepancies (e.g., "Wait, there might be something wrong") and refine your response accordingly.

Input:
Question: {question}
Answer: {answer}
Paths: {paths}
Output:
Finding reasoning paths:
(you "discover" potential reasoning paths yourself by using the given paths if useful or generating your own if not. It should be concise as a list of paths)
Reasoning Process:
(Step-by-step reasoning process, do not assume the given answer is correct and do not mention the existence of answer.)
Conclusion:
(The final answer derived from your reasoning.)

"""
    logger.info(f"Total API cost so far (accumulated in run_llm calls): {utils.api_total_cost}")
    reformat_result = utils.QA_reformat_with_entity_extraction(question, answer, kg, emb_model, nodeemb_dict)
    reformat_result = utils.get_json_from_generated_text(reformat_result)
    
    reformat_question, question_entities = reformat_result["description"]["text"], reformat_result["description"]["entities"]
    reformat_answer, result_entities = reformat_result["conclusion"]["text"], reformat_result["conclusion"]["entities"]
    
    logger.info(f"Reformated question: {reformat_question}")
    logger.info(f"Question entities: {question_entities}")
    logger.info(f"Reformated answer: {reformat_answer}")
    logger.info(f"Result entities: {result_entities}")
    
    path_all = []
    for q_entity in question_entities:
        for a_entity in result_entities:
            try:
                path_all+= list(nx.all_shortest_paths(G, q_entity.lower(), a_entity.lower()))
            except Exception as e:
                logger.info(f"Error type: {type(e).__name__}")
                if type(e).__name__ == 'NodeNotFound':
                    logger.info(q_entity.lower())
                    logger.info(a_entity.lower())
    logger.info(f"the number of extracted pathes are: {len(path_all)}")
    if len(path_all) == 0:
        logger.info("No reasoning path found, can not generate reasoning.")
        return "No reasoning path. Can not find path in KG."
    if filter_path:
        path_all = utils.path_sampling(path_all = path_all,
                                    question = question,
                                    answer = answer,
                                    topK_reasoning_paths = topK_reasoning_paths,
                                    max_path_number_per_group = max_path_number_per_group,
                                    logger = logger)
    
    output_text = '\n'.join([str(idx+1) + ':' + '->'.join(inner_list) for idx,inner_list in enumerate(path_all)])
    
    # if len of output_text is too long, return no reasoning path
    if len(output_text) > max_tokens:
        return "No reasoning path. Too many paths."
    logger.info(f"reasoning paths used: {output_text}")
    prompt = answer_prompt2.format(question = reformat_question,answer = reformat_answer , paths=output_text)
    result = utils.run_llm(prompt,temperature,max_tokens,engine)
    
    if result == "openai error, retry":
        return "No reasoning path. Unknown openai error."
    return result

def worker_init(new_dataset, new_G, new_primekg, new_emb_model, new_nodeemb_dict, dataset_name):
    global logger, dataset, G, primekg, emb_model, nodeemb_dict
    # Each worker sets up its own logger.
    logger = utils.init_logger(name=f"{dataset_name}_{os.getpid()}")
    dataset = new_dataset
    G = new_G
    primekg = new_primekg
    emb_model = new_emb_model
    nodeemb_dict = new_nodeemb_dict
    
# ----------------------------------
# Worker function to process one sample.
# ----------------------------------
def process_sample(sample_id):
    global dataset, logger, G, primekg, emb_model, nodeemb_dict
    sample = dataset[sample_id]
    question = sample['question']
    answer = sample['answer']
    comparing_reasoning = sample['comparison']
    options = sample['options']
    
    logger.info(f"Processing sample id {sample_id}.")

    reason = reasoning_generation(question, answer, primekg, emb_model, nodeemb_dict, filter_path=True)
    logger.info(f"Finished sample id {sample_id} with reasoning: {reason}")
    return {
        "id": sample_id,
        "question": question,
        "reasoning": reason,
        "answer": answer,
        "options": options
    }

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="medqa")
    parser.add_argument("--sample", type=int, default=50)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=1)
    args = parser.parse_args()
    
    object_dataset_name = args.dataset
    test_samples = args.sample
    
    current_dir = os.path.dirname(os.path.abspath(__file__))  # src/data_generation目录
    config_path = os.path.abspath(os.path.join(current_dir, '..', '..', 'configs', 'dataset_configs.yml'))

    print(f"脚本目录：{current_dir}")
    print(f"配置文件完整路径：{config_path}")

    if not os.path.exists(config_path):
        print("错误：配置文件不存在！请确认路径")
        exit(1)

    with open(config_path, 'r') as f:
        dataset_configs = yaml.safe_load(f)

    # 假设你已定义或者从参数获取了object_dataset_name
    print("配置文件内容示例:", list(dataset_configs.keys()))
    assert object_dataset_name in dataset_configs

    logger = utils.init_logger(name=object_dataset_name)
    logger.info("Start reasoning generation for dataset: " + object_dataset_name)

    dataset = QADataset(**dataset_configs[object_dataset_name])

    primekg_path = os.path.abspath(os.path.join(current_dir, 'data', 'primeKG.csv'))
    print("知识图谱路径：", primekg_path)
    if not os.path.exists(primekg_path):
        print("❌ 找不到 primeKG.csv，请确认路径！")
        exit(1)
    primekg = pd.read_csv(primekg_path, low_memory=False)

    selected_columns_list = primekg[['x_name', 'display_relation', 'y_name']].values.tolist()
    G = utils.build_graph(selected_columns_list)

    emb_model = SentenceTransformer("abhinand/MedEmbed-large-v0.1")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🔧 当前使用的设备为: {device}")
    emb_model.to(device)

    # 设定实际路径
    kg_path = os.path.abspath(os.path.join(current_dir, 'data', 'primeKG.csv'))

    # 调用嵌入生成函数（只生成一次即可）
    utils.generate_node_embeddings(
        knowledge_graph_path = kg_path,
        emb_model_name = 'abhinand/MedEmbed-large-v0.1'
    )

    nodeemb_dict = torch.load(os.path.abspath(os.path.join(current_dir, '..', 'node_embeddings.pt')))

    results_dir = os.path.abspath(os.path.join(current_dir, '..', 'results'))
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    test_samples = min(len(dataset) - args.start_idx, test_samples)
    if test_samples <= 0:
        logger.info("No samples to process.")
        exit()

    end_idx = args.start_idx + test_samples

    result_file_name = os.path.join(results_dir, f"{object_dataset_name}_{args.start_idx}_{end_idx}.jsonl")

    if args.batch_size == 1:
        with open(result_file_name, 'w') as f:
            for ids in tqdm(range(args.start_idx, args.start_idx + test_samples)):
                logger.info(f"Processing {ids}th sample.")

                sample = dataset[ids]
                question = sample['question']
                answer = sample['answer']
                comparing_reasoning = sample['comparison']
                options = sample['options']

                logger.info(f"Question: {question}")
                logger.info(f"Answer: {answer}")

                try:
                    reason = reasoning_generation(question, answer, primekg, emb_model, nodeemb_dict, filter_path=True)
                except Exception as e:
                    reason = "No reasoning path found. Error: " + str(e)

                logger.info(f"Reasoning: {reason}")
                data_list = {
                    "id": ids,
                    "question": question,
                    "reasoning": reason,
                    "huatuo": comparing_reasoning,
                    "answer": answer,
                    "options": options
                }

                f.write(json.dumps(data_list) + "\n")
    else:
        with multiprocessing.Pool(
            processes=args.batch_size,
            initializer=worker_init,
            initargs=(dataset, G, primekg, emb_model, nodeemb_dict, object_dataset_name)
        ) as pool, open(result_file_name, 'w') as f:
            results = list(tqdm(pool.imap(process_sample, range(test_samples)), total=test_samples))
            for res in results:
                f.write(json.dumps(res) + "\n")