import networkx as nx
import random
import json
import networkx as nx
import random
import json
import os
from pathlib import Path
from time import sleep
from typing import List, Optional
from vllm import LLM, SamplingParams
import os
from openai import OpenAI
os.environ["CUDA_VISIBLE_DEVICES"] = '2'
# OPENAI_API_KEY="sk-proj-xGVNYlk2VK9uW17qkHeQT3BlbkFJeZtLHVYjeLan2RQV42JE"




RESULT_PATH = "../result/long_data/llama3-ins-hf/"
DATA_PATH = "../data/"
graph_types = [
    ("shortest_path", "Shortest Path Graph"),
    # ("max_flow", "Max Flow Graph"),
    ("graph_traversal", "Graph Traversal Graph"),
    ("tree_structure", "Tree Structure Graph"),
    ("bipartite_graph", "Bipartite Graph"),
    # ("graph_coloring", "Graph Coloring"),
    # ("hamiltonian", "Hamiltonian Graph"),
    # ("tsp", "TSP Graph"),
    # ("eulerian", "Eulerian Graph")
]

def list_files_in_directory(directory):
    try:
        # 获取指定目录下的所有文件和目录名
        files_and_dirs = os.listdir(directory)
        
        # 过滤出文件名
        # files = [f for f in files_and_dirs if os.path.isfile(os.path.join(directory, f))]
        
        return files_and_dirs
    except Exception as e:
        return str(e)
    
def generate_special_prompt(graph_type,Q):
    if graph_type == "shortest_path":
        prompt = "\nQ: " + Q + "Please provide the answer directly without the reasoning process and output it in JSON format. For example: \{\"path\"\:PATH,\"distance\":DISTANCE}  \nA: "
    elif graph_type == "max_flow":
        prompt = "\nQ: " + Q + "Please provide the answer directly without the reasoning process and output it in JSON format. For example: \{\"path\"\:PATH,\"max_flow\":MAX_FLOW}  \nA: "
    elif graph_type == "graph_traversal":
        prompt = "\nQ: " + Q + "Please provide the answer directly without the reasoning process and output it in LIST format. For example: [[1,2],[2,3],[3,4]]  \nA: "     
    elif graph_type == "tree_structure":
        prompt = "\nQ: " + "Is this a tree-structured graph data?" + "Please provide the answer directly without the reasoning process .\nA: "   
    elif graph_type == "bipartite_graph":
        prompt = "\nQ: " + Q + "Please provide the answer directly without the reasoning process .\nA: " 
    elif graph_type == "hamiltonian":
        prompt = "\nQ: " + Q + "Please provide the answer directly without the reasoning process and output it in JSON format. For example: \{\"Hamiltonian\"\:YES OR NO, \"Hamiltonian_path\": PATH}  \nA: "  
    return prompt

def main():
    """
    Entry point of the program for generating text using a pretrained model.

    Args:
        ckpt_dir (str): The directory containing checkpoint files for the pretrained model.
        tokenizer_path (str): The path to the tokenizer model used for text encoding/decoding.
        temperature (float, optional): The temperature value for controlling randomness in generation.
            Defaults to 0.6.
        top_p (float, optional): The top-p sampling parameter for controlling diversity in generation.
            Defaults to 0.9.
        max_seq_len (int, optional): The maximum sequence length for input prompts. Defaults to 512.
        max_batch_size (int, optional): The maximum batch size for generating sequences. Defaults to 8.
        max_gen_len (int, optional): The maximum length of generated sequences. If None, it will be
            set to the model's max sequence length. Defaults to None.
    """
    llm = LLM(model="/userhome/nips_2024_graph/benchmark_data/model/llama3-ins",trust_remote_code=True)        
    tokenizer = llm.get_tokenizer()     
    for graph_type, graph_name in graph_types:
        print(graph_name)
        save_dir_file = RESULT_PATH + graph_name + "/"
        save_path = Path(save_dir_file)
        data_path = Path(DATA_PATH + graph_name + "/")
        if not save_path.exists():
            save_path.mkdir(parents=True)
        file_list = list_files_in_directory(data_path)
        for dataset_name in file_list:
            file_path = str(data_path) + "/" + dataset_name
            with open(file_path,'r') as fr: #同上
                json_file = json.loads(fr.read())
            simple_save_path = str(save_path) +"/" +dataset_name
            simple_save_path = Path(simple_save_path)
            if simple_save_path.exists():
                print("{data_file} is exists!".format(data_file = dataset_name ))
                continue
            result_list = []
            for index,graph in enumerate(json_file):
                result_dic = {
                    "dataset_name": dataset_name,
                    "graph_index": index,
                    "graph_name": graph_name
                }
                graph_description = graph["graph_description"]
                random_graph_description = graph["random_graph_description"]
                structure_qa = graph["QA"]["structure_qa"][0]
                node_number_q = structure_qa["node_number"]["Q"]
                average_degree_q = structure_qa["average_degree"]["Q"]
                is_connected_q = structure_qa["is_connected"]["Q"]
                similarity_q = structure_qa["similarity"]["Q"]
                specail_qa = graph["QA"]["specail_qa"]
                specail_qa_q = specail_qa["Q"]

                dialogs = [
                    [
                        # {"role": "system", "content": "You are an intelligent assistant specializing in graph theory and graph-structured data analysis. Your task is to understand and process complex graph-structured data, including but not limited to solving traversal problems (such as breadth-first search and depth-first search), computing shortest paths, addressing NP-complete problems (such as the Traveling Salesman Problem, Minimum Coloring Problem, and Hamiltonian Path Problem), and solving maximum flow problems. You will be provided with graph data either in textual descriptions or graphical representations. Provide direct and accurate answers with detailed solutions and steps based on the specific task requirements. Ensure your answers adhere to the fundamental principles of graph theory and are detailed and precise."},
                        {"role": "user", "content": graph_description + "\nQ: " + node_number_q + " Please provide the answer directly without the reasoning process. \nA: "},
                    ],
                    [
                        {"role": "user", "content": graph_description + "\nQ: " + average_degree_q + " Round the result to one decimal place. Please provide the answer directly without the reasoning process. \nA: "},
                    ],
                    [
                        {"role": "user", "content": graph_description + "\nQ: " + is_connected_q + " Please provide the answer directly without the reasoning process. \nA: "},
                    ],
                    [
                        {"role": "user", "content": graph_description + "\nQ: " + "Please output the triplet of this graph in JSON format.For example: \{\"triple\"\:[(node_A,node_B,weight)] \nA: "},
                    ],
                    [
                        {"role": "user", "content": graph_description + generate_special_prompt(graph_type,specail_qa_q)},
                    ],
                ]
                promt_list = []
                ans_list = []
                for q in dialogs:
                    promt_list.append(q[0]["content"])
                    conversations = tokenizer.apply_chat_template(
                       q,
                        tokenize=False,
                    )
                    output = llm.generate(
                        [conversations],
                        SamplingParams(
                            temperature=0.5,
                            top_p=0.9,
                            max_tokens=2048,
                            stop_token_ids=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")],  # KEYPOINT HERE
                        )
                    )
                    ans_list.append(output)
                print(ans_list)
                # for input in dialogs:
                #     while 1:
                #         try:
                #             completion = client.chat.completions.create(
                #                 model="gpt-3.5-turbo",
                #                 messages=input
                #             )
                #             break
                #         except Exception as e:
                #             print(e)
                #             print("Connection error!Waite 5s!")
                #             sleep(5)
                #             continue
                #     output = completion.choices[0].message.content
                #     ans_list.append(output)
                Answer = {
                    "node_number":{
                        "Q":promt_list[0],
                        "human": structure_qa["node_number"]["A"],
                        "GPT":ans_list[0][0].outputs[0].text
                    },
                    "average_degree":{
                        "Q":promt_list[1],
                        "human": structure_qa["average_degree"]["A"],
                        "GPT": ans_list[1][0].outputs[0].text
                    },
                    "is_connected":{
                        "Q":promt_list[2],
                        "human": structure_qa["is_connected"]["A"],
                        "GPT": ans_list[2][0].outputs[0].text
                    },
                    "similarity":{
                        "Q":promt_list[3],
                        "human": structure_qa["similarity"]["A"],
                        "GPT": ans_list[3][0].outputs[0].text
                    },
                    "specail_qa":{
                        "Q":promt_list[4],
                        "human": specail_qa["A"],
                        "GPT": ans_list[4][0].outputs[0].text
                    },
                }
                result_dic["Answer"] = Answer
                result_list.append(result_dic)
                print(result_dic)
                # for dialog, result in zip(dialogs, results):
                #     for msg in dialog:

                    #     print(f"{msg['role'].capitalize()}: {msg['content']}\n")
                    # print(
                    #     f"> {result['generation']['role'].capitalize()}: {result['generation']['content']}"
                    # )
            with open(str(save_path) +"/" +dataset_name, 'w',encoding='utf-8') as file:
                json.dump(result_list,file,ensure_ascii=False,indent=1)     
               
if __name__ == "__main__":
    main()


