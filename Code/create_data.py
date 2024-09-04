import networkx as nx
import random
import json
import networkx as nx
import random
import json
from pathlib import Path
import matplotlib.pyplot as plt
import itertools
SAVE_DIR = "./small_data/"

def tsp_nearest_neighbor(G, start_node=0):
    nodes = list(G.nodes)
    path = [start_node]
    visited = set(path)
    
    current_node = start_node
    while len(visited) < len(nodes):
        next_node = min((node for node in nodes if node not in visited), 
                        key=lambda node: G[current_node][node]['weight'])
        path.append(next_node)
        visited.add(next_node)
        current_node = next_node
    
    path.append(start_node)  # 回到起点
    length = sum(G[path[i]][path[i + 1]]['weight'] for i in range(len(path) - 1))
    return path, length

class GraphDatasetGenerator:
    def __init__(self, num_graphs, num_nodes, min_path_length=3,edge_ratio_range=(0.05,0.10)):
        self.num_graphs = num_graphs
        self.num_nodes = num_nodes
        self.min_path_length = min_path_length
        self.edge_ratio_range = edge_ratio_range
        self.edge_radio = random.randint(edge_ratio_range[0]*100,edge_ratio_range[1]*100)/100

    def generate(self, graph_type):
        if graph_type == "shortest_path":
            return self._generate_shortest_path_dataset()
        elif graph_type == "graph_traversal":
            return self._generate_graph_traversal_dataset()
        elif graph_type == "tree_structure":
            return self._generate_tree_structure_dataset()
        elif graph_type == "bipartite_graph":
            return self._generate_bipartite_graph_dataset()
        elif graph_type == "max_flow":
            return self._generate_max_flow_dataset()
        elif graph_type == "graph_coloring":
            return self._generate_graph_coloring_dataset()
        elif graph_type == "hamiltonian":
            return self._generate_hamiltonian_dataset()
        elif graph_type == "tsp":
            return self._generate_tsp_dataset()
        elif graph_type == "eulerian":
            return self._generate_eulerian_dataset()
        else:
            raise ValueError(f"Unsupported graph type: {graph_type}")
        
    def _add_weights(self, G):
        for (u, v) in G.edges():
            G[u][v]['weight'] = random.randint(1, 5)
        return G
    
    def _generate_strongly_connected_digraph(self, max_edges_factor=1.5):
        while True:
            G = nx.gnm_random_graph(self.num_nodes, self.num_nodes * 2, directed=True)
            if nx.is_strongly_connected(G):
                if G.number_of_edges() > self.num_nodes * max_edges_factor:
                    edges_to_remove = G.number_of_edges() - int(self.num_nodes * max_edges_factor)
                    edges = list(G.edges())
                    random.shuffle(edges)
                    G.remove_edges_from(edges[:edges_to_remove])
                return self._add_weights(G)
            
    def _relabel_nodes(self, G):
        nodes = list(G.nodes())
        shuffled_nodes = nodes[:]
        random.shuffle(shuffled_nodes)
        mapping = {nodes[i]: shuffled_nodes[i] for i in range(len(nodes))}
        return nx.relabel_nodes(G, mapping), mapping    

    def _generate_shortest_path_dataset(self):
        dataset = []
        for _ in range(self.num_graphs):
            G = nx.gnm_random_graph(self.num_nodes, self.num_nodes+self.num_nodes*self.num_nodes*self.edge_radio)
            G = self._add_weights(G)
            
            valid_path = False
            while not valid_path:
                source = random.choice(list(G.nodes()))
                target = random.choice(list(G.nodes()))
                if source != target and nx.has_path(G, source, target):
                    path = nx.shortest_path(G, source, target, weight='weight')
                    if len(path) > self.min_path_length:
                        valid_path = True

            dataset.append((G, source, target))
        return dataset

    def _generate_graph_traversal_dataset(self):
        dataset = []
        for _ in range(self.num_graphs):
            G = nx.gnm_random_graph(self.num_nodes, self.num_nodes + self.num_nodes*self.num_nodes*self.edge_radio)
            G = self._add_weights(G)
            start_node = random.choice(list(G.nodes()))
            dataset.append((G, start_node,_))
        return dataset

    def _generate_tree_structure_dataset(self):
        dataset = []
        for _ in range(self.num_graphs):
            G = nx.random_tree(self.num_nodes)
            if random.random() > 0.5:
                source_node , target_node = random.sample(list(G.nodes),2)
                G.add_edge(source_node,target_node)
            G = self._add_weights(G)
            dataset.append(G)
        return dataset

    def _generate_bipartite_graph_dataset(self):
        dataset = []
        for _ in range(self.num_graphs):
            G = nx.bipartite.random_graph(self.num_nodes//2, self.num_nodes//2, 0.5)
            if random.random() > 0.6:
                source_node , target_node = random.sample(list(G.nodes),2)
                G.add_edge(source_node,target_node)
            G = self._add_weights(G)
            dataset.append(G)
        return dataset

    def _generate_max_flow_dataset(self):
        node_count = self.num_nodes
        dataset = []
        for _ in range(self.num_graphs):
            G = nx.DiGraph()
            
            # Add nodes
            for i in range(node_count):
                G.add_node(i)
            
            # Define source and sink
            source = 0
            sink = node_count - 1
            
            # Add edges with random capacities
            for i in range(node_count):
                for j in range(i+1, node_count):
                    if random.random() > 0.5:  # Randomly decide to create an edge or not
                        capacity = random.randint(1, 5)
                        G.add_edge(i, j, weight=capacity)
            
            # Ensure source only flows out and sink only flows in
            for node in G.nodes():
                if node != source:
                    if random.random() > 0.5:
                        capacity = random.randint(1, 5)
                        G.add_edge(source, node, weight=capacity)
                if node != sink:
                    if random.random() > 0.5:
                        capacity = random.randint(1, 5)
                        G.add_edge(node, sink, weight=capacity)
        
            dataset.append((G, source, sink))
        return dataset

    def _generate_graph_coloring_dataset(self):
        dataset = []
        for _ in range(self.num_graphs):
            G = nx.gnm_random_graph(self.num_nodes, self.num_nodes+self.num_nodes*self.num_nodes*self.edge_radio)
            G = self._add_weights(G)
            dataset.append(G)
        return dataset

    def _generate_hamiltonian_dataset(self):
        dataset = []
        hamiltonian_cycles = []
        for _ in range(self.num_graphs):
            if self.num_nodes < 2:
                raise ValueError("Number of nodes must be at least 2.")
            
            # Step 1: Create a directed cycle graph
            G = nx.DiGraph()
            nodes = list(range(self.num_nodes))
            G.add_nodes_from(nodes)
            edges = [(nodes[i], nodes[(i + 1) % self.num_nodes]) for i in range(self.num_nodes)]
            G.add_edges_from(edges)
            original_cycle = edges

            # Step 2: Optionally add some extra random edges
            max_possible_edges = self.num_nodes * (self.num_nodes - 1)
            edges_to_add = self.num_nodes + self.num_nodes*self.num_nodes*self.edge_radio
            added_edges = 0
            
            while added_edges < edges_to_add:
                u = random.randint(0, self.num_nodes - 1)
                v = random.randint(0, self.num_nodes - 1)
                if u != v and not G.has_edge(u, v):
                    G.add_edge(u, v)
                    added_edges += 1
            
            G = self._add_weights(G)
            G, mapping = self._relabel_nodes(G)

            # Remap the original Hamiltonian cycle to the new node labels
            remapped_cycle = [(mapping[u], mapping[v]) for u, v in original_cycle]
            hamiltonian_cycles.append(remapped_cycle)
            
            dataset.append(G)
        
        return dataset, hamiltonian_cycles

    def _generate_tsp_dataset(self):
        dataset = []
        for _ in range(self.num_graphs):
            G = nx.complete_graph(self.num_nodes)
            G = self._add_weights(G)
            dataset.append(G)
        return dataset

    def _generate_eulerian_dataset(self):
        dataset = []
        for _ in range(self.num_graphs):
            if self.num_nodes % 2 != 0:
                self.num_nodes += 1
            
            G = nx.Graph()
            G.add_nodes_from(range(self.num_nodes))
            
            # Add edges to make the graph Eulerian
            for i in range(self.num_nodes):
                for j in range(i + 1, self.num_nodes):
                    if random.choice([True, False]):
                        G.add_edge(i, j)
            
            # Ensure all degrees are even
            for node in G.nodes():
                if G.degree(node) % 2 != 0:
                    while True:
                        target = random.choice(list(G.nodes()))
                        if target != node and G.degree(target) % 2 != 0:
                            G.add_edge(node, target)
                            break
            
            # Find Eulerian circuit
            G = self._add_weights(G)
            # eulerian_circuit = list(nx.eulerian_circuit(G))
            dataset.append(G)
        return dataset
    
class GraphDataProcessor:
    def __init__(self, dataset, hamiltonian_cycles, graph_name):
        self.dataset = dataset
        self.hamiltonian_cycles = hamiltonian_cycles
        self.graph_name = graph_name

    def graph_to_text(self):
        text_data = []
        for i, graph_info in enumerate(self.dataset):
            if isinstance(graph_info, nx.Graph):
                G = graph_info
            elif isinstance(graph_info,tuple):
                G, _, _ = graph_info
            else:
                raise ValueError("Dataset must contain networkx graph objects")

            edges = list(G.edges(data=True))
            is_tree = nx.is_tree(G) if not nx.is_directed(G) else False
            is_connected = nx.is_connected(G.to_undirected()) if nx.is_directed(G) else nx.is_connected(G)
            components = [list(component) for component in nx.connected_components(G.to_undirected())] if nx.is_directed(G) else [list(component) for component in nx.connected_components(G)]
            avg_degree = sum(dict(G.degree()).values()) / G.number_of_nodes()
            diameter = nx.diameter(G.to_undirected()) if is_connected else 'undefined'
            radius = nx.radius(G.to_undirected()) if is_connected else 'undefined'
            adjacency_matrix = nx.adjacency_matrix(G).todense().tolist()

            graph_type = "directed" if nx.is_directed(G) else "undirected"
            edge_descriptions = [f"From node {u} to node {v}, distance is {data['weight']}" for u, v, data in edges]
            graph_description = f"This is a {graph_type} graph with the following edges:\n" + "\n".join(edge_descriptions)
            random.shuffle(edge_descriptions)
            random_graph_description = f"This is a {graph_type} graph with the following edges:\n" + "\n".join(edge_descriptions)

            if self.graph_name == "Shortest Path Graph":
                qa_pairs = self.process_shortest_path_dataset(i)
            elif self.graph_name == "Max Flow Graph":
                qa_pairs = self.process_max_flow_dataset(i)
            elif self.graph_name == "Graph Traversal Graph":
                qa_pairs = self.process_graph_traversal_dataset(i)
            elif self.graph_name == "Tree Structure Graph":
                qa_pairs = self.process_tree_structure_dataset(i)
            elif self.graph_name == "Bipartite Graph":
                qa_pairs = self.process_bipartite_graph_dataset(i)
            elif self.graph_name == "Graph Coloring":
                qa_pairs = self.process_graph_coloring_dataset(i)
            elif self.graph_name == "Hamiltonian Graph":
                qa_pairs = self.process_hamiltonian_dataset(i)
            elif self.graph_name == "TSP Graph":
                qa_pairs = self.process_tsp_dataset(i)
            elif self.graph_name == "Eulerian Graph":
                qa_pairs = self.process_eulerian_dataset(i)

            structure_qa = {
                "node_number":{
                    "Q": "How many nodes in this graph?",
                    "A": G.number_of_nodes()
                },
                "average_degree":{
                    "Q": "What's the average degree of this graph?",
                    "A": avg_degree
                },
                "is_connected":{
                    "Q": "Is this graph a connected graph?",
                    "A": "Yes" if is_connected else "No"
                },
                "similarity":{
                    "Q": "What are the triplets in this graph? ",
                    "A": [(u,  v,  data["weight"]) for u, v, data in edges]
                }
            },
            specail_qa = {"Q":qa_pairs[0],
                          "A":qa_pairs[1]}
            
            text_graph = {
                "nodes": list(G.nodes),
                "edges": [{"source": u, "target": v, "attributes": data['weight']} for u, v, data in edges],
                "params": [],
                "node_count": G.number_of_nodes(),
                "edge_count": G.number_of_edges(),
                "is_directed": nx.is_directed(G),
                "is_tree": is_tree,
                "is_connected": is_connected,
                "components": components,
                "graph_name": self.graph_name,
                "average_degree": avg_degree,
                "diameter": diameter,
                "radius": radius,
                "adjacency_matrix": adjacency_matrix,
                "graph_description": graph_description,
                "random_graph_description": random_graph_description,
                "hamiltonian_cycle": self.hamiltonian_cycles[i] if self.hamiltonian_cycles else None,
                "euler_cycle": nx.is_eulerian(G),
                "QA":{
                    "structure_qa":structure_qa,
                    "specail_qa":specail_qa
                }
            }
            text_data.append(text_graph)
        return text_data


    def process_shortest_path_dataset(self,i):
        G, source, target = self.dataset[i]
        try:
            length, path = nx.single_source_dijkstra(G, source, target)
            question = f"What is the shortest path from node {source} to node {target}? How long is it?"
            answer = f"The shortest path is {path} with length {length}."
        except nx.NetworkXNoPath:
            question = f"What is the shortest path from node {source} to node {target}? How long is it?"
            answer = "There is no path between the given nodes."
        return (question, answer)

    def process_max_flow_dataset(self,i):
        G, source, target = self.dataset[i]
        # pos = nx.spring_layout(G)
        # edge_labels = {(u, v): f"{G[u][v]['weight']}" for u, v in G.edges()}
        # nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=700, font_size=10, font_weight='bold')
        # nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
        # plt.title("Random Directed Graph with Capacities")
        # plt.savefig("1.png")
        flow_value, flow_dict = nx.maximum_flow(G, source, target, capacity='weight')
        flow_paths = [(u, v, flow_dict[u][v]) for u in flow_dict for v in flow_dict[u] if flow_dict[u][v] > 0]
        question = f"What is the maximum flow from node {source} to node {target}?"
        answer = f"The maximum flow is {flow_value}. The flow paths are {flow_paths}."
        return (question, answer)

    def process_graph_traversal_dataset(self,i):
        G, start_node,_ = self.dataset[i]
        start_node = 0
        bfs_edge = list(nx.bfs_edges(G, start_node))
        bfs_sequence = [start_node] + [v for u, v in bfs_edge]
        sequence =  [v for u, v in bfs_edge]
        random.shuffle(sequence)
        B_sequence = [start_node] + sequence
        random.shuffle(sequence)
        C_sequence = [start_node] + sequence
        random.shuffle(sequence)
        D_sequence = [start_node] + sequence
        question = f"Implement the Breadth-First Search (BFS) algorithm to traverse the graph and return the list of nodes traversed in the order they are visited.Visited the graph start node 0.You can choise A,B,C or D.\n A. {bfs_sequence} \n B. {B_sequence} \n C. {C_sequence} D. {D_sequence} \n"
        answer = f"A."            
        return (question, answer)

    def process_tree_structure_dataset(self,i):
        G = self.dataset[i]
        is_tree = nx.is_tree(G)
        question = "A tree is an undirected graph in which any two vertices are connected by exactly one path, and there are no cycles. Based on the following description, determine if the given graph is a tree."
        answer = "Yes, it is a tree." if is_tree else "No, it is not a tree."
        return (question, answer)

    def process_bipartite_graph_dataset(self,i):
        G = self.dataset[i]
        is_bipartite = nx.is_bipartite(G)
        question = " A bipartite graph is a special type of graph where the vertex set can be divided into two disjoint subsets such that every edge connects a vertex in one subset to a vertex in the other subset. Please determine if the given graph is bipartite."
        answer = "Yes, it is bipartite." if is_bipartite else "No, it is not bipartite."
        return (question, answer)

    def process_graph_coloring_dataset(self,i):
        G = self.dataset[i]
        coloring = nx.coloring.greedy_color(G,strategy="largest_first")
        num_colors = len(set(coloring.values()))
        question = "Use a greedy algorithm with the \"largest_first\" strategy to determine the minimum number of colors needed to color the graph, ensuring that no two adjacent nodes share the same color."
        answer = f"Minimum number of colors required: {num_colors}"
        return (question, answer)

    def process_hamiltonian_dataset(self,i):
        G = self.dataset[i]
        # pos = nx.spring_layout(G)
        # edge_labels = {(u, v): f"{G[u][v]['weight']}" for u, v in G.edges()}
        # nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=700, font_size=10, font_weight='bold')
        # nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
        # plt.title("Random Directed Graph with Capacities")
        # plt.savefig("hamilton.png")
        try:
            cycle = self.hamiltonian_cycles[i]
            question = "A Hamiltonian cycle in this graph is a cycle that visits each node exactly once and returns to the starting node. If a Hamiltonian cycle exists in this graph, provide the sequence of nodes that form this cycle. "
            answer = list(G.edges(data=True))
        except nx.NetworkXNoCycle:
            question = "A Hamiltonian cycle in this graph is a cycle that visits each node exactly once and returns to the starting node. If a Hamiltonian cycle exists in this graph, provide the sequence of nodes that form this cycle. "
            answer = "No, it does not have a Hamiltonian cycle."
        return (question, answer)

    def process_tsp_dataset(self,i):
        G = self.dataset[i]
        # 由于 TSP 是 NP 完全问题，这里使用近似算法解决
        path, length = tsp_nearest_neighbor(G)
        question = "The goal is to find the shortest possible route that visits each node exactly once and returns to the starting node.Please determine the optimal solution for this Traveling Salesman Problem (TSP).You can use Nearest Neighbor Algorithm solve this problem. Provide the sequence of nodes that form this shortest route and the total distance of this route.Start from node 0."
        answer = f"The TSP path is {length} and path is {path}."
        return (question, answer)

    def process_eulerian_dataset(self,i):
        G = self.dataset[i]
        # pos = nx.spring_layout(G)
        # edge_labels = {(u, v): f"{G[u][v]['weight']}" for u, v in G.edges()}
        # nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=700, font_size=10, font_weight='bold')
        # nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
        # plt.title("Random Directed Graph with Capacities")
        # plt.savefig("euler.png")
        has_eulerian_circuit = nx.is_eulerian(G)
        question = "An Eulerian circuit in this graph is a cycle that visits every edge exactly once and returns to the starting node.Please determine if an Eulerian circuit exists in this graph. "
        answer = "Yes, it has an Eulerian circuit." if has_eulerian_circuit else "No, it does not have an Eulerian circuit."

        return (question, answer)

    # 可以根据不同的图类型增加其他处理方法

def main(graph_types, num_graphs, num_nodes):
    for i in num_nodes:
        generator = GraphDatasetGenerator(num_graphs, i)
        processors = []
        for graph_type, graph_name in graph_types:
            print(graph_name)
            
            save_dir_file = SAVE_DIR + graph_name + "/"
            path = Path(save_dir_file)
            if not path.exists():
                path.mkdir(parents=True)
            save_name = save_dir_file + "graph_{num_nodes}.json".format(num_nodes = i)
            if graph_type == "hamiltonian":
                dataset, hamiltonian_cycles = generator.generate(graph_type)
                processor = GraphDataProcessor(dataset, hamiltonian_cycles, graph_name)
            else:
                dataset = generator.generate(graph_type)
                processor = GraphDataProcessor(dataset, [], graph_name)
            text_data = processor.graph_to_text()
            with open(save_name, 'w',encoding='utf-8') as file:
                json.dump(text_data,file,ensure_ascii=False,indent=1)
    print("generete end!")
    # 将图数据集转化为文本描述并打印
    # for processor in processors:
    #     text_data = processor.graph_to_text()
    #     for text in text_data:

    #         print(text["graph_name"])
    #         print(text["graph_description"])
    #         print(text["QA"])
    #         print("=====")

    # 处理数据集，生成问题和答案并打印
    # for processor in processors:
    #     if processor.graph_name == "Shortest Path Graph":
    #         qa_pairs = processor.process_shortest_path_dataset()
    #     elif processor.graph_name == "Max Flow Graph":
    #         qa_pairs = processor.process_max_flow_dataset()
    #     elif processor.graph_name == "Graph Traversal Graph":
    #         qa_pairs = processor.process_graph_traversal_dataset()
    #     elif processor.graph_name == "Tree Structure Graph":
    #         qa_pairs = processor.process_tree_structure_dataset()
    #     elif processor.graph_name == "Bipartite Graph":
    #         qa_pairs = processor.process_bipartite_graph_dataset()
    #     elif processor.graph_name == "Graph Coloring":
    #         qa_pairs = processor.process_graph_coloring_dataset()
    #     elif processor.graph_name == "Hamiltonian Graph":
    #         qa_pairs = processor.process_hamiltonian_dataset()
    #     elif processor.graph_name == "TSP Graph":
    #         qa_pairs = processor.process_tsp_dataset()
    #     elif processor.graph_name == "Eulerian Graph":
    #         qa_pairs = processor.process_eulerian_dataset()

        # for question, answer in qa_pairs:
        #     print(question)
        #     print(answer)
        #     print()

# 示例使用
graph_types = [
    ("shortest_path", "Shortest Path Graph"),
    ("max_flow", "Max Flow Graph"),
    ("graph_traversal", "Graph Traversal Graph"),
    ("tree_structure", "Tree Structure Graph"),
    ("bipartite_graph", "Bipartite Graph"),
    ("graph_coloring", "Graph Coloring"),
     ("hamiltonian", "Hamiltonian Graph"),
    ("tsp", "TSP Graph"),
    ("eulerian", "Eulerian Graph")
]

num_graphs = 20

num_nodes = [i for i in range(30,31)]


main(graph_types, num_graphs, num_nodes)