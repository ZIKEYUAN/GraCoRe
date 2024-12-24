# GraCoRe(COLING 2025)
GraCoRe: Benchmarking Graph Comprehension and Complex Reasoning in Large Language Models

Paper Link: [https://arxiv.org/abs/2407.02936](https://arxiv.org/abs/2407.02936)

Full Data Link: [https://huggingface.co/datasets/fly111222/GraCoRe/tree/main](https://huggingface.co/datasets/fly111222/GraCoRe/tree/main)

**Abstract**:

Evaluating the graph comprehension and reasoning abilities of Large Language Models(LLMs) is challenging and often incomplete.Existing benchmarks focus primarily on pure graph understanding, lacking a comprehensive evaluation across all graph types and detailed capability definitions. This paper presents GraCoRe, a benchmark for systematically assessing LLMs’ graph comprehension and reasoning. GraCoRe uses a three-tier hierarchical taxonomy to categorize and test models on pure graph and heterogeneous graphs, subdividing capabilities into 10 distinct areas tested through
19 tasks. Our benchmark includes 11 datasets with 5,140 graphs of varying complexity. We evaluated 4 closed-source and 8 open source LLMs, conducting thorough analyses from both ability and task perspectives. Key findings reveal that semantic enrichment enhances reasoning performance, node ordering impacts task success, and the ability to process longer texts does not necessarily improve graph comprehension or reasoning.

### Citation
If you find this repo useful, please cite our paper:
```
@article{yuan2024gracore,
  title={Gracore: Benchmarking graph comprehension and complex reasoning in large language models},
  author={Yuan, Zike and Liu, Ming and Wang, Hui and Qin, Bing},
  journal={arXiv preprint arXiv:2407.02936},
  year={2024}
}
```