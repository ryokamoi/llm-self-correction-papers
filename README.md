# LLM Self-Correction Papers

This repository contains a list of papers on self-correction of large language models (LLMs).

The list is maintained by [Ryo Kamoi](https://ryokamoi.github.io/). If you have any suggestions or corrections, please feel free to open an issue or a pull request (refer to [Contributing](#contributing) for details).

This list is based on [our survey paper](https://arxiv.org/abs/2406.01297). If you find this list useful, please consider citing our paper:

```bibtex
@article{kamoi2024self-correction,
    author = {Kamoi, Ryo and Zhang, Yusen and Zhang, Nan and Han, Jiawei and Zhang, Rui},
    title = "{When Can LLMs Actually Correct Their Own Mistakes? A Critical Survey of Self-Correction of LLMs}",
    journal = {Transactions of the Association for Computational Linguistics},
    volume = {12},
    pages = {1417-1440},
    year = {2024},
    month = {11},
    issn = {2307-387X},
    doi = {10.1162/tacl_a_00713},
    url = {https://doi.org/10.1162/tacl\_a\_00713},
    eprint = {https://direct.mit.edu/tacl/article-pdf/doi/10.1162/tacl\_a\_00713/2478635/tacl\_a\_00713.pdf},
}
```

## Self-Correction of LLMs

Self-correction of LLMs is a framework that refines responses from LLMs using LLMs during inference. Previous work has proposed various frameworks for self-correction, such as using external tools or information, or training LLMs specifically for self-correction.

<p align="center">
  <img src="figures/categories.png" width="500">
</p>

In this repository, we focus on inference-time self-correction, and differentiate it from training-time self-improvement of LLMs, which uses their own responses for improving themselves only during training.

We also do not cover generate-and-rank (or sample-and-rank), which generate multiple responses and rank them using LLMs or other models. In contrast, self-correction refines their own responses, not only selecting the best one from multiple responses.

## Table of Contents

- [Survey Papers of Self-Correction](#survey-papers-of-self-correction)
- [Intrinsic Self-Correction](#intrinsic-self-correction)
    - [Negative Results of Intrinsic Self-Correction](#negative-results-of-intrinsic-self-correction)
- [Self-Correction with External Tools](#self-correction-with-external-tools)
    - [with In-Context Learning (External Tools)](#with-in-context-learning-external-tools)
    - [with Trianing (External Tools)](#with-trianing-external-tools)
- [Self-Correction with Information Retrieval](#self-correction-with-information-retrieval)
    - [with In-Context Learning (Information Retrieval)](#with-in-context-learning-information-retrieval)
    - [with Trianing (Information Retrieval)](#with-trianing-information-retrieval)
- [Self-Correction with Training Designed for Self-Correction](#self-correction-with-training-designed-for-self-correction)
    - [Supervised Fine-tuning](#supervised-fine-tuning)
    - [Reinforcement Learning](#reinforcement-learning)
    - [o1-like Frameworks](#o1-like-frameworks)
- [Contributing](#contributing)
- [License](#license)

## Survey Papers of Self-Correction

* **Automatically Correcting Large Language Models: Surveying the Landscape of Diverse Automated Correction Strategies.** *Liangming Pan, Michael Saxon, Wenda Xu, Deepak Nathani, Xinyi Wang, and William Yang Wang.* TACL. 2024. [[paper](https://doi.org/10.1162/tacl_a_00660)] [[paper list](https://github.com/teacherpeterpan/self-correction-llm-papers)]
* **When Can LLMs Actually Correct Their Own Mistakes? A Critical Survey of Self-Correction of LLMs.** *Ryo Kamoi, Yusen Zhang, Nan Zhang, Jiawei Han, and Rui Zhang.* TACL. 2024. [[paper](https://doi.org/10.1162/tacl_a_00713)]

## Intrinsic Self-Correction

Intrinsic self-correction is a framework that refines responses from LLMs using the same LLMs without using external feedback or training designed for self-correction.

* **Constitutional AI: Harmlessness from AI Feedback.** *Yuntao Bai, Saurav Kadavath, Sandipan Kundu, Amanda Askell, Jackson Kernion, Andy Jones, Anna Chen, Anna Goldie, Azalia Mirhoseini, Cameron McKinnon, Carol Chen, Catherine Olsson, Christopher Olah, Danny Hernandez, Dawn Drain, Deep Ganguli, Dustin Li, Eli Tran-Johnson, Ethan Perez, Jamie Kerr, Jared Mueller, Jeffrey Ladish, Joshua Landau, Kamal Ndousse, Kamile Lukosuite, Liane Lovitt, Michael Sellitto, Nelson Elhage, Nicholas Schiefer, Noemi Mercado, Nova DasSarma, Robert Lasenby, Robin Larson, Sam Ringer, Scott Johnston, Shauna Kravec, Sheer El Showk, Stanislav Fort, Tamera Lanham, Timothy Telleen-Lawton, Tom Conerly, Tom Henighan, Tristan Hume, Samuel R. Bowman, Zac Hatfield-Dodds, Ben Mann, Dario Amodei, Nicholas Joseph, Sam McCandlish, Tom Brown, and Jared Kaplan.* Preprint. 2022. [[paper](https://arxiv.org/abs/2212.08073)]
* [Self-Refine] **Self-Refine: Iterative Refinement with Self-Feedback.** *Aman Madaan, Niket Tandon, Prakhar Gupta, Skyler Hallinan, Luyu Gao, Sarah Wiegreffe, Uri Alon, Nouha Dziri, Shrimai Prabhumoye, Yiming Yang, Shashank Gupta, Bodhisattwa Prasad Majumder, Katherine Hermann, Sean Welleck, Amir Yazdanbakhsh, and Peter Clark.* NeurIPS. 2023. [[paper](https://proceedings.neurips.cc/paper_files/paper/2023/file/91edff07232fb1b55a505a9e9f6c0ff3-Paper-Conference.pdf)]
* [CoVe] **Chain-of-Verification Reduces Hallucination in Large Language Models.** *Shehzaad Dhuliawala, Mojtaba Komeili, Jing Xu, Roberta Raileanu, Xian Li, Asli Celikyilmaz, and Jason Weston.* Preprint. 2023. [[paper](https://arxiv.org/abs/2309.11495)]
* [RCI] **Language Models can Solve Computer Tasks.** *Geunwoo Kim, Pierre Baldi, and Stephen McAleer.* Preprint. 2023. [[paper](https://arxiv.org/abs/2303.17491)]
* [Reflexion] **Reflexion: language agents with verbal reinforcement learning.** *Noah Shinn, Federico Cassano, Ashwin Gopinath, Karthik Narasimhan, and Shunyu Yao.* NeurIPS. 2023. [[paper](https://proceedings.neurips.cc/paper_files/paper/2023/file/1b44b878bb782e6954cd888628510e90-Paper-Conference.pdf)]


### Negative Results of Intrinsic Self-Correction

It has been reported that intrinsic self-correction does not work well in many tasks.

* **Large Language Models Cannot Self-Correct Reasoning Yet.** *Jie Huang, Xinyun Chen, Swaroop Mishra, Huaixiu Steven Zheng, Adams Wei Yu, Xinying Song, and Denny Zhou.* ICLR. 2024. [[paper](https://openreview.net/forum?id=IkmD3fKBPQ)]
* [CRITIC] **CRITIC: Large Language Models Can Self-Correct with Tool-Interactive Critiquing.** *Zhibin Gou, Zhihong Shao, Yeyun Gong, Yelong Shen, Yujiu Yang, Nan Duan, and Weizhu Chen.* ICLR. 2024. [[paper](https://openreview.net/forum?id=Sx038qxjek)]
* **When is Tree Search Useful for LLM Planning? It Depends on the Discriminator.** *Ziru Chen, Michael White, Raymond Mooney, Ali Payani, Yu Su, and Huan Sun.* ACL. 2024. [[paper](https://arxiv.org/abs/2402.10890)] [[code](https://github.com/OSU-NLP-Group/llm-planning-eval)]

## Self-Correction with External Tools

Previous work has proposed self-correction frameworks that use external tools, such as code executors for code generation tasks and proof assistants for theorem proving tasks.

### with In-Context Learning (External Tools)

* [Reflexion] **Reflexion: language agents with verbal reinforcement learning.** *Noah Shinn, Federico Cassano, Ashwin Gopinath, Karthik Narasimhan, and Shunyu Yao.* NeurIPS. 2023. [[paper](https://proceedings.neurips.cc/paper_files/paper/2023/file/1b44b878bb782e6954cd888628510e90-Paper-Conference.pdf)]
* [SelfEvolve] **SelfEvolve: A Code Evolution Framework via Large Language Models.** *Shuyang Jiang, Yuhao Wang, and Yu Wang.* Preprint. 2023. [[paper](https://arxiv.org/abs/2306.02907)]
* [Logic-LM] **Logic-LM: Empowering Large Language Models with Symbolic Solvers for Faithful Logical Reasoning.** *Liangming Pan, Alon Albalak, Xinyi Wang, and William Wang.* Findings of EMNLP. 2023. [[paper](https://aclanthology.org/2023.findings-emnlp.248)]
* [Self-Debug] **Teaching Large Language Models to Self-Debug.** *Xinyun Chen, Maxwell Lin, Nathanael Schärli, and Denny Zhou.* ICLR. 2024. [[paper](https://openreview.net/forum?id=KuPixIqPiq)]
* [CRITIC] **CRITIC: Large Language Models Can Self-Correct with Tool-Interactive Critiquing.** *Zhibin Gou, Zhihong Shao, Yeyun Gong, Yelong Shen, Yujiu Yang, Nan Duan, and Weizhu Chen.* ICLR. 2024. [[paper](https://openreview.net/forum?id=Sx038qxjek)]

### with Trianing (External Tools)

* [CodeRL] **CodeRL: Mastering Code Generation through Pretrained Models and Deep Reinforcement Learning.** *Hung Le, Yue Wang, Akhilesh Deepak Gotmare, Silvio Savarese, and Steven Chu Hong Hoi.* NeurIPS. 2022. [[paper](https://proceedings.neurips.cc/paper_files/paper/2022/file/8636419dea1aa9fbd25fc4248e702da4-Paper-Conference.pdf)]
* [Self-Edit] **Self-Edit: Fault-Aware Code Editor for Code Generation.** *Kechi Zhang, Zhuo Li, Jia Li, Ge Li, and Zhi Jin.* ACL. 2023. [[paper](https://aclanthology.org/2023.acl-long.45)]
* [Baldur] **Baldur: Whole-Proof Generation and Repair with Large Language Models.** *Emily First, Markus Rabe, Talia Ringer, and Yuriy Brun.* ESEC/FSE. 2023. [[paper](https://doi.org/10.1145/3611643.3616243)]

## Self-Correction with Information Retrieval

Previous work has proposed self-correction frameworks that use information retrieval during inference.

### with In-Context Learning (Information Retrieval)

* [RARR] **RARR: Researching and Revising What Language Models Say, Using Language Models.** *Luyu Gao, Zhuyun Dai, Panupong Pasupat, Anthony Chen, Arun Tejasvi Chaganty, Yicheng Fan, Vincent Zhao, Ni Lao, Hongrae Lee, Da-Cheng Juan, and Kelvin Guu.* ACL. 2023. [[paper](https://aclanthology.org/2023.acl-long.910)]
* [Verify-and-Edit] **Verify-and-Edit: A Knowledge-Enhanced Chain-of-Thought Framework.** *Ruochen Zhao, Xingxuan Li, Shafiq Joty, Chengwei Qin, and Lidong Bing.* ACL. 2023. [[paper](https://aclanthology.org/2023.acl-long.320)]
* **A Stitch in Time Saves Nine: Detecting and Mitigating Hallucinations of LLMs by Validating Low-Confidence Generation.** *Neeraj Varshney, Wenlin Yao, Hongming Zhang, Jianshu Chen, and Dong Yu.* Preprint. 2023. [[paper](https://arxiv.org/abs/2307.03987)]
* **Improving Language Models via Plug-and-Play Retrieval Feedback.** *Wenhao Yu, Zhihan Zhang, Zhenwen Liang, Meng Jiang, and Ashish Sabharwal.* Preprint. 2023. [[paper](https://arxiv.org/abs/2305.14002)]
* [CRITIC] **CRITIC: Large Language Models Can Self-Correct with Tool-Interactive Critiquing.** *Zhibin Gou, Zhihong Shao, Yeyun Gong, Yelong Shen, Yujiu Yang, Nan Duan, and Weizhu Chen.* ICLR. 2024. [[paper](https://openreview.net/forum?id=Sx038qxjek)]
* [FLARE] **Active Retrieval Augmented Generation.** *Zhengbao Jiang, Frank Xu, Luyu Gao, Zhiqing Sun, Qian Liu, Jane Dwivedi-Yu, Yiming Yang, Jamie Callan, and Graham Neubig.* EMNLP. 2023. [[paper](https://aclanthology.org/2023.emnlp-main.495)]

### with Trianing (Information Retrieval)

* 

## Self-Correction with Training Designed for Self-Correction

This section includes self-correction frameworks that train LLMs specifically for self-correction, but do not use external tools or information retrieval during inference.

### Supervised Fine-tuning

* **Self-critiquing models for assisting human evaluators.** *William Saunders, Catherine Yeh, Jeff Wu, Steven Bills, Long Ouyang, Jonathan Ward, and Jan Leike.* Preprint. 2022. [[paper](https://arxiv.org/abs/2206.05802)]
* [Re3] **Re3: Generating Longer Stories With Recursive Reprompting and Revision.** *Kevin Yang, Yuandong Tian, Nanyun Peng, and Dan Klein.* EMNLP. 2022. [[paper](https://aclanthology.org/2022.emnlp-main.296)]
* [SelFee] **SelFee: Iterative Self-Revising LLM Empowered by Self-Feedback Generation.** *Seonghyeon Ye, Yongrae Jo, Doyoung Kim, Sungdong Kim, Hyeonbin Hwang, and Minjoon Seo.* Blog post. 2023. [[blog](https://kaistai.github.io/SelFee/)]
* [Volcano] **Volcano: Mitigating Multimodal Hallucination through Self-Feedback Guided Revision.** *Seongyun Lee, Sue Park, Yongrae Jo, and Minjoon Seo.* NAACL. 2024. [[paper](https://aclanthology.org/2024.naacl-long.23)]
* [Self-corrective learning] **Generating Sequences by Learning to Self-Correct.** *Sean Welleck, Ximing Lu, Peter West, Faeze Brahman, Tianxiao Shen, Daniel Khashabi, and Yejin Choi.* ICLR. 2023. [[paper](https://openreview.net/forum?id=hH36JeQZDaO)]
* [REFINER] **REFINER: Reasoning Feedback on Intermediate Representations.** *Debjit Paul, Mete Ismayilzada, Maxime Peyrard, Beatriz Borges, Antoine Bosselut, Robert West, and Boi Faltings.* EACL. 2024. [[paper](https://aclanthology.org/2024.eacl-long.67)]
* [GLoRe] **GLoRe: When, Where, and How to Improve LLM Reasoning via Global and Local Refinements.** *Alex Havrilla, Sharath Raparthy, Christoforus Nalmpantis, Jane Dwivedi-Yu, Maksym Zhuravinskyi, Eric Hambro, and Roberta Raileanu.* Preprint. 2024. [[paper](https://arxiv.org/abs/2402.10963)].
* **Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters.** *Charlie Snell, Jaehoon Lee, Kelvin Xu, and Aviral Kumar.* Preprint. 2024. [[paper](https://arxiv.org/abs/2408.03314)].

### Reinforcement Learning

* [RL4F] **RL4F: Generating Natural Language Feedback with Reinforcement Learning for Repairing Model Outputs.** *Afra Feyza Akyurek, Ekin Akyurek, Ashwin Kalyan, Peter Clark, Derry Tanti Wijaya, and Niket Tandon.* ACL. 2023. [[paper](https://aclanthology.org/2023.acl-long.427)]
* [RISE] **Recursive Introspection: Teaching Language Model Agents How to Self-Improve.** *Yuxiao Qu, Tianjun Zhang, Naman Garg, and Aviral Kumar.* Preprint. 2024. [[paper](https://arxiv.org/abs/2407.18219)]
* [SCoRe] **Training Language Models to Self-Correct via Reinforcement Learning.** *Aviral Kumar, Vincent Zhuang, Rishabh Agarwal, Yi Su, John D Co-Reyes, Avi Singh, Kate Baumli, Shariq Iqbal, Colton Bishop, Rebecca Roelofs, Lei M Zhang, Kay McKinney, Disha Shrivastava, Cosmin Paduraru, George Tucker, Doina Precup, Feryal Behbahani, and Aleksandra Faust.* Preprint. 2024. [[paper](https://arxiv.org/abs/2409.12917)]

### o1-like Frameworks

OpenAI o1 is a framework focusing on improving reasoning capabilities of LLMs trained with **reinforcement learning** to explore multiple reasoning processess and correct their own mistakes during inference. After the release of OpenAI o1, several papers or projects have proposed frameworks similar to OpenAI o1.

* [OpenAI o1] **Learning to Reason with LLMs.** *OpenAI.* Blog post. 2024. [[blog](https://openai.com/index/learning-to-reason-with-llms/)] [[system card](https://cdn.openai.com/o1-system-card-20241205.pdf)]
* **Evaluation of OpenAI o1: Opportunities and Challenges of AGI.** *Tianyang Zhong, Zhengliang Liu, Yi Pan, Yutong Zhang, Yifan Zhou, Shizhe Liang, Zihao Wu, Yanjun Lyu, Peng Shu, Xiaowei Yu, Chao Cao, Hanqi Jiang, Hanxu Chen, Yiwei Li, Junhao Chen, Huawen Hu, Yihen Liu, Huaqin Zhao, Shaochen Xu, Haixing Dai, Lin Zhao, Ruidong Zhang, Wei Zhao, Zhenyuan Yang, Jingyuan Chen, Peilong Wang, Wei Ruan, Hui Wang, Huan Zhao, Jing Zhang, Yiming Ren, Shihuan Qin, Tong Chen, Jiaxi Li, Arif Hassan Zidan, Afrar Jahin, Minheng Chen, Sichen Xia, Jason Holmes, Yan Zhuang, Jiaqi Wang, Bochen Xu, Weiran Xia, Jichao Yu, Kaibo Tang, Yaxuan Yang, Bolun Sun, Tao Yang, Guoyu Lu, Xianqiao Wang, Lilong Chai, He Li, Jin Lu, Lichao Sun, Xin Zhang, Bao Ge, Xintao Hu, Lian Zhang, Hua Zhou, Lu Zhang, Shu Zhang, Ninghao Liu, Bei Jiang, Linglong Kong, Zhen Xiang, Yudan Ren, Jun Liu, Xi Jiang, Yu Bao, Wei Zhang, Xiang Li, Gang Li, Wei Liu, Dinggang Shen, Andrea Sikora, Xiaoming Zhai, Dajiang Zhu, and Tianming Liu.* Preprint. 2024. [[paper](https://arxiv.org/abs/2409.18486)].
* [Skywork-o1] **Skywork-o1 Open Series.** *Skywork-o1 Team.* Models. 2024. [[models](https://huggingface.co/Skywork)].
* [LLaVA-CoT] **LLaVA-CoT: Let Vision Language Models Reason Step-by-Step.** *Guowei Xu, Peng Jin, Hao Li, Yibing Song, Lichao Sun, and Li Yuan.* Preprint. 2024. [[paper](https://arxiv.org/abs/2411.10440)].
* [Marco-o1] **Marco-o1: Towards Open Reasoning Models for Open-Ended Solutions.** *Yu Zhao, Huifeng Yin, Bo Zeng, Hao Wang, Tianqi Shi, Chenyang Lyu, Longyue Wang, Weihua Luo, and Kaifu Zhang.* Preprint. 2024. [[paper](https://arxiv.org/abs/2411.14405)]
* [QwQ] **QwQ: Reflect Deeply on the Boundaries of the Unknown.** *Qwen Team.* Blog post. 2024. [[blog](https://qwenlm.github.io/blog/qwq-32b-preview/)]

For more papers related to OpenAI o1, please also refer to the following repository.

* [Awesome LLM Strawberry (OpenAI o1)](https://github.com/hijkzzz/Awesome-LLM-Strawberry)


## Contributing

We welcome contributions! If you’d like to add a new paper to this list, please submit a pull request. **Ensure that your commit and PR have descriptive and unique titles** rather than generic ones like "Updated README.md." 

Kindly use the following format for your entry:

```md
* [Short name (if exists)] **Paper Title.** *Author1, Author2, ... , and Last Author.* Conference/Journal/Preprint/Blog post. year. [[paper](url)] [[code, follow up paper, etc.](url)]
```

If you have any questions or suggestions, please feel free to open an issue or reach out to [Ryo Kamoi](https://ryokamoi.github.io/) (ryokamoi@psu.edu).

## License

Please refer to [LICENSE](LICENSE).
