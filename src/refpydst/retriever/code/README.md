# Retriever Training

Many of the methods for training a supervised retriever with contrastive learning 
in this folder are adapted from the methods and code for 
["In-Context Learning for Dialogue State Tracking"](https://arxiv.org/abs/2203.08568)
found in [Yushi-Hu/IC-DST](https://github.com/Yushi-Hu/IC-DST/).

```bibtex
@inproceedings{hu-etal-2022-context,
    title = "In-Context Learning for Few-Shot Dialogue State Tracking",
    author = "Hu, Yushi  and
      Lee, Chia-Hsuan  and
      Xie, Tianbao  and
      Yu, Tao  and
      Smith, Noah A.  and
      Ostendorf, Mari",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2022",
    month = dec,
    year = "2022",
    address = "Abu Dhabi, United Arab Emirates",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.findings-emnlp.193",
    pages = "2627--2643",
    abstract = "Collecting and annotating task-oriented dialogues is time-consuming and costly. Thus, zero and few shot learning for dialogue tasks presents an exciting opportunity. In this work, we propose an in-context (IC) learning framework for zero-shot and few-shot learning dialogue state tracking (DST), where a large pretrained language model (LM) takes a test instance and a few exemplars as input, and directly decodes the dialogue state without any parameter updates. This approach is more flexible and scalable than prior DST work when adapting to new domains and scenarios. To better leverage a tabular domain description in the LM prompt, we reformulate DST into a text-to-SQL problem. We also propose a novel approach to retrieve annotated dialogues as exemplars. Empirical results on MultiWOZ show that our method IC-DST substantially outperforms previous fine-tuned state-of-the-art models in few-shot settings. In addition, we test IC-DST in zero-shot settings, in which the model only takes a fixed task instruction as input, finding that it outperforms previous zero-shot methods by a large margin.",
}
```
