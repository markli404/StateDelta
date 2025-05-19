import os
import json
import re
import string
import numpy as np
import pandas as pd
from collections import Counter
from typing import List, Union

from src.root_path import ROOT_PATH


class BaseDataset:
    def __init__(self):
        pass

    @classmethod
    def normalize_answer(cls, s):
        def remove_articles(text):
            return re.sub(r"\b(a|an|the)\b", " ", text)
        def white_space_fix(text):
            return " ".join(text.split())
        def remove_punc(text):
            exclude = set(string.punctuation)
            return "".join(ch for ch in text if ch not in exclude)
        def lower(text):
            return text.lower()
        return white_space_fix(remove_articles(remove_punc(lower(s))))

    @classmethod
    def exact_match_score(
        cls,
        prediction: str,
        ground_truth: Union[str, List[str]],
        ground_truth_id: Union[str, List[str]] = None
    ):
        ground_truths = {ground_truth} if isinstance(ground_truth, str) else set(ground_truth)
        if ground_truth_id and isinstance(ground_truth_id, str):
            ground_truths.update(cls.get_all_alias(ground_truth_id))

        correct = np.max([int(cls.normalize_answer(prediction) == cls.normalize_answer(gt)) for gt in ground_truths])
        return correct 

    @classmethod
    def f1_score(
        cls,
        prediction: str,
        ground_truth: Union[str, List[str]],
        ground_truth_id: Union[str, List[str]] = None
    ):
        ground_truths = {ground_truth} if isinstance(ground_truth, str) else set(ground_truth)
        if ground_truth_id and isinstance(ground_truth_id, str):
            ground_truths.update(cls.get_all_alias(ground_truth_id))
            
        final_metric = {"f1": 0, "precision": 0, "recall": 0}
        for ground_truth in ground_truths:
            normalized_prediction = cls.normalize_answer(prediction)
            normalized_ground_truth = cls.normalize_answer(ground_truth)
            if normalized_prediction in ["yes", "no", "noanswer"] and normalized_prediction != normalized_ground_truth:
                continue
            if normalized_ground_truth in ["yes", "no", "noanswer"] and normalized_prediction != normalized_ground_truth:
                continue
            prediction_tokens = normalized_prediction.split()
            ground_truth_tokens = normalized_ground_truth.split()
            common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
            num_same = sum(common.values())
            if num_same == 0:
                continue

            precision = 1.0 * num_same / len(prediction_tokens)
            recall = 1.0 * num_same / len(ground_truth_tokens)
            f1 = (2 * precision * recall) / (precision + recall)
            for k in ["f1", "precision", "recall"]:
                final_metric[k] = max(eval(k), final_metric[k])
        return final_metric

    def random_shuffle(self, seed=0):
        np.random.seed(seed)
        np.random.shuffle(self.dataset)
    
    def sample(self, sample_cnt: int):
        self.full_dataset = self.dataset
        if sample_cnt != -1:
            self.dataset = self.dataset[:sample_cnt]

    def parse_answer(self, output: str):
        return output

    def get_item(self, test_id):
        if hasattr(self, "full_dataset"):
            return self.full_dataset[test_id]
        else:
            return  self.dataset[test_id]


class GSM8K(BaseDataset):
    eval_metrics = ["em"]

    def __init__(self, data_root_path):
        with open(os.path.join(data_root_path, "GSM8K/test.jsonl")) as fin:
            lines = fin.readlines()
        self.dataset = []
        for idx, li in enumerate(lines):
            data = json.loads(li)
            data["test_id"] = idx
            ans = data["answer"]
            data["full_answer"] = ans
            data["answer"] = int(self.fetch_number(ans))
            self.dataset.append(data)

    def fetch_last_number(self, sentence: str):
        pattern = r"\d+\.?\d*"
        matches = re.findall(pattern, sentence)
        if matches:
            return matches[-1]
        return None

    def fetch_last_number_in_brackets(self, sentence: str):
        pattern = r"\{([0-9.,$]*)\}"
        matches = re.findall(pattern, sentence)
        for match_str in matches[::-1]:
            solu = re.sub(r"[^0-9.]", "", match_str)
            if solu:
                return solu 
        return None

    def fetch_number(self, output: str):
        try:
            pred = int(output)
        except:
            pred = self.fetch_last_number_in_brackets(output)
            if pred is None:
                pred = self.fetch_last_number(output)
        return pred
    
    def evaluate(self, output: str, test_id: int):
        ground_truth = self.get_item(test_id)["answer"]
        pred = self.fetch_number(output) 
        try:
            pred = int(pred)
            em = 1 if pred == ground_truth else 0
        except:
            em = 0
        return {"predict": output, "evaluate_predict": pred, "em": em}


class MMLU(BaseDataset):
    eval_metrics = ["em"]

    def __init__(self, data_root_path, field_name):
        assert field_name != "all", "not support all field_name in MMLU"

        def solve_row(test_id, row):
            qq = row["question"]
            if qq.endswith("."):
                qq = qq[:-1]
            question = "{}: A) {}, B) {}, C) {}, D) {}".format(
                qq, row["A"], row["B"], row["C"], row["D"])
            return {
                "question": question, 
                "test_id": test_id,
                "answer": row["answer"], 
                "origin_question": row["question"],
                "choices": {"A": row["A"], "B": row["B"], "C": row["C"], "D": row["D"]},
            }

        self.field_name = field_name
        filedir = os.path.join(data_root_path, "MMLU/test")
        filepath = os.path.join(filedir, f"{field_name}_test.csv")
        if not os.path.exists(filepath):
            raise ValueError(f"Invalid field_name {field_name}")
        df = pd.read_csv(filepath, header=None, names=["question", "A", "B", "C", "D", "answer"])
        self.dataset = []
        for idx in range(len(df)):
            self.dataset.append(solve_row(idx, df.iloc[idx]))
        
    def sample(self, sample_cnt: int):
        self.full_dataset = self.dataset
        if sample_cnt != -1:
            self.dataset = self.dataset[:sample_cnt]
         
    def parse_answer(self, input_str):
        pattern = r"\((\w)\)"
        matches = re.findall(pattern, input_str)
        solution = None
        for match_str in matches[::-1]:
            solution = match_str.upper()
            if solution:
                break
        return solution
    
    def fetch_last_choice(self, sentence: str):
        sentence = sentence.replace("\\", "").replace("(", " ").replace(")", " ").replace("{", " ").replace("}", " ").replace(".", " ")
        words = sentence.split()
        for word in words[::-1]:
            if word.endswith("."):
                word = word[:-1]
            if word in ["A", "B", "C", "D"]:
                return word
        return None

    def evaluate(self, output: str, test_id: int):
        data = self.get_item(test_id)
        ground_truth = data["answer"]
        choices = data["choices"]
        tmp = output.replace("(X)", "") # (X) is the example in prompt
        pred = self.parse_answer(tmp)
        if pred is None:
            if "boxed{" in output:
                pred = output[output.rfind("boxed{") + len("boxed{"):]
                pred = pred[:pred.find("}")]
                for choice, val in choices.items():
                    if val == pred:
                        pred = choice
                        break
                else:
                    pred = self.fetch_last_choice(tmp)
            else:
                pred = self.fetch_last_choice(tmp)
        em = 1 if pred == ground_truth else 0
        return {"predict": output, "evaluate_predict": pred, "em": em}


def multihopqa_reevaluate(output, test_id, super_eval_func):
    if "text{" in output:
        marked_answer = True
        pred = output[output.rfind("text{") + len("text{"):]
        pred = pred[:pred.find("}")]            
    elif "boxed{" in output: 
        marked_answer = True
        pred = output[output.rfind("boxed{") + len("boxed{"):]
        pred = pred[:pred.find("}")]
    else:
        marked_answer = False
        pred = output
    ret = super_eval_func(pred, test_id)
    ret["predict"] = output
    ret["marked_answer"] = marked_answer
    return ret


class WikiMultihopQA(BaseDataset):
    stop_words = ["\\boxed{", "\\text{"]
    eval_metrics = ["em", "f1", "precision", "recall"]

    def __init__(self, data_root_path: str, retrieval_topk: int = 0):
        with open(os.path.join(data_root_path, "2WikiMultihopQA/dev.json"), "r") as fin:
            dataset = json.load(fin)
        with open(os.path.join(data_root_path, "2WikiMultihopQA/id_aliases.json"), "r") as fin:
            aliases = dict()
            for li in fin:
                t = json.loads(li)
                aliases[t["Q_id"]] = t["aliases"]
        if retrieval_topk != 0:
            with open(os.path.join(ROOT_PATH, "wikidpr_retrieval", "2WikiMultihopQA.json"), "r") as fin:
                retrieval_passages = json.load(fin)
        else:
            from src.retriever import bm25_retrieve
            retrieval_passages = None
        self.dataset = []
        for did, data in enumerate(dataset):
            ans_id = data["answer_id"]
            val = {
                "qid": data["_id"], 
                "test_id": did, 
                "question": data["question"], 
                "answer": aliases[ans_id] if ans_id else data["answer"]
            }
            for key, value in data.items():
                if key not in ["_id", "question", "answer"]:
                    val[key] = value
            if retrieval_topk != 0:
                if retrieval_passages is not None:
                    val["passages"] = retrieval_passages[did]["passages"][:retrieval_topk]
                else:
                    val["passages"] = bm25_retrieve(
                        question=data["question"],
                        topk=retrieval_topk,
                    )
            self.dataset.append(val)
    
    def direct_evaluate(self, output: str, test_id: int):
        ground_truth = self.get_item(test_id)["answer"]
        pred = self.parse_answer(output) 
        em = self.exact_match_score(pred, ground_truth)
        f1_score = self.f1_score(pred, ground_truth)
        f1, prec, recall = f1_score["f1"], f1_score["precision"], f1_score["recall"]
        return {"predict": output, "evaluate_predict": pred, "em": em, "f1": f1, "precision": prec, "recall": recall}

    def evaluate(self, output, test_id):
        return multihopqa_reevaluate(output, test_id, self.direct_evaluate)


class StrategyQA(BaseDataset):
    stop_words = ["\\boxed{", "\\text{"]
    eval_metrics = ["em"]

    def __init__(self, data_root_path: str, retrieval_topk: int = 0):
        with open(os.path.join(data_root_path, "StrategyQA/strategyqa_train.json"), "r") as fin:
            dataset = json.load(fin)
        if retrieval_topk != 0:
            with open(os.path.join(ROOT_PATH, "wikidpr_retrieval", "StrategyQA.json"), "r") as fin:
                retrieval_passages = json.load(fin)
        else:
            from src.retriever import bm25_retrieve
            retrieval_passages = None
        self.dataset = []
        for did, data in enumerate(dataset):
            val = {
                "qid": data["qid"], 
                "test_id": did, 
                "question": data["question"], 
                "answer": "Yes" if data["answer"] == True else "No"
            }
            for key, value in data.items():
                if key not in ["qid", "question", "answer"]:
                    val[key] = value
            if retrieval_topk != 0:
                if retrieval_passages is not None:
                    val["passages"] = retrieval_passages[did]["passages"][:retrieval_topk]
                else:
                    val["passages"] = bm25_retrieve(
                        question=data["question"],
                        topk=retrieval_topk,
                    )
            self.dataset.append(val)
    
    def direct_evaluate(self, output: str, test_id: int):
        ground_truth = self.get_item(test_id)["answer"]
        pred = self.parse_answer(output) 
        em = self.exact_match_score(pred, ground_truth)
        return {"predict": output, "evaluate_predict": pred, "em": em}

    def evaluate(self, output, test_id):
        return multihopqa_reevaluate(output, test_id, self.direct_evaluate)


class ComplexWebQuestions(BaseDataset):
    stop_words = ["\\boxed{", "\\text{"]
    eval_metrics = ["em", "f1", "precision", "recall"]

    def __init__(self, data_root_path: str, retrieval_topk: int = 0):
        with open(os.path.join(data_root_path, "ComplexWebQuestions", "ComplexWebQuestions_dev.json"), "r") as fin:
            dataset = json.load(fin)
        if retrieval_topk != 0:
            with open(os.path.join(ROOT_PATH, "wikidpr_retrieval", "ComplexWebQuestions.json"), "r") as fin:
                retrieval_passages = json.load(fin)
        else:
            from src.retriever import bm25_retrieve
            retrieval_passages = None
        self.dataset = []
        for did, data in enumerate(dataset):
            question = data["question"]
            answer = []
            for ans in data["answers"]:
                answer.append(ans["answer"])
                answer.extend(ans["aliases"])
            answer = list(set(answer))
            val = {
                "qid": data["ID"],
                "test_id": did, 
                "question": question, 
                "answer": answer,
            }        
            for key, value in data.items():
                if key not in ["ID", "question", "answers"]:
                    val[key] = value
            if retrieval_topk != 0:
                if retrieval_passages is not None:
                    val["passages"] = retrieval_passages[did]["passages"][:retrieval_topk]
                else:
                    val["passages"] = bm25_retrieve(
                        question=data["question"],
                        topk=retrieval_topk,
                    )
            self.dataset.append(val)
    
    def direct_evaluate(self, output: str, test_id: int):
        ground_truth = self.get_item(test_id)["answer"]
        pred = self.parse_answer(output) 
        em = self.exact_match_score(pred, ground_truth)
        f1_score = self.f1_score(pred, ground_truth)
        f1, prec, recall = f1_score["f1"], f1_score["precision"], f1_score["recall"]
        return {"predict": output, "evaluate_predict": pred, "em": em, "f1": f1, "precision": prec, "recall": recall}

    def evaluate(self, output, test_id):
        return multihopqa_reevaluate(output, test_id, self.direct_evaluate)


class QuasarT(BaseDataset):
    stop_words = ["\\boxed{", "\\text{"]
    eval_metrics = ["em", "f1", "precision", "recall"]

    def __init__(self, data_root_path: str, retrieval_topk: int = 0):
        with open(os.path.join(data_root_path, "QuasarT/dev_questions.json"), "r") as fin:
            lines = fin.readlines()
        if retrieval_topk != 0:
            with open(os.path.join(ROOT_PATH, "wikidpr_retrieval", "QuasarT.json"), "r") as fin:
                retrieval_passages = json.load(fin)
        else:
            from src.retriever import bm25_retrieve
            retrieval_passages = None
        self.dataset = []
        for did, li in enumerate(lines):
            data = json.loads(li)
            val = {
                "qid": data["uid"], 
                "test_id": did, 
                "question": data["question"], 
                "answer": data["answer"],
            }
            for key, value in data.items():
                if key not in ["question", "answer"]:
                    val[key] = value
            if retrieval_topk != 0:
                if retrieval_passages is not None:
                    val["passages"] = retrieval_passages[did]["passages"][:retrieval_topk]
                else:
                    val["passages"] = bm25_retrieve(
                        question=data["question"],
                        topk=retrieval_topk,
                    )
            self.dataset.append(val)
    
    def direct_evaluate(self, output: str, test_id: int):
        ground_truth = self.get_item(test_id)["answer"]
        pred = self.parse_answer(output) 
        em = self.exact_match_score(pred, ground_truth)
        f1_score = self.f1_score(pred, ground_truth)
        f1, prec, recall = f1_score["f1"], f1_score["precision"], f1_score["recall"]
        return {"predict": output, "evaluate_predict": pred, "em": em, "f1": f1, "precision": prec, "recall": recall}

    def evaluate(self, output, test_id):
        return multihopqa_reevaluate(output, test_id, self.direct_evaluate)


class FEVER(BaseDataset):
    eval_metrics = ["em"]

    def __init__(self, data_root_path: str):
        with open(os.path.join(data_root_path, "FEVER2.0/fever2-fixers-dev.jsonl"), "r") as fin:
            lines = fin.readlines()
        self.dataset = []
        for did, li in enumerate(lines):
            data = json.loads(li)
            val = {
                "qid": data["id"], 
                "test_id": did, 
                "question": data["claim"], 
                "answer": data["label"],
            }
            for key, value in data.items():
                if key not in ["id", "claim", "label"]:
                    val[key] = value
            self.dataset.append(val)
    
    def direct_evaluate(self, output: str, test_id: int):
        ground_truth = self.get_item(test_id)["answer"]
        pred = self.parse_answer(output) 
        em = self.exact_match_score(pred, ground_truth)
        return {"predict": output, "evaluate_predict": pred, "em": em}

    def evaluate(self, output, test_id):
        return multihopqa_reevaluate(output, test_id, self.direct_evaluate) 
    

class HotpotQA(BaseDataset):
    stop_words = ["\\boxed{", "\\text{"]
    eval_metrics = ["em", "f1", "precision", "recall"]

    def __init__(self, data_root_path: str):
        data_path = os.path.join(data_root_path, "HotpotQA/hotpot_dev_distractor_v1.json")
        with open(data_path, "r") as fin:
            dataset = json.load(fin)
        self.dataset = []
        for did, data in enumerate(dataset):
            val = {
                "qid": data["_id"], 
                "test_id": did, 
                "question": data["question"], 
                "answer": data["answer"]
            }
            for key, value in data.items():
                if key not in ["_id", "question", "answer"]:
                    val[key] = value
            val["type"] = data["type"]
            self.dataset.append(val)
    
    def direct_evaluate(self, output: str, test_id: int):
        ground_truth = self.get_item(test_id)["answer"]
        pred = self.parse_answer(output) 
        em = self.exact_match_score(pred, ground_truth)
        f1_score = self.f1_score(pred, ground_truth)
        f1, prec, recall = f1_score["f1"], f1_score["precision"], f1_score["recall"]
        return {"predict": output, "evaluate_predict": pred, "em": em, "f1": f1, "precision": prec, "recall": recall}

    def evaluate(self, output, test_id):
        return multihopqa_reevaluate(output, test_id, self.direct_evaluate)
