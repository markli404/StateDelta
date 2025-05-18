import os
import yaml
import copy
import time
import torch
from typing import Union, List

from src.tasks.utils import (
    get_merged_embedding_for_cipher,
    get_merged_ids_mask_hs_for_sde,
)

from src.tasks.base_task import BaseTask


class DebateTask(BaseTask):
    def __init__(self, agents, dataset, args):
        super().__init__(agents, dataset, args)
        if args.method == "single":
            if len(self.agents) > 0:
                self.agents = [self.agents[0]]
            self.args.agent_cnt = 1
        with open(args.prompt_file) as fin:
            name = "mmlu" if args.dataset.startswith("mmlu_") else args.dataset
            self.prompt_template = yaml.safe_load(fin)[name]
    
    def run(self, data):
        question = data["question"]
        ground_truth = data["answer"] 
        detail = {
            "question": question, 
            "ground_truth": ground_truth,
        }
        prompts = copy.deepcopy(self.prompt_template)
        for kk in prompts:
            prompts[kk] = prompts[kk].replace("{question}", question)
        
        run_time = self.run_func(prompts)

        for agent_idx, agent in enumerate(self.agents):
            det = self.dataset.evaluate(
                output=agent.final_output_text,
                test_id=data["test_id"],
            )
            for met in det:
                if met in self.dataset.eval_metrics:
                    try:
                        det[met] = det[met].item()
                    except:
                        pass
            det["history"] = agent.history
            detail[f"agent_{agent_idx}"] = det
        detail["run_time(s)"] = run_time
        return detail
    
    def run_func(self, prompts):
        raise NotImplementedError

    def generate_result(self, details):
        res = {
            "run_time(s)": sum([d["run_time(s)"] for d in details]) / len(details),
        }
        agent_cnt = self.args.agent_cnt
        for agent_idx in range(agent_cnt):
            agent_res = {
                met: sum([float(d[f"agent_{agent_idx}"][met]) for d in details]) / len(details)
                for met in self.dataset.eval_metrics
            }
            res[f"agent_{agent_idx}"] = agent_res
        res["average"] = {
            met: sum([float(res[f"agent_{agent_idx}"][met]) for agent_idx in range(agent_cnt)]) / agent_cnt
            for met in self.dataset.eval_metrics
        }
        return res


class SingleDebateTask(DebateTask):
    def run_func(self, prompts):
        start_time = time.perf_counter()
        agent = self.agents[0]
        prompt = prompts["first_prompt"]
        agent.init_history(first_user_prompt=prompt)
        output = agent.generate(agent.history_msgs)
        agent.final_output_text = output
        agent.history = agent.history_msgs
        end_time = time.perf_counter()
        return end_time - start_time


class NlDebateTask(DebateTask):
    def run_func(self, prompts):
        start_time = time.perf_counter()
        agent_cnt = self.args.agent_cnt
        # 首次独立出去
        for agent in self.agents:
            agent.init_history(first_user_prompt=prompts["first_prompt"])
            agent.generate(agent.history_msgs)
        
        for rd in range(self.args.rounds-1): # 第一轮独立出去了
            for cur in range(agent_cnt):
                agent = self.agents[cur]
                all_other_resp = ""
                for other in range(agent_cnt):
                    if other != cur:
                        other_resp = self.agents[other].assistant_output[rd]
                        all_other_resp += prompts["other_response_prompt"].replace("{other_response}", other_resp)
                agent.history_msgs.append({
                    "role": "user",
                    "content": prompts["debate_prompt"].replace("{all_other_response}", all_other_resp)
                })
                agent.generate(agent.history_msgs)
        for agent in self.agents:
            agent.final_output_text = agent.assistant_output[-1]
        end_time = time.perf_counter()
        for agent in self.agents:
            agent.history = agent.history_msgs
        return end_time - start_time


class CipherDebateTask(DebateTask):
    def run_func(self, prompts):
        start_time = time.perf_counter()
        agent_cnt = self.args.agent_cnt
        # 首次独立出去
        for agent in self.agents:
            agent.init_history(first_user_prompt=prompts["first_prompt"])
            agent.generate(agent.history_embs)
        
        for rd in range(self.args.rounds-1): # 第一轮独立出去了
            for cur in range(agent_cnt):
                agent = self.agents[cur]
                merged_other_embs = []
                for other in range(agent_cnt):
                    if cur != other:
                        merged_other_embs.append(
                            get_merged_embedding_for_cipher(
                                t2e_func=agent.text_to_embedding, 
                                prompt_template=prompts["other_response_prompt"],
                                placeholder="{other_response}",
                                input_embs=self.agents[other].assistant_output[rd]
                            )
                        )
                user_embs = get_merged_embedding_for_cipher(
                    t2e_func=agent.text_to_embedding,
                    prompt_template=prompts["debate_prompt"],
                    placeholder="{all_other_response}",
                    input_embs=torch.cat(merged_other_embs, dim=0),
                )
                agent.history_embs = torch.cat([
                    agent.history_embs, 
                    agent.user_embs_fr, 
                    user_embs, 
                    agent.user_embs_ed, 
                    agent.assistant_embs_fr, 
                ], dim=0)
                agent.generate(agent.history_embs)
        for agent in self.agents:
            agent.final_output_text = agent.get_human_output(agent.assistant_output[-1])
        end_time = time.perf_counter()
        for agent in self.agents:
            agent.history = agent.get_human_output(agent.history_embs)
        return end_time - start_time


class SDEDebateTask(DebateTask):
    def run_func(self, prompts):
        start_time = time.perf_counter()
        agent_cnt = self.args.agent_cnt
        # 首次独立出去
        for agent in self.agents:
            agent.init_history(first_user_prompt=prompts["first_prompt"])
            agent.generate(
                input_ids = agent.history_ids,
                if_edit=False,
                edit_layer_idx=self.args.edit_layer_idx,
            )
        
        for rd in range(self.args.rounds-1): # 第一轮独立出去了
            for cur in range(agent_cnt):
                agent = self.agents[cur]
                # 获取其他模型的输出
                merged_input_ids = []
                merged_mask = []
                merged_hs = []
                for other in range(agent_cnt):
                    if other != cur:
                        resp_ids = self.agents[other].assistant_ids[rd]
                        ids, mask, hs = get_merged_ids_mask_hs_for_sde(
                            tokenizer=agent.tokenizer,
                            prompt_template=prompts["other_response_prompt"],
                            placeholder="{other_response}",
                            input_ids=resp_ids,  
                            input_mask=torch.zeros(len(resp_ids), dtype=torch.bool),
                            input_hs=self.agents[other].assistant_hs[rd],
                        )
                        merged_input_ids.append(ids)
                        merged_mask.append(mask)
                        merged_hs.append(hs)
                all_resp_ids = []
                for ids in merged_input_ids:
                    all_resp_ids += ids
                user_input_ids, user_mask, user_hs = get_merged_ids_mask_hs_for_sde(
                    tokenizer=agent.tokenizer,
                    prompt_template=prompts["debate_prompt"],
                    placeholder="{all_other_response}",
                    input_ids=all_resp_ids,
                    input_mask=torch.cat(merged_mask, dim=0),
                    input_hs={
                        layer_idx: torch.cat([merged_hs[_][layer_idx] for _ in range(len(merged_hs))], dim=1) 
                        for layer_idx in self.args.edit_layer_idx
                    },
                )
                history_mask = torch.cat([
                    torch.zeros(len(agent.history_ids) + len(agent.user_prompt_fr), dtype=torch.bool),
                    user_mask,
                    torch.zeros(len(agent.user_prompt_ed) + len(agent.assistant_prompt_fr), dtype=torch.bool),
                ], dim=0)
                history_hs = {}
                for layer_idx, hs in user_hs.items():
                    history_hs[layer_idx] = torch.cat([
                        torch.zeros((1, len(agent.history_ids)+len(agent.user_prompt_fr), hs.shape[-1])),
                        hs,
                        torch.zeros((1, len(agent.user_prompt_ed)+len(agent.assistant_prompt_fr), hs.shape[-1])),
                    ], dim=1)
                agent.history_ids = agent.history_ids + agent.user_prompt_fr + \
                    user_input_ids + agent.user_prompt_ed + agent.assistant_prompt_fr
                agent.generate(
                    input_ids=agent.history_ids, 
                    if_edit=True, 
                    edit_layer_idx=self.args.edit_layer_idx,
                    edit_mask=history_mask,
                    edit_tensor=history_hs,
                )
        for agent in self.agents:
            agent.final_output_text = agent.get_human_output(agent.assistant_ids[-1])
        end_time = time.perf_counter()
        for agent in self.agents:
            agent.history = agent.get_human_output(agent.history_ids)
        return end_time - start_time