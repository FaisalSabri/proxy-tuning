from typing import Optional, Dict, Any, List
import torch
from transformers import AutoModelForCausalLM, PreTrainedTokenizer
import torch.nn.functional as F
from transformers.generation.utils import (
    ModelOutput,
    StoppingCriteriaList,
    LogitsProcessorList
)
from collections import defaultdict
from modeling.dexperts import top_k_top_p_filtering

B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

class MultiDExpertsLlama:
    def __init__(
        self,
        base_model_name_or_path: str,
        expert_model_names_or_paths: List[str],
        expert_weights: List[float],
        antiexpert_model_name_or_path: str,
        tokenizer: PreTrainedTokenizer,
        system_prompt: str = None,
        chat_response_prefix: str = None,
        model_kwargs: Dict[str, Any] = None
    ):
        """
        Initialize MultiDExpertsLlama with multiple expert models.
        
        Args:
            expert_model_names_or_paths: List of paths to expert models
            expert_weights: List of weights (alphas) for each expert
            Other args same as DExpertsLlama
        """
        assert len(expert_model_names_or_paths) == len(expert_weights), \
            "Number of expert models must match number of weights"

        self.base = AutoModelForCausalLM.from_pretrained(
            base_model_name_or_path, **model_kwargs
        )

        self.experts = []
        for expert_path in expert_model_names_or_paths:
            expert = AutoModelForCausalLM.from_pretrained(
                expert_path, **model_kwargs
            )
            expert.eval()
            self.experts.append(expert)

        self.antiexpert = AutoModelForCausalLM.from_pretrained(
            antiexpert_model_name_or_path, **model_kwargs
        )
        self.antiexpert.eval()

        self.expert_weights = expert_weights
        self.base.eval()
        
        self.tokenizer = tokenizer
        self.device = self.base.device
        self.chat_response_prefix = chat_response_prefix

        # Llama chat experts need different formatting
        self.use_chat_format_for_expert = any('chat' in path.lower() for path in expert_model_names_or_paths)

        if self.use_chat_format_for_expert:
            self.chat_prefix = "[INST]"
            self.chat_suffix = "[/INST]"

            if system_prompt:
                self.chat_prefix += f"{B_SYS}{system_prompt}{E_SYS}"

            if self.chat_response_prefix:
                self.chat_suffix += f" {chat_response_prefix}"

    def forward(
        self,
        base_inputs,
        expert_inputs_list,
        antiexpert_inputs,
        return_dict=None
    ):
        base_outputs = self.base(**base_inputs, return_dict=return_dict)
        
        expert_outputs = []
        for expert, expert_inputs in zip(self.experts, expert_inputs_list):
            expert_output = expert(**expert_inputs, return_dict=return_dict)
            expert_outputs.append(expert_output)
            
        antiexpert_outputs = self.antiexpert(**antiexpert_inputs, return_dict=return_dict)

        return base_outputs, expert_outputs, antiexpert_outputs

    # Rest of the methods same as DExpertsLlama, with modifications to handle multiple experts
    def _get_tokenized_chat_inputs(self, input_ids):
        """Same as DExpertsLlama._get_tokenized_chat_inputs"""
        prompts = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)

        if self.chat_response_prefix:
            cleaned_prompts = []
            for p in prompts:
                if self.chat_response_prefix in p:
                    p = p.replace(self.chat_response_prefix, '').rstrip()
                cleaned_prompts.append(p)
        else:
            cleaned_prompts = prompts

        chat_prompts = [f'{self.chat_prefix} {p} {self.chat_suffix}' for p in cleaned_prompts]
        chat_inputs = self.tokenizer(
            chat_prompts, padding="longest", return_tensors="pt",
            add_special_tokens=True
        )
        chat_inputs.input_ids = chat_inputs.input_ids.to(self.device)
        chat_inputs.attention_mask = chat_inputs.attention_mask.to(self.device)

        return chat_inputs

    def generate(
        self,
        input_ids: Optional[torch.Tensor] = None,
        max_new_tokens: Optional[int] = 100,
        do_sample: bool = False,
        top_p: float = 1.0,
        temperature: float = 1.0,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        return_logits_for_analysis: bool = False,
        **kwargs
    ):
        base_kwargs = kwargs.copy()
        expert_kwargs_list = [kwargs.copy() for _ in self.experts]  # One kwargs dict per expert
        antiexpert_kwargs = kwargs.copy()

        if self.use_chat_format_for_expert:
            chat_inputs = self._get_tokenized_chat_inputs(input_ids)
            expert_input_ids = chat_inputs.input_ids.to(input_ids.device)
            for expert_kwargs in expert_kwargs_list:
                expert_kwargs['attention_mask'] = chat_inputs.attention_mask
        else:
            expert_input_ids = input_ids.to(input_ids.device)

        unfinished_sequences = torch.ones(input_ids.shape[0], dtype=torch.long, device=input_ids.device)
        eos_token_id_tensor = torch.tensor([self.tokenizer.eos_token_id]).to(input_ids.device)

        if return_logits_for_analysis:
            analysis_data = defaultdict(list)

        for step in range(max_new_tokens):
            base_inputs = self.base.prepare_inputs_for_generation(input_ids, **base_kwargs)
            
            expert_inputs_list = [
                expert.prepare_inputs_for_generation(expert_input_ids, **expert_kwargs)
                for expert, expert_kwargs in zip(self.experts, expert_kwargs_list)
            ]
            
            antiexpert_inputs = self.antiexpert.prepare_inputs_for_generation(input_ids, **antiexpert_kwargs)

            base_outputs, expert_outputs_list, antiexpert_outputs = self.forward(
                base_inputs, expert_inputs_list, antiexpert_inputs, return_dict=True
            )

            # Get next token logits
            base_next_token_logits = base_outputs.logits[..., -1, :]
            antiexpert_next_token_logits = antiexpert_outputs.logits[..., -1, :]

            # Combine expert logits according to weights
            expert_logits_sum = torch.zeros_like(base_next_token_logits)
            for expert_outputs, weight in zip(expert_outputs_list, self.expert_weights):
                expert_next_token_logits = expert_outputs.logits[..., -1, :]
                expert_next_token_logits = expert_next_token_logits[:, :base_next_token_logits.shape[-1]]
                expert_logits_sum += weight * (expert_next_token_logits - antiexpert_next_token_logits)

            next_token_logits = base_next_token_logits + expert_logits_sum

            if logits_processor:
                next_token_logits = logits_processor(input_ids, next_token_logits)

            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature
            if top_p < 1.0:
                next_token_logits = top_k_top_p_filtering(next_token_logits, top_p=top_p)

            if do_sample:
                probs = F.softmax(next_token_logits, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                next_tokens = torch.argmax(next_token_logits, dim=-1)

            next_tokens = next_tokens * unfinished_sequences + self.tokenizer.pad_token_id * (1 - unfinished_sequences)

            # Update inputs
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            expert_input_ids = torch.cat([expert_input_ids, next_tokens[:, None]], dim=-1)

            # Update kwargs for next iteration
            base_kwargs = self._update_model_kwargs_for_generation(base_outputs, base_kwargs)
            for i, expert_outputs in enumerate(expert_outputs_list):
                expert_kwargs_list[i] = self._update_model_kwargs_for_generation(expert_outputs, expert_kwargs_list[i])
            antiexpert_kwargs = self._update_model_kwargs_for_generation(antiexpert_outputs, antiexpert_kwargs)

            if stopping_criteria and stopping_criteria(input_ids, None):
                break

            unfinished_sequences = unfinished_sequences.mul(
                next_tokens.tile(eos_token_id_tensor.shape[0], 1).ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0)
            )

            if unfinished_sequences.max() == 0:
                break

            if return_logits_for_analysis:
                next_token_logits_dict = {
                    'dexperts': next_token_logits,
                    'base': base_next_token_logits,
                    'expert': expert_logits_sum,  # Using combined expert logits
                    'antiexpert': antiexpert_next_token_logits
                }
                analysis_data = self.update_analysis_data(analysis_data, next_tokens, next_token_logits_dict)

        if return_logits_for_analysis:
            for k in analysis_data.keys():
                if k.startswith('logits'):
                    analysis_data[k] = torch.cat(analysis_data[k], dim=1)
            return input_ids, analysis_data

        return input_ids

    def _update_model_kwargs_for_generation(
        self,
        outputs: ModelOutput,
        kwargs: Dict[str, Any],
    ) -> Dict[str, Any]:
        kwargs["past_key_values"] = outputs.past_key_values
        if "attention_mask" in kwargs:
            attention_mask = kwargs["attention_mask"]
            kwargs["attention_mask"] = torch.cat(
                [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
            )
        return kwargs 

    def update_analysis_data(self, analysis_data, next_tokens, next_token_logits_dict):
        """Helper method to update analysis data during generation"""
        analysis_data['tokens'].append([self.tokenizer.decode(t) for t in next_tokens])
        analysis_data['token_ids'].append(next_tokens)

        # logits from each model for the next token
        for model in next_token_logits_dict.keys():
            analysis_data[f'logits_{model}'].append(next_token_logits_dict[model].unsqueeze(dim=1))

        return analysis_data