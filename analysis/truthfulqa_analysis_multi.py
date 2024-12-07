import torch
from tqdm import tqdm
import pandas as pd


import sys
import os

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.utils import load_multi_dexperts_model_and_tokenizer
from analysis.utils import flatten_batch_results, summarize_results


def build_prompt(input_text):
    return f"""### Instruction:
Given a sentence, correct its grammar if needed and translate it into both English and Spanish. If the input is in English, provide the corrected English sentence first, followed by the Spanish translation. If the input is in Spanish, provide the corrected Spanish sentence first, followed by the English translation.

Examples:
1. Input: My brother are very tall
   Output: My brother is very tall - Mi hermano es muy alto

2. Input: Nosotros comé pizza anoche
   Output: Nosotros comimos pizza anoche - We ate pizza last night

### Input:
{input_text}

### Response:
"""

@torch.inference_mode()
def main():
    # load model
    model, tokenizer = load_multi_dexperts_model_and_tokenizer(
        base_model_name_or_path='meta-llama/Llama-2-13b-hf',
        expert_model_names_or_paths=['dteran/Llama-2-7b-hf-ranslation', 'dteran/Llama-2-7b-hf-grammar', 'dteran/Llama-2-7b-hf-grammar-and-translation'],
        expert_weights=[0.25, 0.25, 1.0],
        chat_response_prefix='### Response:'
    )

    # truthfulqa_df = pd.read_csv('data/eval/truthfulqa/TruthfulQA.csv')

    # print(truthfulqa_df.head(10))
    # print(f"TruthfulQA dataset dimensions: {truthfulqa_df.shape}")
    # truthfulqa_df = truthfulqa_df.head(2)

    # # construct prompts
    # prompts = [row['Question'] + '\n\nAnswer:' for _, row in truthfulqa_df.iterrows()]

    # print(prompts)

    # alpaca_prompt1 = """
    # ### Instruction:
    # Correct the grammar

    # ### Input:
    # The people are eat the food

    # ### Response:
    # """

    # alpaca_prompt2 = """
    # ### Instruction:
    # Given a sentence, correct its grammar if needed and translate it into both English and Spanish. If the input is in English, provide the corrected English sentence first, followed by the Spanish translation. If the input is in Spanish, provide the corrected Spanish sentence first, followed by the English translation.

    # ### Input:
    # The people is running to the store

    # ### Response:
    # """

    # prompts = [alpaca_prompt1, alpaca_prompt2]

    # load dataset
    import json
    with open('data/downloads/grammas_and_translation_dataset_with_instructions_test.jsonl', 'r') as f:
        data = json.load(f)
    
    # Take first few examples for testing
    # data = data[:2]  # Remove this line later to process all examples
    
    # Build prompts using the alpaca format
    prompts = [build_prompt(item['input']) for item in data]
    
    print("Sample prompt:")
    print(prompts)
    
    
    # Rest of your code remains the same
    batch_size = 2
    all_results = []
    for i in tqdm(range(0, len(prompts), batch_size), desc="Batches"):
        batch_prompts = prompts[i: i + batch_size]
        batch_inputs = tokenizer(batch_prompts, return_tensors='pt', padding='longest')
        input_ids = batch_inputs.input_ids.cuda()
        attention_mask = batch_inputs.attention_mask.cuda()
        _, results = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=512,
            do_sample=False,
            return_logits_for_analysis=True
        )

        
        results = flatten_batch_results(results)
        shortened_results = summarize_results(results)
        all_results.extend(shortened_results)


    torch.save(all_results, 'analysis/pkl/test_triple_expert_gt_data.pkl')


if __name__ == "__main__":
    main()
