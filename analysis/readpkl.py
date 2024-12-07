import torch
import json
from transformers import AutoTokenizer

def convert_to_readable(pkl_file, base_model_name, expert_model_name):
    # Load the tokenizer
    base_tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    expert_tokenizer = AutoTokenizer.from_pretrained(expert_model_name)
    
    # Load results
    results = torch.load(pkl_file)
    # print(results)
    
    readable_results = []
    for result in results:
        # print(result.keys())
        output = {
            'tokens': result['tokens'],
            'text': base_tokenizer.decode(result['token_ids']),
            'probabilities': {
                'base': result['p_base'].tolist() if torch.is_tensor(result['p_base']) else result['p_base'],
                'expert': result['p_expert'].tolist() if torch.is_tensor(result['p_expert']) else result['p_expert']
            },
            'predictions': {
                'base': base_tokenizer.decode(result['preds_base']),
                'expert': expert_tokenizer.decode(result['preds_expert']),
                'dexperts': expert_tokenizer.decode(result['preds_dexperts']),
                'antiexpert': expert_tokenizer.decode(result['preds_antiexpert'])
            }
        }
        readable_results.append(output)
    
    # Save as JSON for readability
    with open(pkl_file.replace('.pkl', '_readable.json'), 'w') as f:
        json.dump(readable_results, f, indent=2)

    return readable_results

# Usage
results = convert_to_readable(
    'analysis/pkl/test_triple_expert_gt_data.pkl',
    'meta-llama/Llama-2-13b-hf',
    'meta-llama/Llama-2-7b-hf'
)


# import matplotlib.pyplot as plt
# def plot_token_probabilities(probabilities, output_path):
#     plt.bar(range(len(probabilities)), probabilities)
#     plt.title('Base Model Token Probabilities')
#     plt.xlabel('Token Index')
#     plt.ylabel('Probability')
#     plt.savefig(output_path)
#     plt.close()

# Print first result
for result in results:

    # print(result['tokens'])
    print(result['text'])
    for key, value in result['predictions'].items():
        print(key)
        print(value)
        print('*************************')

    
    # Create a line plot of probabilities
    # base_probs = result['probabilities']['base']
    # plot_token_probabilities(base_probs, 'analysis/plots/base_probs.png')
    # expert_probs = result['probabilities']['expert']
    # plot_token_probabilities(expert_probs, 'analysis/plots/expert_probs.png')

   
    print('----------------------------------')