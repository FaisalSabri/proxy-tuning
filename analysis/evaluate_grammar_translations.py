import torch
import json
from transformers import AutoTokenizer
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
from nltk.translate.gleu_score import sentence_gleu
import nltk
import spacy
import errant
from langdetect import detect
nltk.download('wordnet')

# Load spaCy models
try:
    nlp_eng = spacy.load('en_core_web_sm')
    nlp_es = spacy.load('es_core_news_sm')
except OSError:
    # If models aren't installed, download them
    print("Downloading required spaCy models...")
    import os
    os.system('python -m spacy download en_core_web_sm')
    os.system('python -m spacy download es_core_news_sm')
    nlp_eng = spacy.load('en_core_web_sm')
    nlp_es = spacy.load('es_core_news_sm')

def convert_to_readable(pkl_file, base_model_name, expert_model_name):
    # Load the tokenizer
    base_tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    expert_tokenizer = AutoTokenizer.from_pretrained(expert_model_name)
    
    # Load results and ground truth data
    results = torch.load(pkl_file)

    with open('data/downloads/grammas_and_translation_dataset_with_instructions_test.jsonl', 'r') as f:
        ground_truth = json.load(f)
    
    readable_results = []
    metrics = {
        'bleu_scores': [],
        'meteor_scores': [],
        'gleu_scores': [],
        'errant_scores': [],
        'hallucination_count': 0
    }
    
    for idx, result in enumerate(results):
        # Get model outputs
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
        
        # Calculate metrics
        reference = ground_truth[idx]['output'].split()
        hypothesis = output['predictions']['dexperts'].split()

        print(reference)
        print(hypothesis)

        # Split languages
        ref_sep_idx = reference.index('-')
        hyp_sep_idx = hypothesis.index('-')
        
        ref_first = reference[:ref_sep_idx]
        ref_second = reference[ref_sep_idx + 1:]
        hyp_first = hypothesis[:hyp_sep_idx]
        hyp_second = hypothesis[hyp_sep_idx + 1:]
        
        # Detect language for both reference and hypothesis parts
        ref_first_lang = detect_language(ref_first)
        hyp_first_lang = detect_language(hyp_first)
        
        # Align segments based on language detection
        if ref_first_lang == hyp_first_lang:
            # Same language order in both reference and hypothesis
            ref_eng = ref_first if ref_first_lang == 'en' else ref_second
            ref_es = ref_first if ref_first_lang == 'es' else ref_second
            hyp_eng = hyp_first if hyp_first_lang == 'en' else hyp_second
            hyp_es = hyp_first if hyp_first_lang == 'es' else hyp_second
        else:
            # Languages may be swapped between reference and hypothesis
            ref_eng = ref_first if ref_first_lang == 'en' else ref_second
            ref_es = ref_first if ref_first_lang == 'es' else ref_second
            hyp_eng = hyp_second if hyp_first_lang == 'es' else hyp_first
            hyp_es = hyp_second if hyp_first_lang == 'en' else hyp_first
            
        # Calculate all metrics using aligned language segments
        bleu_eng = sentence_bleu([ref_eng], hyp_eng)
        bleu_es = sentence_bleu([ref_es], hyp_es)
        bleu_score = (bleu_eng + bleu_es) / 2
        
        meteor_eng = meteor_score([ref_eng], hyp_eng)
        meteor_es = meteor_score([ref_es], hyp_es)
        meteor = (meteor_eng + meteor_es) / 2
        
        gleu_eng = sentence_gleu([ref_eng], hyp_eng)
        gleu_es = sentence_gleu([ref_es], hyp_es)
        gleu_score = (gleu_eng + gleu_es) / 2
        
        # ERRANT score (passing already aligned segments)
        errant_score = calculate_errant_aligned(ref_eng, hyp_eng, ref_es, hyp_es)
        
        metrics['bleu_scores'].append(bleu_score)
        metrics['meteor_scores'].append(meteor)
        metrics['gleu_scores'].append(gleu_score)
        metrics['errant_scores'].append(errant_score)
        
        # Check for hallucination
        if '#' in ''.join(hypothesis):
            print(f"\nHallucination detected in hypothesis: {' '.join(hypothesis)}")
            metrics['hallucination_count'] += 1
            output['metrics'] = {
                'bleu': {'overall': 0.0, 'english': 0.0, 'spanish': 0.0},
                'meteor': {'overall': 0.0, 'english': 0.0, 'spanish': 0.0},
                'gleu': {'overall': 0.0, 'english': 0.0, 'spanish': 0.0},
                'errant': 0.0,
                'is_hallucination': True
            }
            metrics['bleu_scores'].append(0.0)
            metrics['meteor_scores'].append(0.0)
            metrics['gleu_scores'].append(0.0)
            metrics['errant_scores'].append(0.0)
            readable_results.append(output)
            continue
        
        # Add detailed metrics to output
        output['metrics'] = {
            'bleu': {'overall': bleu_score, 'english': bleu_eng, 'spanish': bleu_es},
            'meteor': {'overall': meteor, 'english': meteor_eng, 'spanish': meteor_es},
            'gleu': {'overall': gleu_score, 'english': gleu_eng, 'spanish': gleu_es},
            'errant': errant_score
        }
        
        readable_results.append(output)

        # break
        
    
    # Calculate average metrics
    total_samples = len(results)
    avg_metrics = {
        'average_bleu': sum(metrics['bleu_scores']) / len(metrics['bleu_scores']),
        'average_meteor': sum(metrics['meteor_scores']) / len(metrics['meteor_scores']),
        'average_gleu': sum(metrics['gleu_scores']) / len(metrics['gleu_scores']),
        'average_errant': sum(metrics['errant_scores']) / len(metrics['errant_scores']),
        'hallucinations': {
            'count': metrics['hallucination_count'],
            'percentage': (metrics['hallucination_count'] / total_samples) * 100
        }
    }
    
    # Print hallucination summary
    print(f"\nHallucination Summary:")
    print(f"Total hallucinations: {metrics['hallucination_count']}/{total_samples} ({avg_metrics['hallucinations']['percentage']:.2f}%)")
    
    # Save results and metrics as JSON
    output_data = {
        'results': readable_results,
        'overall_metrics': avg_metrics
    }
    
    with open(pkl_file.replace('.pkl', '_evaluated.json'), 'w') as f:
        json.dump(output_data, f, indent=2)

    return output_data

def calculate_gleu(reference, hypothesis):
    """
    Calculate GLEU score between reference and hypothesis
    Args:
        reference: list of tokens from reference sentence
        hypothesis: list of tokens from hypothesis sentence
    Returns:
        float: GLEU score
    """
    return sentence_gleu([reference], hypothesis)

def calculate_errant_aligned(ref_eng, hyp_eng, ref_es, hyp_es):
    """
    Calculate ERRANT score for grammar corrections with pre-aligned language segments
    """
    try:
        scores = []
        print("\nCalculating ERRANT scores:")
        print(f"Reference English: {ref_eng}")
        print(f"Hypothesis English: {hyp_eng}")
        print(f"Reference Spanish: {ref_es}") 
        print(f"Hypothesis Spanish: {hyp_es}")
        
        # English evaluation
        if ref_eng != hyp_eng:
            print("\nProcessing English evaluation...")
            ref_text_eng = ' '.join(ref_eng)
            hyp_text_eng = ' '.join(hyp_eng)
            
            # English ERRANT evaluation
            annotator = errant.load('en')
            ref_doc_eng = nlp_eng(ref_text_eng)
            hyp_doc_eng = nlp_eng(hyp_text_eng)
            ref_orig_eng = annotator.parse(str(ref_doc_eng))
            hyp_orig_eng = annotator.parse(str(hyp_doc_eng))
            alignment_eng = annotator.align(ref_orig_eng, hyp_orig_eng)
            edits_eng = annotator.merge(alignment_eng)
            
            score_eng = 1.0 - (len(edits_eng) / len(ref_eng))
            print(f"Number of English edits: {len(edits_eng)}")
            print(f"English score: {score_eng}")
            scores.append(max(0.0, score_eng))
        
        # Spanish evaluation
        if ref_es != hyp_es:
            print("\nProcessing Spanish evaluation...")
            ref_text_es = ' '.join(ref_es)
            hyp_text_es = ' '.join(hyp_es)
            
            # Basic Spanish difference evaluation using spaCy
            doc_ref_es = nlp_es(ref_text_es)
            doc_hyp_es = nlp_es(hyp_text_es)
            
            diff_count = 0
            for token_ref, token_hyp in zip(doc_ref_es, doc_hyp_es):
                if token_ref.text != token_hyp.text:
                    print(f"Spanish difference found: '{token_ref.text}' vs '{token_hyp.text}'")
                    diff_count += 1
            
            score_es = 1.0 - (diff_count / len(ref_es))
            print(f"Number of Spanish differences: {diff_count}")
            print(f"Spanish score: {score_es}")
            scores.append(max(0.0, score_es))
        
        final_score = sum(scores) / len(scores) if scores else 1.0
        print(f"\nFinal ERRANT score: {final_score}")
        return final_score
            
    except Exception as e:
        print(f"Error calculating bilingual ERRANT score: {e}")
        return 0.0
    
def detect_language(text_segment):
    """
    Detect language using langdetect with fallback
    Returns: 'en' for English, 'es' for Spanish
    """

    # Join tokens into text for detection
    text = ' '.join(text_segment)
    lang = detect(text)
    return lang


# Usage
results = convert_to_readable(
    'analysis/pkl/test_triple_expert_gt_data.pkl',
    'meta-llama/Llama-2-13b-hf',
    'meta-llama/Llama-2-7b-hf'
)













# tests

def test_calculate_errant_aligned():
    # Test case 1: Minor variation in Spanish translation
    reference = ['He', "doesn't", 'have', 'any', 'idea', 'what', 'to', 'do', '-', 
                 'Él', 'no', 'tiene', 'idea', 'de', 'qué', 'hacer']
    hypothesis = ['She', "doesn't", 'have', 'any', 'idea', 'what', 'to', 'do', '-',
                 'No', 'tiene', 'ni', 'idea', 'de', 'qué', 'hacer']
    
    # Split into language pairs
    ref_sep_idx = reference.index('-')
    hyp_sep_idx = hypothesis.index('-')
    
    
    ref_eng = reference[:ref_sep_idx]
    ref_es = reference[ref_sep_idx + 1:]
    hyp_eng = hypothesis[:hyp_sep_idx]
    hyp_es = hypothesis[hyp_sep_idx + 1:]

    ref_eng = ['He', 'doesn\'t', 'have', 'any', 'idea', 'what', 'to', 'do']
    hyp_eng = ['She', 'doesn\'t', 'have', 'any', 'idea', 'what', 'to', 'do']
    ref_es = ['Él', 'no', 'tiene', 'idea', 'de', 'qué', 'hacer']
    hyp_es = ['No', 'tiene', 'ni', 'idea', 'de', 'qué', 'hacer']
    
    score = calculate_errant_aligned(ref_eng, hyp_eng, ref_es, hyp_es)
    
    print("\nTest Results:")
    print(f"Reference English: {' '.join(ref_eng)}")
    print(f"Hypothesis English: {' '.join(hyp_eng)}")
    print(f"Reference Spanish: {' '.join(ref_es)}")
    print(f"Hypothesis Spanish: {' '.join(hyp_es)}")
    print(f"ERRANT Score: {score:.3f}")
    
    # The score should be relatively high as the differences are minor

def test_english_errant_evaluation():
    """Test ERRANT evaluation specifically for English text with different cases"""
    
    test_cases = [
        # Case 1: Identical sentences
        {
            'ref': ['He', "doesn't", 'have', 'any', 'idea', 'what', 'to', 'do'],
            'hyp': ['He', "doesn't", 'have', 'any', 'idea', 'what', 'to', 'do'],
            'expected_score': 1.0,
            'description': "Identical sentences should have perfect score"
        },
        # Case 2: One word different
        {
            'ref': ['He', "doesn't", 'have', 'any', 'idea', 'what', 'to', 'do'],
            'hyp': ['She', "doesn't", 'have', 'any', 'idea', 'what', 'to', 'do'],
            'expected_score': 0.875,  # 7/8 words correct
            'description': "One word difference"
        },
        # Case 3: Multiple differences
        {
            'ref': ['The', 'cat', 'sits', 'on', 'the', 'mat'],
            'hyp': ['A', 'cat', 'is', 'on', 'a', 'mat'],
            'expected_score': 0.5,  # 3/6 words different
            'description': "Multiple word differences"
        }
    ]
    
    for case in test_cases:
        print(f"\nTesting: {case['description']}")
        print(f"Reference: {' '.join(case['ref'])}")
        print(f"Hypothesis: {' '.join(case['hyp'])}")
        
        # Prepare ERRANT evaluation
        ref_text = ' '.join(case['ref'])
        hyp_text = ' '.join(case['hyp'])
        
        annotator = errant.load('en')
        ref_doc = nlp_eng(ref_text)
        hyp_doc = nlp_eng(hyp_text)

        print('sdkjhf')
        print(str(ref_doc).split())
        
        ref_orig = annotator.parse(str(ref_doc))
        hyp_orig = annotator.parse(str(hyp_doc))
        
        alignment = annotator.align(ref_orig, hyp_orig)
        edits = annotator.merge(alignment)
        
        score = 1.0 - (len(edits) / len(case['ref']))
        score = max(0.0, score)
        
        print(f"Score: {score:.3f}")
        print(f"Expected: {case['expected_score']:.3f}")
        print(f"Number of edits: {len(edits)}")
        
        # Assert with some tolerance for floating point comparison
        assert abs(score - case['expected_score']) < 0.001, \
            f"Expected score {case['expected_score']}, but got {score}"

# test_english_errant_evaluation()

def test_spanish_evaluation():
    """Test Spanish evaluation with different cases using spaCy comparison"""
    
    test_cases = [
        # Case 1: Identical sentences
        {
            'ref': ['El', 'gato', 'está', 'en', 'la', 'mesa'],
            'hyp': ['El', 'gato', 'está', 'en', 'la', 'mesa'],
            'expected_score': 1.0,
            'description': "Identical sentences should have perfect score"
        },
        # Case 2: One word different
        {
            'ref': ['El', 'gato', 'está', 'sobre', 'la', 'mesa'],
            'hyp': ['El', 'gato', 'está', 'en', 'la', 'mesa'],
            'expected_score': 0.833,  # 5/6 words correct
            'description': "One word difference (preposition)"
        },
        # Case 3: Multiple differences
        {
            'ref': ['Él', 'no', 'tiene', 'idea', 'de', 'qué', 'hacer'],
            'hyp': ['No', 'tiene', 'ni', 'idea', 'de', 'qué', 'hacer'],
            'expected_score': 0.714,  # 5/7 words correct
            'description': "Multiple word differences"
        }
    ]
    
    for case in test_cases:
        # Load Spanish spaCy model
        nlp_spa = spacy.load('es_core_news_sm')

        # Load ERRANT for Spanish
        annotator = errant.load('es')

        print(f"\nTesting: {case['description']}")
        print(f"Reference: {' '.join(case['ref'])}")
        print(f"Hypothesis: {' '.join(case['hyp'])}")

        # Prepare ERRANT evaluation
        ref_text = ' '.join(case['ref'])
        hyp_text = ' '.join(case['hyp'])

        ref_doc = nlp_spa(ref_text)
        hyp_doc = nlp_spa(hyp_text)

        print('sdkjhf')
        print(str(ref_doc).split())

        ref_orig = annotator.parse(str(ref_doc))
        hyp_orig = annotator.parse(str(hyp_doc))

        alignment = annotator.align(ref_orig, hyp_orig)
        edits = annotator.merge(alignment)

        score = 1.0 - (len(edits) / len(case['ref']))
        score = max(0.0, score)

        print(f"Score: {score:.3f}")
        print(f"Expected: {case['expected_score']:.3f}")
        print(f"Number of edits: {len(edits)}")

# test_spanish_evaluation()
# print(detect_language(['Mi', 'hermana', 'va', 'a', 'estudiar', 'en', 'el', 'extranjero']))
# f = [['The', 'books', 'were', 'on', 'the', 'table', '-', 'Los', 'libros', 'estaban', 'sobre', 'la', 'mesa'], ['The', 'books', 'were', 'on', 'the', 'table', '-', 'Los', 'libros', 'estaban', 'en', 'la', 'mesa']]
# e = [['Mi', 'hermana', 'va', 'a', 'estudiar', 'en', 'el', 'extranjero', '-', 'My', 'sister', 'is', 'going', 'to', 'study', 'abroad'],    ['My', 'sister', 'is', 'going', 'to', 'study', 'abroad', '-', 'Mi', 'hermana', 'va', 'a', 'estudiar', 'en', 'el', 'extranjero']]
# print(calculate_errant(['Los', 'libros', 'estaban', 'sobre', 'la', 'mesa'], ['Los', 'libros', 'estaban', 'en', 'la', 'mesa']))
# print(calculate_errant(e[0], e[1]))


