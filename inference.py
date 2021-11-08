import sys
import json
import pickle
import time

from tqdm import tqdm
from transformers import *
from argparse import ArgumentParser
from os.path import join
from genre.trie import Trie
from genre.hf_model import GENRE
from wikimapper import WikiMapper


# Constants
BASE_RESOURCES_DIR = 'resources'
AQUAINT = 'aquaint'
RSS500 = 'rss500'
ISTEX = 'istex'
TWEETKI = 'tweetki'
WIKIPEDIAEL = 'wikipediael'
DATASETS = [AQUAINT, RSS500, ISTEX, TWEETKI, WIKIPEDIAEL]

# Class Entity Mention
class EntityMention:
    def __init__(
        self, docid, mention_text, context_left, context_right, kbid = None
    ):
        self.docid = docid
        self.mention_text = mention_text
        self.context_left = context_left
        self.context_right = context_right
        self.kbid = kbid

    def __str__(self):
        return 'DocID = {} | Mention Text = {} | Context Left = {} | Context Right = {} | KBID = {}'.format(
            self.docid, self.mention_text, self.context_left,
            self.context_right, self.kbid
        )

# Main code
if __name__ == "__main__":
    # Parse argument
    parser = ArgumentParser()
    parser.add_argument('--trained_model_path', default='/shared/nas/data/m1/tuanml/genre_models')
    parser.add_argument('--dataset', default=RSS500, choices=DATASETS)
    args = parser.parse_args()

    # input fp and output fp
    args.in_fp = join(BASE_RESOURCES_DIR, args.dataset, 'test.jsonl')
    args.out_fp = join(BASE_RESOURCES_DIR, args.dataset, 'pred_test.jsonl')

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained('roberta-base', add_prefix_space=True)

    # Load WikiMapper
    print('Loading WikiMapper')
    start_time = time.time()
    wiki_mapper = WikiMapper(join(args.trained_model_path, 'index_enwiki-latest.db'))
    print('Took {} seconds'.format(time.time() - start_time))

    # Load the prefix tree (trie)
    print('Loading the prefix tree (trie)')
    start_time = time.time()
    with open(join(args.trained_model_path, 'kilt_titles_trie_dict.pkl'), 'rb') as f:
        trie = Trie.load_from_dict(pickle.load(f))
    print('Took {} seconds'.format(time.time() - start_time))

    # for pytorch/fairseq
    print('Loading the model')
    start_time = time.time()
    #model = GENRE.from_pretrained(join(args.trained_model_path, 'fairseq_entity_disambiguation_aidayago')).eval()
    model = GENRE.from_pretrained(join(args.trained_model_path, 'hf_entity_disambiguation_aidayago')).eval()
    model.cuda()
    print('Took {} seconds'.format(time.time() - start_time))

    # Read data from args.in_fp
    all_datas, entity_mentions = [], []
    with open(args.in_fp, 'r') as f:
        for line in f:
            data = json.loads(line)
            all_datas.append(data)
            input_tokens = data['input_tokens']
            input_ids = tokenizer.convert_tokens_to_ids(input_tokens)
            for start, end, kbid in data['anchors']:
                # Left Context
                left_ids = input_ids[:start]
                left_tokens = tokenizer.convert_ids_to_tokens(left_ids, skip_special_tokens=True)
                left_context = tokenizer.convert_tokens_to_string(left_tokens)
                # Mention Text
                mention_ids = input_ids[start:end+1]
                mention_tokens = tokenizer.convert_ids_to_tokens(mention_ids, skip_special_tokens=True)
                mention_text = tokenizer.convert_tokens_to_string(mention_tokens)
                # Right Context
                right_ids = input_ids[end+1:]
                right_tokens = tokenizer.convert_ids_to_tokens(right_ids, skip_special_tokens=True)
                right_context = tokenizer.convert_tokens_to_string(right_tokens)
                # Update entity_mentions
                entity_mentions.append(EntityMention(data['id'], mention_text, left_context, right_context, kbid))

    # Inference
    doc2candidates, doc2candidatescores = {}, {}
    recall_count, total_count = 0, 0
    progress_bar = tqdm(entity_mentions)
    for em in progress_bar:
        sentences = []
        sentences.append(em.context_left.strip() + ' [START_ENT] ' + \
                         em.mention_text.strip() + ' [END_ENT] ' + \
                         em.context_right.strip())                        
        result = model.sample(
            sentences,
            prefix_allowed_tokens_fn=lambda batch_id, sent: trie.get(sent.tolist()),
        )[0][0]
        wikidata_id = wiki_mapper.title_to_id(result['text'].replace(' ', '_'))
        # Update doc2candidates and doc2candidatescores
        cur_doc_id = em.docid
        if cur_doc_id not in doc2candidates:
            doc2candidates[cur_doc_id] = []
            doc2candidatescores[cur_doc_id] = []
        doc2candidates[cur_doc_id].append([wikidata_id])
        doc2candidatescores[cur_doc_id].append([1.0])
        # Update progress_bar
        if wikidata_id == em.kbid: recall_count += 1
        total_count += 1
        progress_bar.set_description(f'R@1: {recall_count / total_count}')
        

    # Write to args.out_fp
    with open(args.out_fp, 'w') as f:
        for data in all_datas:
            doc_id = data['id']
            data['candidates'] = doc2candidates.get(doc_id, [])
            data['candidates_scores'] = doc2candidatescores.get(doc_id, [])
            f.write(json.dumps(data) + '\n')

 
