
'''合并词典'''
# def get_vocab(filepath):
#     vocab_li = []
#     with open(filepath, 'r', encoding='utf-8') as f:
#         for line in f.readlines():
#             vocab_li.append(line.strip().split()[0])
#     print(len(vocab_li))
#     return vocab_li

# vocab = set(get_vocab('/code/mte/data/opus/enit/dict.en.txt') + get_vocab('/code/mte/data/opus/enit/dict.it.txt'))
# print(len(vocab))

# with open('/code/mte/data/opus/enit/vocab_enit.txt', 'w', encoding='utf-8') as f:
#     f.write('<unk>' + '\n')
#     f.write('<s>' + '\n')
#     f.write('</s>' + '\n')
#     for w in vocab:
#         f.write(w + '\n')

# exit()
'''语素分割'''
import json
from polyglot.text import Word
import enchant
from collections import Counter
from src.tokenization import load_vocab_file
from tqdm import tqdm


# print(enchant.list_languages())

def mor_func(word, lang='it'):
    mors = Word(word, language=lang).morphemes
    mors = list(mors)
    return mors


def word_check(w):
    it_dict = enchant.Dict("it")
    if it_dict.check(w):
        return True
    return False

def word_check_ait(w):
    it_dict = enchant.Dict("it")

    for d in it_dict.suggest(w):
        if d.startswith(w):
            return True

    return False



vocab_dict = load_vocab_file('./data/opus/enit/vocab_enit.txt')
inv_vocab_dict = {index: token for token, index in vocab_dict.items()}

wordId2mor = {}

for id, w in tqdm(inv_vocab_dict.items()):
    if id <= 2:
        wordId2mor[id] = [w]
        continue

    if w.endswith('@@'):
        lang = 'it' if word_check_ait(w[:-2]) else 'en'
        mors = mor_func(w[:-2], lang=lang) + ['@@']
    else:
        lang = 'it' if word_check(w) else 'en'
        mors = mor_func(w, lang=lang)

    if len(mors) > 3:
        mors = [mors[0], mors[1], "".join(mors[2:])]

    wordId2mor[id] = mors

mor_set = list(set([mor for mors in wordId2mor.values() for mor in mors]))
print(sorted(Counter([len(bpe_li) for w, bpe_li in wordId2mor.items()]).items()))
print(len(inv_vocab_dict))
print(len(mor_set))
print(len(wordId2mor))

bpe_mor_res = {'mor_set': mor_set, 'wordId2mor': wordId2mor}

json.dump(bpe_mor_res, open('./data/opus/enit/' + 'enit-jointed_morSeg_results.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=2)



