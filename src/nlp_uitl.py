import MeCab


def tokenize(text):
    wakati = MeCab.Tagger("-O wakati")
    wakati.parse("")
    return wakati.parse(text).strip().split(" ")


def extract_keyphrase_candidates(text):
    tagger = MeCab.Tagger()
    tagger.parse("")

    node = tagger.parseToNode(text)

    keyphrase_candidates = []
    phrase = []
    phrase_noun = []
    is_adj_candidate = False
    is_multinoun_candidate = False

    while node:
        # adjectives + nouns
        if node.feature.startswith('形容詞'):
            is_adj_candidate = True
            phrase.append(node.surface)
        if node.feature.startswith("名詞") and is_adj_candidate:
            phrase.append(node.surface)
        elif len(phrase) >= 2:
            keyphrase_candidates.append(phrase)

            is_adj_candidate = False
            phrase = []

        # multiple nouns
        if node.feature.startswith("名詞"):
            phrase_noun.append(node.surface)
            is_multinoun_candidate = True
        elif len(phrase_noun) >= 2:
            keyphrase_candidates.append(phrase_noun)

            is_multinoun_candidate = False
            phrase_noun = []
        else:
            is_multinoun_candidate = False
            phrase_noun = []

        node = node.next

    return keyphrase_candidates
