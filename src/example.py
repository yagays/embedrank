from gensim.models.doc2vec import Doc2Vec

from embedrank import EmbedRank
from nlp_uitl import tokenize

model = Doc2Vec.load("model/jawiki.doc2vec.dbow300d.model")
embedrank = EmbedRank(model=model, tokenize=tokenize)

text = """バーレーンの首都マナマ(マナーマとも)で現在開催されている
ユネスコ(国際連合教育科学文化機関)の第42回世界遺産委員会は日本の推薦していた
「長崎と天草地方の潜伏キリシタン関連遺産」 (長崎県、熊本県)を30日、
世界遺産に登録することを決定した。文化庁が同日発表した。
日本国内の文化財の世界遺産登録は昨年に登録された福岡県の
「『神宿る島』宗像・沖ノ島と関連遺産群」に次いで18件目。
2013年の「富士山-信仰の対象と芸術の源泉」の文化遺産登録から6年連続となった。"""

print(embedrank.extract_keyword(text))
