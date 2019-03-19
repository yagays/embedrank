# EmbedRank

Python Implementaion of "[Simple Unsupervised Keyphrase Extraction using Sentence Embeddings](https://arxiv.org/abs/1801.04470)"

## Usage

EmbedRank requires pretrained document embeddings (now doc2vec supported). Please see [my blog ](https://yag-ays.github.io/project/pretrained_doc2vec_wikipedia/) for using pretrained Japanese doc2vec models.

```py
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
```

```py
In []: embedrank.extract_keyword(text)
[('世界遺産登録', 0.61837685), ('(長崎県', 0.517046), ('ユネスコ(国際連合教育科学文化機関)', 0.5726031), ('潜伏キリシタン関連遺産', 0.544827), ('首都マナマ(マナーマ', 0.4898381)]

```

(Source: [潜伏キリシタン関連遺産、世界遺産登録 \- ウィキニュース](https://ja.wikinews.org/wiki/%E6%BD%9C%E4%BC%8F%E3%82%AD%E3%83%AA%E3%82%B7%E3%82%BF%E3%83%B3%E9%96%A2%E9%80%A3%E9%81%BA%E7%94%A3%E3%80%81%E4%B8%96%E7%95%8C%E9%81%BA%E7%94%A3%E7%99%BB%E9%8C%B2))

## Docker

Set the extracted doc2vec model in `model/` directory and run the following commands.

```
$ docker build -t embedrank .
$ docker run --rm -p 8080:8080 --memory 7g -it embedrank
```

```
$ curl -XPOST "localhost:8080/embedrank" --data-urlencode text='バーレーンの首都マナマ(マナーマとも)で現在開催されている
                                            ユネスコ(国際連合教育科学文化機関)の第42回世界遺産委員会は日本の推薦していた
                                            「長崎と天草地方の潜伏キリシタン関連遺産」 (長崎県、熊本県)を30日、
                                            世界遺産に登録することを決定した。文化庁が同日発表した。
                                            日本国内の文化財の世界遺産登録は昨年に登録された福岡県の
                                            「『神宿る島』宗像・沖ノ島と関連遺産群」に次いで18件目。
                                            2013年の「富士山-信仰の対象と芸術の源泉」の文化遺産登録から6年連続となった。'
                                            -d 'num_keywords=3'

{
  "keywords": [
    {
      "keyword": "世界遺産登録",
      "score": "0.58336747"
    },
    {
      "keyword": "天草地方",
      "score": "0.52296615"
    },
    {
      "keyword": "首都マナマ(マナーマ",
      "score": "0.5126816"
    }
  ]
}                                            
```

Caution:

- You need to allocate total memory size more than 7GB.
- Container size is very large (7.38GB)
