# Embeddings: how computers search by meaning

Code for the video **Embeddings explained: how computers search by meaning** on [Show Me the Query](https://www.youtube.com/@ShowMeTheQuery).

A keyword search for "cheap dark roast for my morning brew" returns *cheap flights to tokyo*. An embedding search returns *best espresso beans for a home coffee machine*, even though the two share no words.

## Run it

```bash
pip install numpy
python3 search_by_meaning.py                                   # the coffee query from the video
python3 search_by_meaning.py "my puppy is too noisy at night"  # the query where averaging breaks
python3 search_by_meaning.py "your own query"
```

The first run downloads the GloVe 6B 50-dimension word vectors (about 66 MB, public domain under PDDL) from the [gensim-data](https://github.com/RaRe-Technologies/gensim-data) release on GitHub, and caches them in `~/.cache/glove`. After that it runs in a few seconds.

## What it does

1. Loads 400,000 words, each with 50 numbers.
2. Compares a few word pairs with cosine similarity.
3. Runs a keyword search: shared words after dropping filler words.
4. Runs an embedding search: each sentence becomes the average of its word vectors, ranked by cosine similarity to the query.
5. Projects the sentence vectors to 2D with PCA, for the map in the video.

## What we got

From [`sample-output.txt`](sample-output.txt):

```
query: 'cheap dark roast for my morning brew'
keyword search: [(1, 'cheap flights to tokyo in spring')]
embedding search:
  0.85  best espresso beans for a home coffee machine
  0.73  stop a kitten from biting your hands
  ...
```

And where it breaks, from [`sample-output-puppy.txt`](sample-output-puppy.txt): for "my puppy is too noisy at night", the kitten document (0.81) edges out the dog-barking one (0.79). Averaging keeps the topic (pets) but blurs the details.

## Try this next

- Swap the averaging for a modern sentence embedding model, for example from [Sentence Transformers](https://www.sbert.net/), and rerun the puppy query.
- Add your own documents to `DOCS` and see where they land on the 2D map.
- Combine both: rank by embedding score, then boost documents that also match keywords exactly.

## Sources

- Pennington, Socher and Manning (2014), [GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/projects/glove/)
- [Sentence Transformers documentation](https://www.sbert.net/)
