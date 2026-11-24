# Short-post sentiment analysis

A reproducible Python experiment for classifying short posts as negative, neutral or positive. It compares a TF-IDF logistic regression model against a majority-class baseline on the same held-out split.

## Interview demo and measured benchmark

Install the package, get the publisher’s fixed splits, then train and launch the browser interface.

```sh
python -m pip install .
git clone https://github.com/cardiffnlp/tweeteval.git data/tweeteval
git -C data/tweeteval checkout 4fbd22cd78421f05b1ecdb4fc5725bc7a7bd8f66
python -m post_sentiment.benchmark data/tweeteval/datasets/sentiment --output runs/tweeteval
python -m post_sentiment.web --model runs/tweeteval/model.joblib
```

Open http://127.0.0.1:8001. The browser shows model scores, the attached evaluation and per-class errors. It runs locally. Starting without `--model` uses the synthetic smoke test.

The linear model achieved **62.27% macro recall**, **60.85% macro-F1** and **60.92% accuracy** on 12,284 official test posts. The majority baseline achieved **33.33% macro recall**. We fit word and character features on the official training set, select regularisation on the validation split and score the locked model on the test split. Macro recall is the benchmark’s primary sentiment metric. The audit found zero exact normalised training-text matches in validation or test. This does not rule out paraphrases, similar topics, or temporal overlap. [Measured results and input hashes](docs/tweeteval-results.json)

The benchmark evaluates an English short-post classifier and its scores are uncalibrated. Treat it as a lightweight baseline on this historical dataset. The benchmark uses TweetEval by Barbieri, Camacho-Collados, Neves and Espinosa-Anke (2020) and the underlying SemEval sentiment task. See upstream terms before downloading or reusing any posts. The project repository includes aggregate measurements. [TweetEval sources and terms](https://github.com/cardiffnlp/tweeteval/tree/4fbd22cd78421f05b1ecdb4fc5725bc7a7bd8f66)

## Run it locally

Requires Python 3.10 or later.

```sh
python -m pip install .
post-sentiment train examples/labelled_posts.csv --output runs/demo
post-sentiment predict runs/demo/model.joblib "The support team was helpful"
python -m unittest discover -s tests -v
```

The demo uses 30 hand-written synthetic posts. It exercises training, evaluation and model reload. Eight held-out examples are too few to estimate real-world performance.

## Use another dataset

Supply a UTF-8 CSV with `text,label` columns and at least five distinct examples per class. The pipeline masks links and mentions, collapses exact normalised duplicates and rejects conflicts. It makes a seeded stratified split before fitting word and character TF-IDF. The output records macro-F1, accuracy, confusion matrices, class supports and the input file's SHA-256. Use a fresh output directory for each run.

Collect posts under terms that allow your intended use. For a realistic benchmark, split by author, conversation or time where that metadata exists. Exact deduplication does not catch paraphrases, repeated topics or shared authors. Sarcasm, domain shifts and annotator disagreement can reduce accuracy. Prediction scores are model outputs and have not been calibrated. Load only model artefacts you created locally because joblib loads executable Python objects.

Khush Bhavin Maru · Adelaide University MAAIM portfolio. Codex assisted with implementation and documentation.
