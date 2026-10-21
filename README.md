# शब्दसंपत्ती: Marathi Vocabulary Enrichment System

A Streamlit web application that identifies simple and repetitive words in Marathi essays and suggests richer alternatives. Students review each suggestion and choose which replacements to apply.

**Project type:** Academic Mini Project

## Features

- **Three-tier vocabulary suggestions:**
  1. Exact lookup using custom simple-to-enriched sentence pairs.
  2. Marathi word–synonym dictionary lookup.
  3. fastText nearest neighbours filtered using a trained RBF-SVM classifier.
- **Context-aware suggestions** using word-pair rules and optional IndicSBERT ranking.
- **Per-occurrence word replacement**, allowing different replacements for the same word.
- **Grammar helpers** for gender agreement and adverb forms.
- **Essay analysis** with repeated-word charts, vocabulary richness scores, and before/after comparisons.
- **Model Performance dashboard** with SVM evaluation and an architecture diagram.

## Technologies Used

- Python
- Streamlit
- Pandas
- scikit-learn
- fastText word embeddings
- Optional IndicSBERT sentence embeddings

## Model Performance

| Metric | Baseline | Final Model |
|---|---:|---:|
| Accuracy | 76.2% | 80.8% |
| Precision | 95.3% | 89.0% |
| Recall | 70.3% | 83.5% |
| F1-score | 80.9% | 86.2% |

The reported evaluation uses held-out dictionary words with an 80/20 split. Five-fold cross-validated F1-score: 85.3% ± 0.6%.

**Note:** Randomly selected negative examples may make evaluation scores optimistic compared with real-world usage.

## Project Structure

- `app.py` — Streamlit interface and core application logic
- `marathi_vocab_lookup.json` — Sentence-based vocabulary lookup
- `marathi_dictionary.csv` — Marathi words and synonyms
- `new_sentence_pairs.csv` — Reviewed sentence pairs
- `expand_dataset.py` — Dataset cleaning and lookup expansion
- `improve_accuracy.py` — SVM feature and tuning experiments
- `compare_models.py` — Model comparison
- `clean_lookup.py` — Lookup data cleaning
- `.streamlit/config.toml` — Streamlit theme configuration

## Installation and Setup

1. Clone this repository:

   ```bash
   git clone https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git
   cd YOUR-REPOSITORY
   ```

2. Create and activate a virtual environment:

   ```bash
   python -m venv venv
   ```

   On Windows:

   ```bash
   venv\Scripts\activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Download the Marathi fastText vectors, `cc.mr.300.vec.gz`, from the [fastText website](https://fasttext.cc/docs/en/crawl-vectors.html), extract the file, and place `cc.mr.300.vec` in the project root.

5. Start the application:

   ```bash
   streamlit run app.py
   ```

### Optional Context Ranking

Install the optional dependency:

```bash
pip install sentence-transformers
```

Enable Context Ranking in the application sidebar if supported by the installed configuration.

## Limitations

- Marathi word endings may cause grammatical mismatches.
- Lookup suggestions depend on the available dataset and exact word forms.
- Rule-based corrections cover only selected cases.
- Random negative examples may not fully represent difficult real-world synonym classification.
- Suggested replacements are advisory and should be reviewed by the user.

## References

- Bojanowski et al., *Enriching Word Vectors with Subword Information*, TACL, 2017.
- Grave et al., *Learning Word Vectors for 157 Languages*, LREC, 2018.
- L3Cube-Pune, IndicSBERT sentence embeddings.
- Pedregosa et al., *Scikit-learn: Machine Learning in Python*, JMLR, 2011.
- Hugging Face dataset: `Process-Venue/Hindi-Marathi-Synonyms`.

## Authors

Add the names of the project team members, college, and project guide here.

## License

The project code is intended to use the MIT License. Add a `LICENSE` file before claiming the repository is officially licensed. Dataset and model licenses remain subject to their original terms.
