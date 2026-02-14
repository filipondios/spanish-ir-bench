# spanish-ir-bench

## About

This project implements and evaluates an Information Retrieval (IR) system in Spanish using the [MessIRve](https://huggingface.co/datasets/spanish-ir/messirve) dataset and a subset of Spanish Wikipedia, [eswiki_20240401_corpus](https://huggingface.co/datasets/spanish-ir/eswiki_20240401_corpus).

Two retrieval approaches are compared:

- TF-IDF (sparse lexical retrieval)
- Sentence-Transformer embeddings (dense semantic retrieval)

The goal is to analyze performance differences between traditional term-based methods and modern semantic models in a large-scale Spanish IR setting.

## Project Structure

The notebook follows this pipeline:

- Data Loading
    - Load train/test splits and corpus
    - Convert to pandas
    - Remove unused columns
    - Filter corpus to relevant documents

- Preproessing
  - Heavy preprocessing for TF-IDF (stopword removal, normalization)
  - Light preprocessing for embeddings (lowercasing + punctuation removal)
  - Parallelized with joblib

- TF-IDF Retrieval
    - TfidfVectorizer with:
        - unigrams + bigrams
        - max 50,000 features
        - sublinear TF scaling
    - Cosine similarity
    - Top-100 ranking per query

- Embedding-Based Retrieval
    - Model: [sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2)
    - 384-dimensional dense vectors
    - Normalized embeddings
    - Dot product for fast similarity computation
    - Top-100 ranking per query

- Evaluation
    - Precision@k
    - Recall@k
    - nDCG@k
    - Evaluated on test split only
 
## Results
### TF-IDF
| Metric      | Value |
| ----------- | ----- |
| Precision@1 | 0.100 |
| Recall@10   | 0.343 |
| nDCG@10     | 0.209 |

### Embeddings
| Metric      | Value |
| ----------- | ----- |
| Precision@1 | 0.175 |
| Recall@10   | 0.419 |
| nDCG@10     | 0.290 |

## Key Findings

- Embeddings significantly outperform TF-IDF across all metrics.
- Precision@1 improves from 10% to 17.5%.
- Recall@10 increases from 34.3% to 41.9%.
- nDCG@10 improves substantially, indicating better ranking quality.

These results confirm that dense semantic representations capture query-document similarity more effectively than purely lexical term matching, especially in scenarios with lexical variation.

## Possible Improvements

Future work could include:

- Hybrid sparse + dense retrieval
- More advanced embedding models
- Cross-encoder re-ranking
- Fine-tuning on the Messirve dataset
- Spanish-specific transformer models
- Approximate nearest neighbor search for scalability

## Motivation

This project serves as an experimental benchmark comparing sparse and dense retrieval methods in Spanish. It is intended as an educational and exploratory implementation rather than a production-grade search engine.
