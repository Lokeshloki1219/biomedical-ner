# Biomedical NER: Extracting Chemicals and Diseases from Clinical Literature

![Python](https://img.shields.io/badge/python-3.10-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-2.5-ee4c2c) ![Transformers](https://img.shields.io/badge/🤗%20Transformers-4.57-yellow)

This project compares four NER models that tag **chemical** and **disease** mentions in PubMed abstracts. The four models fall into two model families. The comparison runs on the BC5CDR benchmark and is scored at the span level. I also wrote a reconciliation step so that subword-tokenizing transformers and word-level BiLSTMs are scored on equal terms.

> *"**Naloxone** reverses the antihypertensive effect of **clonidine**."* → `Naloxone` = Chemical, `clonidine` = Chemical
>
> *"…the decrease in blood pressure in spontaneously **hypertensive** rats…"* → `hypertensive` = Disease

## Why it matters

Biomedical literature grows faster than anyone can read it. Downstream work such as drug discovery, adverse-event detection and patient-trial matching first needs the entities pulled out of free text. This is hard for three reasons:
- the vocabulary is specialized and keeps changing
- abbreviations are everywhere
- entities often span several words, such as *"non-small cell lung cancer"*

The project asks two questions:
- How much does contextual modeling matter compared with static word embeddings?
- Does domain-specific pre-training beat a stronger general-purpose architecture?

## Results

Test set: 5,865 sentences and 9,809 gold entities (5,385 chemical, 4,424 disease). The metric is span-level micro F1 from `seqeval`.

| Track | Model | Representation | Precision | Recall | **F1** |
|---|---|---|---|---|---|
| B | **DeBERTa-v3-base** | general-domain transformer | 87.6 | 90.9 | **89.2** |
| A | BioBERT v1.1 | biomedical transformer | — | — | 88.4 |
| C1 | BiLSTM-CRF | BioWordVec (200d, PubMed + MIMIC-III) | — | — | 73.8 |
| C2 | BiLSTM-CRF | FastText wiki-news (300d) | — | — | 73.3 |

F1 broken down by entity type:

| Model | Chemical F1 | Disease F1 |
|---|---|---|
| DeBERTa-v3 | 92.4 | 85.4 |
| BioBERT | 92.5 | 83.5 |
| BiLSTM-CRF + BioWordVec | 73.7 | 73.8 |
| BiLSTM-CRF + FastText | 74.1 | 72.3 |

### Key findings

1. **Contextual embeddings beat static embeddings by about 15 F1 points.** Both transformers resolve ambiguous and multi-word entities that the BiLSTM-CRF models miss.
2. **A better general-domain architecture slightly beat domain pre-training.** DeBERTa's disentangled attention outscored BioBERT by 0.8 F1, even though BioBERT was pre-trained on 4.5 billion words of biomedical text.
3. **Diseases are harder to tag than chemicals.** The transformers score about 7 to 9 F1 points lower on diseases. Disease mentions mix symptoms, syndromes and general clinical terms, while chemical names follow systematic patterns.
4. **The two embeddings have different strengths.** FastText's character n-grams catch drug-name suffixes such as `-idine` and `-acin`. For example, it tags *famotidine* correctly where BioWordVec misses it, and it covers 85.7% of the vocabulary against BioWordVec's 74.0%. BioWordVec's clinical training data helps more on disease terms.
5. **The two model families fail differently.** The transformers tend to draw entity boundaries too wide, for example tagging *"stress ulcers"* when only *ulcers* is gold. The BiLSTM-CRF models are conservative, with precision higher than recall.

## How it works

![Multi-track NER workflow](assets/workflow.jpg)

**Track A / B: fine-tuned transformers**
- **Subword label alignment.** The first subword of each word keeps the word's label. The continuation subwords of a `B-` word become `I-`, so *f ##amo ##ti ##dine* becomes `B I I I`. Special tokens get `-100` so the loss ignores them.
- **Training setup.** Fine-tuning uses the Hugging Face `Trainer` with these settings:
  - learning rate 2e-5 with warmup and linear decay, batch size 16, weight decay 0.01
  - mixed precision (fp16) when a GPU is available
  - early stopping on validation F1 with a patience of 3

**Track C1 / C2: BiLSTM-CRF**
- **Architecture:**
  - a lowercase vocabulary, with out-of-vocabulary words mapped to `<UNK>`
  - pre-trained embeddings that are fine-tuned during training
  - a bidirectional LSTM with a hidden size of 256
  - a CRF layer (`pytorch-crf`) that enforces valid BIO transitions
- **CRF padding fix.** The CRF can't take `-100` labels, so padding positions get a valid tag index and are masked out of the loss.
- **Training setup.** Adam with learning rate 0.01, gradient clipping at 1.0, and early stopping on validation F1.

**Scoring: prioritized-vote reconciliation**
- The transformers predict one tag per subword, but gold labels are per word. The pipeline merges subword predictions back into one tag per word.
- If any subword of a word predicts a `B-` tag, that tag wins over `I-` tags, even when most subwords predict `I-`.
- After merging, all four models are scored by `seqeval` on identical word-level spans, so the comparison is fair.

**Analysis in the notebook:**
- learning curves for each track
- inference-time comparison
- McNemar significance tests between models
- traces of individual sentences through each model
- the best and worst predicted examples for error analysis

## Repository layout

```
biomedical_ner.ipynb   # end-to-end pipeline: data prep → 4 tracks → evaluation → analysis
dataset/               # BC5CDR, sentence-split, BIO tags (train / valid / test + label map)
assets/workflow.jpg    # pipeline diagram
docs/report.pdf        # full technical report with per-class tables and error analysis
requirements.txt
```

## Reproducing

Use **Python 3.10**, which is the version this was developed and tested on. A CUDA GPU is strongly recommended because full training takes hours on a CPU.

```bash
python -m venv .venv && .venv\Scripts\activate      # (Linux/macOS: source .venv/bin/activate)
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121   # GPU build; skip for CPU
pip install -r requirements.txt
```

1. **Transformer checkpoints.** `dmis-lab/biobert-base-cased-v1.1` and `microsoft/deberta-v3-base` download automatically from the Hugging Face Hub.
2. **Static embeddings for Tracks C1 and C2.** Download both files into `./embeddings/`. They are not committed because together they take up about 4 GB.
   - **BioWordVec:** get the *intrinsic* set from the [BioWordVec repo](https://github.com/ncbi-nlp/BioWordVec) ([Figshare](https://doi.org/10.6084/m9.figshare.6882647)) and save it as `embeddings/bio_embedding_intrinsic`. It is a 200-d binary word2vec file.
   - **FastText:** download [wiki-news-300d-1M.vec.zip](https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip) and unzip it to `embeddings/wiki-news-300d-1M.vec`.
3. **Run the notebook.** Run `biomedical_ner.ipynb` from top to bottom. Each track saves its best checkpoint, and the BiLSTM weights are saved as `Track_C*_best.pth`.

## Dataset

The data is the [BC5CDR corpus](https://huggingface.co/datasets/tner/bc5cdr) (BioCreative V Chemical Disease Relation; Li et al., 2016):
- 1,500 PubMed articles annotated by experts for chemicals and diseases
- split into sentences and tagged with five BIO labels: `O`, `B-/I-Chemical`, `B-/I-Disease`
- train / validation / test splits of 5,228 / 5,330 / 5,865 sentences

## Team

Built with Nagraj Venkatesham Vallakati and Swet Vimalkumar Patel.

## References

- Lee et al., *BioBERT: a pre-trained biomedical language representation model*, Bioinformatics 2020
- He et al., *DeBERTa: Decoding-enhanced BERT with Disentangled Attention*, ICLR 2021
- Huang et al., *Bidirectional LSTM-CRF Models for Sequence Tagging*, 2015
- Bojanowski et al., *Enriching Word Vectors with Subword Information*, TACL 2017
- Li et al., *BioCreative V CDR task corpus*, Database 2016
