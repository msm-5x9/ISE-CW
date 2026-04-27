# =============================================================================
# ISE Coursework — Bug Report Classification
# Baseline  : Naive Bayes + TF-IDF
# Proposed  : DistilBERT fine-tuning
# Evaluation: 30 runs, 70/30 stratified split, macro F1/Precision/Recall
# Statistics: Wilcoxon Rank-Sum, Vargha-Delaney A12 effect size
# Extras    : Histograms, results table, cross-project generalisation
#
# Acknowledgement:
#   Sections of this file marked [LAB] are adapted from the module lab
# =============================================================================

# ──────────────────────────────────────────────
# 1. IMPORTS
# ──────────────────────────────────────────────
import os
import re
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')          # headless — no display needed

from scipy.stats import ranksums

import nltk
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import (precision_score, recall_score,
                              f1_score, accuracy_score)

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (DistilBertTokenizer,
                          DistilBertForSequenceClassification,
                          get_linear_schedule_with_warmup)
from torch.optim import AdamW

# ──────────────────────────────────────────────
# 2. CONFIGURATION 
# ──────────────────────────────────────────────
PROJECTS   = ['tensorflow', 'pytorch', 'keras', 'incubator-mxnet', 'caffe']
REPEAT     = 30          # number of stochastic runs
TEST_SIZE  = 0.30        # 70/30 split

BERT_MODEL     = 'distilbert-base-uncased'
BERT_EPOCHS    = 3
BERT_MAX_LEN   = 256
BERT_BATCH     = 16
BERT_LR        = 2e-5

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

OUTPUT_DIR = 'results'
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, 'plots'), exist_ok=True)

# ──────────────────────────────────────────────
# 3. TEXT PRE-PROCESSING
# [LAB] All four functions below (remove_html, remove_emoji,
# remove_stopwords, clean_str) and the preprocess() pipeline are
# taken directly from the module lab solution (Dr. Tao Chen,
# br_classification.py). They are standard NLP preprocessing steps
# required to reproduce the baseline under identical conditions.
# ──────────────────────────────────────────────
STOP_WORDS = set(stopwords.words('english'))

def remove_html(text):
    # [LAB] Adapted from lab solution — removes HTML tags via regex
    return re.compile(r'<.*?>').sub('', text)

def remove_emoji(text):
    # [LAB] Adapted from lab solution — strips Unicode emoji ranges
    pattern = re.compile(
        u"[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF"
        u"\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF"
        u"\U00002702-\U000027B0\U000024C2-\U0001F251]+",
        flags=re.UNICODE)
    return pattern.sub('', text)

def remove_stopwords(text):
    # [LAB] Adapted from lab solution — filters NLTK English stopwords
    return " ".join(w for w in str(text).split() if w not in STOP_WORDS)

def clean_str(s):
    # [LAB] Adapted from lab solution — normalises punctuation and casing
    s = re.sub(r"[^A-Za-z0-9(),.!?\'\`]", " ", s)
    s = re.sub(r"\'s",   " 's",  s)
    s = re.sub(r"\'ve",  " 've", s)
    s = re.sub(r"\)",    " ) ",  s)
    s = re.sub(r"\?",    " ? ",  s)
    s = re.sub(r"\s{2,}", " ",   s)
    s = re.sub(r"[\\\'\"]+", "",  s)
    return s.strip().lower()

def preprocess(text):
    # [LAB] Pipeline order taken from lab solution
    text = remove_html(str(text))
    text = remove_emoji(text)
    text = remove_stopwords(text)
    text = clean_str(text)
    return text

# ──────────────────────────────────────────────
# 4. DATA LOADING
# ──────────────────────────────────────────────
def load_project(name):
    """Read <name>.csv, merge Title+Body, preprocess text, return DataFrame."""
    df = pd.read_csv(f'{name}.csv')
    # [LAB] Title+Body merge taken from lab solution — falls back to Title if Body is NaN
    df['text'] = df.apply(
        lambda r: r['Title'] + '. ' + r['Body']
        if pd.notna(r['Body']) else r['Title'], axis=1)
    df['text']  = df['text'].apply(preprocess)
    df['label'] = df['class'].astype(int)
    return df[['text', 'label']].dropna().reset_index(drop=True)

# ──────────────────────────────────────────────
# 5. VARGHA-DELANEY A12 EFFECT SIZE
# ──────────────────────────────────────────────
def a12(x, y):
    """
    P(X > Y).  Thresholds: negligible <0.56, small 0.56, medium 0.64, large 0.71
    x = BERT scores, y = NB scores  → A12 > 0.5 means BERT tends to win.
    """
    wins = sum((1 if xi > yi else 0.5 if xi == yi else 0)
               for xi in x for yi in y)
    return wins / (len(x) * len(y))

def effect_label(a):
    if a >= 0.71: return 'large'
    if a >= 0.64: return 'medium'
    if a >= 0.56: return 'small'
    return 'negligible'

# ──────────────────────────────────────────────
# 6. NAIVE BAYES BASELINE
# ──────────────────────────────────────────────
def run_naive_bayes(df):
    res = {m: [] for m in ['precision', 'recall', 'f1', 'accuracy']}

    # [LAB] Repeated random-split structure (run per seed) adapted from lab solution
    for seed in range(REPEAT):
        X_tr, X_te, y_tr, y_te = train_test_split(
            df['text'], df['label'],
            test_size=TEST_SIZE,
            random_state=seed,
            stratify=df['label'])

        # [LAB] TF-IDF parameters (ngram_range, max_features) taken from lab solution
        tfidf   = TfidfVectorizer(ngram_range=(1, 2), max_features=1000)
        Xtr_vec = tfidf.fit_transform(X_tr)
        Xte_vec = tfidf.transform(X_te)

        # class_prior to handle imbalance (mirrors class_weight='balanced')
        n_pos = (y_tr == 1).sum(); n_neg = (y_tr == 0).sum()
        clf = MultinomialNB(class_prior=[n_neg / len(y_tr), n_pos / len(y_tr)])
        clf.fit(Xtr_vec, y_tr)
        y_pred = clf.predict(Xte_vec)

        res['precision'].append(precision_score(y_te, y_pred, average='macro', zero_division=0))
        res['recall'].append(   recall_score   (y_te, y_pred, average='macro', zero_division=0))
        res['f1'].append(       f1_score       (y_te, y_pred, average='macro', zero_division=0))
        res['accuracy'].append( accuracy_score (y_te, y_pred))

    return res

# ──────────────────────────────────────────────
# 7. BERT FINE-TUNING
# ──────────────────────────────────────────────
class BugDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.enc = tokenizer(
            texts, padding='max_length', truncation=True,
            max_length=BERT_MAX_LEN, return_tensors='pt')
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self): return len(self.labels)

    def __getitem__(self, i):
        return {k: v[i] for k, v in self.enc.items()}, self.labels[i]


def _train_one_run(X_tr, X_te, y_tr, y_te, tokenizer, seed):
    torch.manual_seed(seed)

    # Weighted loss for class imbalance
    n_pos = sum(y_tr); n_neg = len(y_tr) - n_pos
    w = torch.tensor(
        [len(y_tr)/(2*n_neg), len(y_tr)/(2*n_pos)],
        dtype=torch.float).to(DEVICE)
    loss_fn = torch.nn.CrossEntropyLoss(weight=w)

    tr_loader = DataLoader(
        BugDataset(X_tr, y_tr, tokenizer),
        batch_size=BERT_BATCH, shuffle=True)
    te_loader = DataLoader(
        BugDataset(X_te, y_te, tokenizer),
        batch_size=BERT_BATCH)

    model = DistilBertForSequenceClassification.from_pretrained(
        BERT_MODEL, num_labels=2).to(DEVICE)
    opt   = AdamW(model.parameters(), lr=BERT_LR, weight_decay=0.01)
    steps = len(tr_loader) * BERT_EPOCHS
    sched = get_linear_schedule_with_warmup(
        opt, num_warmup_steps=steps // 10, num_training_steps=steps)

    for _ in range(BERT_EPOCHS):
        model.train()
        for batch, lbls in tr_loader:
            opt.zero_grad()
            out  = model(**{k: v.to(DEVICE) for k, v in batch.items()})
            loss = loss_fn(out.logits, lbls.to(DEVICE))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()

    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for batch, lbls in te_loader:
            out = model(**{k: v.to(DEVICE) for k, v in batch.items()})
            preds.extend(torch.argmax(out.logits, 1).cpu().numpy())
            trues.extend(lbls.numpy())

    return {
        'precision': precision_score(trues, preds, average='macro', zero_division=0),
        'recall':    recall_score   (trues, preds, average='macro', zero_division=0),
        'f1':        f1_score       (trues, preds, average='macro', zero_division=0),
        'accuracy':  accuracy_score (trues, preds)
    }


def run_bert(df, tokenizer):
    res = {m: [] for m in ['precision', 'recall', 'f1', 'accuracy']}

    # [LAB] Repeated random-split structure (run per seed) adapted from lab solution
    for seed in range(REPEAT):
        print(f"    BERT run {seed+1}/{REPEAT}...", flush=True)
        X_tr, X_te, y_tr, y_te = train_test_split(
            df['text'].tolist(), df['label'].tolist(),
            test_size=TEST_SIZE, random_state=seed,
            stratify=df['label'].tolist())

        run = _train_one_run(X_tr, X_te, y_tr, y_te, tokenizer, seed)
        for m in res:
            res[m].append(run[m])

    return res

# ──────────────────────────────────────────────
# 8. STATISTICAL ANALYSIS
# ──────────────────────────────────────────────
def analyse(nb, bert, project):
    summary = {}
    for metric in ['f1', 'precision', 'recall', 'accuracy']:
        nb_v   = np.array(nb[metric])
        bert_v = np.array(bert[metric])
        _, p   = ranksums(bert_v, nb_v)
        a      = a12(bert_v.tolist(), nb_v.tolist())
        summary[metric] = {
            'nb_median':   np.median(nb_v),
            'nb_iqr':      np.percentile(nb_v, 75) - np.percentile(nb_v, 25),
            'bert_median': np.median(bert_v),
            'bert_iqr':    np.percentile(bert_v, 75) - np.percentile(bert_v, 25),
            'p_value':     p,
            'significant': p < 0.05,
            'a12':         a,
            'effect':      effect_label(a)
        }
    return summary

# ──────────────────────────────────────────────
# 9. HOLM-BONFERRONI CORRECTION
# Applied across the 5 projects per metric to control
# family-wise error rate from multiple comparisons.
# Preferred over plain Bonferroni — less conservative.
# ──────────────────────────────────────────────
def holm_bonferroni(all_summaries, metric='f1'):
    """
    Takes raw p-values from Wilcoxon tests across all projects for a given
    metric, applies Holm-Bonferroni step-down correction, and returns a dict
    of {project: adjusted_p_value}.

    Procedure:
      1. Sort p-values ascending.
      2. For rank k (1-indexed), threshold = alpha / (m - k + 1).
      3. Reject H0 while p_k <= threshold; stop at first non-rejection.
    """
    alpha = 0.05
    projects = list(all_summaries.keys())
    raw_p    = [(proj, all_summaries[proj][metric]['p_value'])
                for proj in projects]

    # Sort by p-value ascending
    raw_p.sort(key=lambda x: x[1])
    m = len(raw_p)

    adjusted   = {}
    still_rejecting = True

    for k, (proj, p) in enumerate(raw_p, start=1):
        threshold = alpha / (m - k + 1)
        if still_rejecting and p <= threshold:
            adjusted[proj] = {'p_corrected': p, 'significant_corrected': True}
        else:
            still_rejecting = False          # stop rejecting from here
            adjusted[proj] = {'p_corrected': p, 'significant_corrected': False}

    return adjusted


def apply_correction_to_summaries(all_summaries):
    """
    Adds p_corrected and significant_corrected fields to every project/metric
    entry in all_summaries, in-place.
    """
    for metric in ['f1', 'precision', 'recall', 'accuracy']:
        corrections = holm_bonferroni(all_summaries, metric)
        for proj, corr in corrections.items():
            all_summaries[proj][metric].update(corr)


# ──────────────────────────────────────────────
# 10. PLOTS
# ──────────────────────────────────────────────
def plot_boxplots(nb, bert, project):
    """
    Per-project 2x2 box plot grid.
    Each panel shows NB vs BERT side-by-side for one metric.
    Box = IQR (Q1-Q3), middle line = median, whiskers = 1.5*IQR, dots = outliers.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    metrics = ['f1', 'precision', 'recall', 'accuracy']
    labels  = ['Macro F1', 'Macro Precision', 'Macro Recall', 'Accuracy']

    for ax, m, lbl in zip(axes.flatten(), metrics, labels):
        nb_v   = nb[m]
        bert_v = bert[m]

        bp = ax.boxplot(
            [nb_v, bert_v],
            labels=['NB + TF-IDF', 'DistilBERT'],
            patch_artist=True,
            medianprops=dict(color='black', linewidth=2),
            whiskerprops=dict(linewidth=1.2),
            capprops=dict(linewidth=1.2),
            flierprops=dict(marker='o', markersize=4, alpha=0.5)
        )

        # Colour the boxes
        bp['boxes'][0].set_facecolor('steelblue')
        bp['boxes'][0].set_alpha(0.7)
        bp['boxes'][1].set_facecolor('coral')
        bp['boxes'][1].set_alpha(0.7)

        # Annotate median values above each box
        for i, vals in enumerate([nb_v, bert_v], start=1):
            ax.text(i, np.median(vals) + 0.005,
                    f'{np.median(vals):.3f}',
                    ha='center', va='bottom', fontsize=8, fontweight='bold')

        ax.set_title(f'{lbl}  —  {project}', fontsize=10)
        ax.set_ylabel(lbl, fontsize=9)
        ax.grid(axis='y', linestyle='--', alpha=0.4)

    plt.suptitle(
        f'NB vs DistilBERT over {REPEAT} runs  —  {project}', fontsize=12)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'plots', f'{project}_boxplots.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Box plot saved  → {path}")


def plot_histograms(nb, bert, project):
    """
    Per-project 2x2 histogram grid — kept alongside box plots to
    visually inspect whether the 30-run distributions are approximately
    normal (justifies use of median/IQR as summary statistics).
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    metrics = ['f1', 'precision', 'recall', 'accuracy']
    labels  = ['Macro F1', 'Macro Precision', 'Macro Recall', 'Accuracy']

    for ax, m, lbl in zip(axes.flatten(), metrics, labels):
        nb_v   = nb[m];   bert_v = bert[m]
        ax.hist(nb_v,   bins=10, alpha=0.6, color='steelblue',
                label='NB + TF-IDF', edgecolor='white')
        ax.hist(bert_v, bins=10, alpha=0.6, color='coral',
                label='DistilBERT',  edgecolor='white')
        ax.axvline(np.median(nb_v),   color='steelblue', ls='--', lw=1.5,
                   label=f'NB median = {np.median(nb_v):.3f}')
        ax.axvline(np.median(bert_v), color='coral',     ls='--', lw=1.5,
                   label=f'BERT median = {np.median(bert_v):.3f}')
        ax.set_title(f'{lbl}  —  {project}', fontsize=10)
        ax.set_xlabel(lbl, fontsize=9)
        ax.set_ylabel('Frequency', fontsize=9)
        ax.legend(fontsize=7)

    plt.suptitle(
        f'Metric distributions over {REPEAT} runs  —  {project}', fontsize=12)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'plots', f'{project}_histograms.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Histogram saved → {path}")


def plot_all_projects_f1(all_nb_results, all_bert_results):
    """
    Single summary box plot showing F1 across all projects.
    Each project gets two boxes (NB blue, BERT coral) side by side.
    This is the key comparison figure for the report.
    """
    projects = list(all_nb_results.keys())
    n        = len(projects)

    fig, ax = plt.subplots(figsize=(4 + 2 * n, 6))

    positions_nb   = []
    positions_bert = []
    data_nb        = []
    data_bert      = []

    gap = 3  # space between project groups
    for i, proj in enumerate(projects):
        base = i * gap
        positions_nb.append(base)
        positions_bert.append(base + 1)
        data_nb.append(all_nb_results[proj]['f1'])
        data_bert.append(all_bert_results[proj]['f1'])

    bp_nb = ax.boxplot(
        data_nb, positions=positions_nb, widths=0.7,
        patch_artist=True,
        medianprops=dict(color='black', linewidth=2),
        whiskerprops=dict(linewidth=1.2),
        capprops=dict(linewidth=1.2),
        flierprops=dict(marker='o', markersize=4, alpha=0.5)
    )
    bp_bert = ax.boxplot(
        data_bert, positions=positions_bert, widths=0.7,
        patch_artist=True,
        medianprops=dict(color='black', linewidth=2),
        whiskerprops=dict(linewidth=1.2),
        capprops=dict(linewidth=1.2),
        flierprops=dict(marker='o', markersize=4, alpha=0.5)
    )

    for box in bp_nb['boxes']:
        box.set_facecolor('steelblue'); box.set_alpha(0.7)
    for box in bp_bert['boxes']:
        box.set_facecolor('coral'); box.set_alpha(0.7)

    # Annotate medians
    for i, proj in enumerate(projects):
        base = i * gap
        nb_med   = np.median(all_nb_results[proj]['f1'])
        bert_med = np.median(all_bert_results[proj]['f1'])
        ax.text(base,     nb_med   + 0.005, f'{nb_med:.3f}',
                ha='center', va='bottom', fontsize=7.5, fontweight='bold')
        ax.text(base + 1, bert_med + 0.005, f'{bert_med:.3f}',
                ha='center', va='bottom', fontsize=7.5, fontweight='bold',
                color='coral')

    # X-axis labels centred between each pair
    tick_pos = [i * gap + 0.5 for i in range(n)]
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(projects, fontsize=10)
    ax.set_ylabel('Macro F1', fontsize=11)
    ax.set_title(f'Macro F1 comparison across all projects ({REPEAT} runs)',
                 fontsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.4)

    # Legend
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor='steelblue', alpha=0.7, label='NB + TF-IDF'),
        Patch(facecolor='coral',     alpha=0.7, label='DistilBERT')
    ], fontsize=10)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'plots', 'all_projects_f1_boxplot.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  All-projects F1 box plot → {path}")

# ──────────────────────────────────────────────
# 10. SAVE RESULTS TABLE
# ──────────────────────────────────────────────
def save_results_table(all_summaries):
    rows = []
    for project, summary in all_summaries.items():
        for metric, s in summary.items():
            rows.append({
                'project':               project,
                'metric':                metric,
                'nb_median':             round(s['nb_median'],   4),
                'nb_iqr':                round(s['nb_iqr'],      4),
                'bert_median':           round(s['bert_median'],  4),
                'bert_iqr':              round(s['bert_iqr'],     4),
                'p_value_raw':           round(s['p_value'],      4),
                'p_value_holm':          round(s.get('p_corrected', s['p_value']), 4),
                'significant_corrected': s.get('significant_corrected', s['significant']),
                'a12':                   round(s['a12'],          4),
                'effect_size':           s['effect']
            })
    df = pd.DataFrame(rows)
    path = os.path.join(OUTPUT_DIR, 'results_summary.csv')
    df.to_csv(path, index=False)
    print(f"\nResults table → {path}")
    return df

# ──────────────────────────────────────────────
# 11. SAVE RAW PER-RUN SCORES
# ──────────────────────────────────────────────
def save_raw_scores(nb, bert, project):
    raw = pd.DataFrame({
        'run':            list(range(1, REPEAT + 1)),
        'nb_f1':          nb['f1'],
        'nb_precision':   nb['precision'],
        'nb_recall':      nb['recall'],
        'nb_accuracy':    nb['accuracy'],
        'bert_f1':        bert['f1'],
        'bert_precision': bert['precision'],
        'bert_recall':    bert['recall'],
        'bert_accuracy':  bert['accuracy'],
    })
    path = os.path.join(OUTPUT_DIR, f'{project}_raw_scores.csv')
    raw.to_csv(path, index=False)
    print(f"  Raw scores → {path}")

# ──────────────────────────────────────────────
# 12. CROSS-PROJECT EXPERIMENT
# ──────────────────────────────────────────────
def run_cross_project(projects_data, tokenizer):
    """Train on TensorFlow, test on every other project. No folding."""
    print("\n" + "="*55)
    print("CROSS-PROJECT GENERALISATION")
    print("Source: tensorflow  →  target: all others")
    print("="*55)

    source = 'tensorflow'
    if source not in projects_data:
        print("  tensorflow.csv not found — skipping cross-project.")
        return

    src_df = projects_data[source]
    rows   = []

    # Fit NB once on full source dataset
    tfidf  = TfidfVectorizer(ngram_range=(1, 2), max_features=1000)
    Xsrc_v = tfidf.fit_transform(src_df['text'])
    n_pos  = (src_df['label'] == 1).sum()
    n_neg  = (src_df['label'] == 0).sum()
    nb_clf = MultinomialNB(
        class_prior=[n_neg / len(src_df), n_pos / len(src_df)])
    nb_clf.fit(Xsrc_v, src_df['label'])

    for target, tgt_df in projects_data.items():
        if target == source:
            continue
        print(f"\n  → {target}")

        # NB
        Xte_v   = tfidf.transform(tgt_df['text'])
        nb_pred = nb_clf.predict(Xte_v)
        nb_f1   = f1_score(tgt_df['label'], nb_pred, average='macro', zero_division=0)
        nb_pr   = precision_score(tgt_df['label'], nb_pred, average='macro', zero_division=0)
        nb_re   = recall_score   (tgt_df['label'], nb_pred, average='macro', zero_division=0)

        # BERT (single run — full source → full target)
        bert_run = _train_one_run(
            src_df['text'].tolist(), tgt_df['text'].tolist(),
            src_df['label'].tolist(), tgt_df['label'].tolist(),
            tokenizer, seed=42)

        print(f"    NB   F1={nb_f1:.4f}  |  BERT F1={bert_run['f1']:.4f}")
        rows.append({
            'source':         source,
            'target':         target,
            'nb_f1':          round(nb_f1,         4),
            'nb_precision':   round(nb_pr,          4),
            'nb_recall':      round(nb_re,          4),
            'bert_f1':        round(bert_run['f1'], 4),
            'bert_precision': round(bert_run['precision'], 4),
            'bert_recall':    round(bert_run['recall'],    4),
        })

    cross_df = pd.DataFrame(rows)
    path = os.path.join(OUTPUT_DIR, 'cross_project_results.csv')
    cross_df.to_csv(path, index=False)
    print(f"\nCross-project results → {path}")
    print(cross_df.to_string(index=False))
    return cross_df

# ──────────────────────────────────────────────
# 13. MAIN
# ──────────────────────────────────────────────
def main():
    print(f"Device : {DEVICE}")
    print(f"Repeats: {REPEAT}   |   Test split: {int(TEST_SIZE*100)}%\n")

    # ── Load datasets ──
    print("Loading datasets...")
    projects_data = {}
    for p in PROJECTS:
        try:
            df = load_project(p)
            projects_data[p] = df
            pct = 100 * df['label'].mean()
            print(f"  {p:<20} {len(df):>5} reports  "
                  f"| {int(df['label'].sum())} positive ({pct:.1f}%)")
        except FileNotFoundError:
            print(f"  {p}: NOT FOUND — skipping")

    if not projects_data:
        print("\nNo CSVs found. Place them in the working directory and re-run.")
        return

    tokenizer       = DistilBertTokenizer.from_pretrained(BERT_MODEL)
    all_nb_results  = {}
    all_bert_results= {}
    all_summaries   = {}

    # ── Per-project experiments ──
    for project, df in projects_data.items():
        print(f"\n{'='*55}")
        print(f"PROJECT: {project.upper()}")
        print(f"{'='*55}")

        # Naive Bayes
        print(f"\n[1/2] Naive Bayes — {REPEAT} runs...")
        nb_res = run_naive_bayes(df)
        all_nb_results[project] = nb_res
        print(f"  NB  median F1 = {np.median(nb_res['f1']):.4f}")

        # BERT
        print(f"\n[2/2] DistilBERT — {REPEAT} runs  (this will take a while)...")
        bert_res = run_bert(df, tokenizer)
        all_bert_results[project] = bert_res
        print(f"  BERT median F1 = {np.median(bert_res['f1']):.4f}")

        # Statistics
        summary = analyse(nb_res, bert_res, project)
        all_summaries[project] = summary

        f = summary['f1']
        print(f"\n  ── F1 summary ──")
        print(f"  NB   : median={f['nb_median']:.4f}  IQR={f['nb_iqr']:.4f}")
        print(f"  BERT : median={f['bert_median']:.4f}  IQR={f['bert_iqr']:.4f}")
        print(f"  Wilcoxon p={f['p_value']:.4f} (raw)  "
              f"({'significant *' if f['significant'] else 'not significant'})"
              f"  — Holm-Bonferroni correction applied after all projects")
        print(f"  A12={f['a12']:.4f}  ({f['effect']} effect)")

        # Plots & raw scores
        plot_boxplots(nb_res, bert_res, project)
        plot_histograms(nb_res, bert_res, project)
        save_raw_scores(nb_res, bert_res, project)

    # ── Holm-Bonferroni correction across 5 projects ──
    # Applied per metric to control family-wise error rate.
    apply_correction_to_summaries(all_summaries)

    # ── Global F1 box plot across all projects ──
    plot_all_projects_f1(all_nb_results, all_bert_results)

    # ── Summary table ──
    results_df = save_results_table(all_summaries)
    print("\n── Full results ──")
    print(results_df.to_string(index=False))

    # ── Cross-project ──
    run_cross_project(projects_data, tokenizer)

    print("\n\nAll done. Outputs in:", os.path.abspath(OUTPUT_DIR))


if __name__ == '__main__':
    main()