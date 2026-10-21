import streamlit as st
import json
import re
import os
import ast
import html
import difflib
import numpy as np
from collections import Counter
import pandas as pd
from gensim.models import KeyedVectors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import random

# LAYOUT CHANGE: wide layout + sidebar (was layout="centered")
st.set_page_config(
    page_title="शब्दसंपत्ती | Marathi Vocabulary Assistant",
    page_icon="📘",
    layout="wide",
    initial_sidebar_state="expanded",
)

# LAYOUT CHANGE: max-width 760px -> 1400px, compact hero, metric cards, tab styling
st.markdown("""
<style>
    .main .block-container {
        padding-top: 1.2rem;
        max-width: 1400px;
    }
    .hero-banner {
        background: linear-gradient(135deg, #D97706 0%, #B45309 100%);
        padding: 1.3rem 1.8rem;
        border-radius: 14px;
        margin-bottom: 1rem;
        box-shadow: 0 4px 16px rgba(180, 83, 9, 0.22);
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .hero-banner h1 { color: white; font-size: 1.6rem; margin: 0; }
    .hero-banner p { color: #FFE8CC; margin: 0.15rem 0 0 0; font-size: 0.95rem; }
    .hero-banner .tag { color: #FFE8CC; font-size: 0.8rem; text-align: right; line-height: 1.5; }
    .stTabs [data-baseweb="tab-list"] { gap: 6px; }
    .stTabs [data-baseweb="tab"] { font-weight: 600; padding: 0.5rem 1rem; }
    div[data-testid="stMetric"] {
        background: #FFF8EC;
        border: 1px solid #EFE0C8;
        border-radius: 10px;
        padding: 0.7rem 0.9rem;
    }
    .stTextArea textarea {
        border-radius: 12px;
        border: 1.5px solid #E8D5B7;
        font-size: 1.05rem;
    }
    div[data-testid="stExpander"] {
        border-radius: 10px;
        border: 1px solid #EFE0C8;
        margin-bottom: 0.4rem;
    }
    .stButton button { border-radius: 8px; font-weight: 600; padding: 0.5rem 1.4rem; }
    .result-card {
        background: #FFF3E0;
        border-left: 4px solid #D97706;
        border-radius: 8px;
        padding: 1.2rem 1.4rem;
        margin-top: 1rem;
        font-size: 1.1rem;
        line-height: 1.8;
    }
    .result-card { color: #2b2b2b; }
    .hl-old { background: #FDE2E2; border-radius: 4px; padding: 0 3px; text-decoration: line-through; text-decoration-color: #C0392B; }
    .hl-new { background: #D9F2D9; border-radius: 4px; padding: 0 3px; font-weight: 600; }
    .rep-box { background: #FFF8EC; border: 1px solid #EFE0C8; border-radius: 10px; padding: 0.8rem 1rem; margin-bottom: 0.6rem; }
    .rep-row { display: flex; align-items: center; gap: 0.7rem; margin: 0.3rem 0; color: #2b2b2b; }
    .rep-word { min-width: 6rem; font-weight: 600; }
    .rep-track { flex: 1; background: #F5E6CC; border-radius: 5px; height: 10px; overflow: hidden; }
    .rep-bar { display: block; height: 10px; background: #D97706; }
    .rep-count { min-width: 2.2rem; text-align: right; font-weight: 600; }
    footer {visibility: hidden;}
    #MainMenu {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero-banner">
    <div>
        <h1>📘 शब्दसंपत्ती सहाय्यक</h1>
        <p>Intelligent Marathi Vocabulary Assistant — paste your essay, discover richer words, choose what fits.</p>
    </div>
""", unsafe_allow_html=True)


def _prefix_ratio(a, b):
    m = min(len(a), len(b))
    return len(os.path.commonprefix([a, b])) / m if m > 0 else 0.0


def _bigram_jaccard(a, b):
    A = {a[i:i + 2] for i in range(len(a) - 1)}
    B = {b[i:i + 2] for i in range(len(b) - 1)}
    return len(A & B) / len(A | B) if (A | B) else 0.0


def pair_features(wv, a, b):
    """8 features for a word pair (a, b), both of which must be in wv."""
    ra = np.log1p(wv.key_to_index[a])   # fastText vectors are sorted by frequency
    rb = np.log1p(wv.key_to_index[b])
    return [
        float(wv.similarity(a, b)),
        _prefix_ratio(a, b),
        abs(len(a) - len(b)),
        _prefix_ratio(a[::-1], b[::-1]),           # common suffix ratio
        _bigram_jaccard(a, b),
        difflib.SequenceMatcher(None, a, b).ratio(),
        float(abs(ra - rb)),
        float(min(ra, rb)),
    ]


def make_svm():
    # Hyper-parameters chosen by grid search with 5-fold CV on training data
    return make_pipeline(StandardScaler(), SVC(kernel="rbf", C=1, gamma=0.03))


@st.cache_resource
def load_everything():
    with open("marathi_vocab_lookup.json", encoding="utf-8") as f:
        LOOKUP = json.load(f)

    dict_df = pd.read_csv("marathi_dictionary.csv", encoding="utf-8-sig")
    # literal_eval is a safe replacement for eval on CSV cell contents
    dict_df["synonyms_list"] = dict_df["synonyms_list"].apply(ast.literal_eval)
    dict_lookup = dict(zip(dict_df["word"], dict_df["synonyms_list"]))

    combined_lookup = dict(dict_lookup)
    combined_lookup.update(LOOKUP)

    wv = KeyedVectors.load_word2vec_format("cc.mr.300.vec", limit=500000)

    def common_prefix_ratio(a, b):
        prefix = os.path.commonprefix([str(a), str(b)])
        return len(prefix) / min(len(str(a)), len(str(b))) if min(len(str(a)), len(str(b))) > 0 else 0

    X, y = [], []
    all_words = list(dict_lookup.keys())
    for word, syns in dict_lookup.items():
        if word not in wv:
            continue
        for syn in syns:
            if syn in wv:
                X.append(pair_features(wv, word, syn))
                y.append(1)
    random.seed(42)
    for word in dict_lookup:
        if word not in wv:
            continue
        r = random.choice(all_words)
        if r == word or r not in wv:
            continue
        X.append(pair_features(wv, word, r))
        y.append(0)

    svm_pipeline = make_svm()
    svm_pipeline.fit(X, y)

    MARATHI_STOPWORDS = {
        "आहे","आहेत","आणि","व","ते","तो","ती","हे","ही","हा","मी","तू","तुम्ही","आम्ही",
        "त्यांनी","त्याने","त्या","यांनी","पण","किंवा","जर","तर","म्हणून","साठी","पासून",
        "मध्ये","वर","खाली","नाही","होते","होता","होती","होतो","असे","असा","अशी","कोण","काय",
        "कधी","कुठे","कसे","का","मला","माझा","माझी","माझे","मज","तुला","तुझा","तुझी","तुझे",
        "त्याला","तिला","आम्हाला","तुम्हाला","त्यांना",
        "कारण","तरी","सर्व","परंतु","मात्र","फक्त","देखील","सुद्धा","अशाप्रकारे",
        "तसेच","शिवाय","म्हणजे","जेव्हा","तेव्हा","जिथे","तिथे","ज्यामुळे",
        # known problem word: dictionary sense ("duration") differs from
        # common usage ("yesterday")
        "काल",
        "केले","केला","केली","केल्या","नेले","नेला","नेली",
        "दिले","दिला","दिली","घेतले","घेतला","घेतली",
    }

    return LOOKUP, dict_lookup, combined_lookup, wv, svm_pipeline, common_prefix_ratio, MARATHI_STOPWORDS


with st.spinner("मॉडेल लोड होत आहे... (पहिल्यांदा थोडा वेळ लागू शकतो)"):
    LOOKUP, dict_lookup, combined_lookup, wv, svm_pipeline, common_prefix_ratio, MARATHI_STOPWORDS = load_everything()

PUNCT_CHARS = ".,!?।"


def tokenize(text):
    raw = re.findall(rf"[^\s{re.escape(PUNCT_CHARS)}]+|[{re.escape(PUNCT_CHARS)}]", text)
    return [t for t in raw if t not in PUNCT_CHARS]


def segment_preserving(text):
    """Split into whitespace / punctuation / word segments, keeping
    everything so the essay can be rebuilt with only chosen words swapped."""
    return re.findall(rf"\s+|[{re.escape(PUNCT_CHARS)}]|[^\s{re.escape(PUNCT_CHARS)}]+", text)


# --- Gender-agreement correction (targeted heuristic, not full grammar) ---
NOUN_GENDER = {
    "पाऊस": "m", "वारा": "m", "रस्ता": "m", "मार्ग": "m", "दिवस": "m",
    "मोसम": "m", "ऋतू": "m", "आनंद": "m", "मोद": "m", "चहा": "m",
    "थंडी": "f", "शाळा": "f", "वेळ": "f", "भाषा": "f", "मदत": "f",
    "घर": "n", "ऊन": "n", "पाणी": "n", "फूल": "n", "मूल": "n",
}
GENDER_ENDING = {"m": "ा", "f": "ी", "n": "े"}


def adjust_adjective_gender(word, next_word):
    if not word or word[-1] not in "ाीे":
        return word
    gender = NOUN_GENDER.get(next_word)
    if gender is None:
        return word
    return word[:-1] + GENDER_ENDING[gender]


# --- Adverbial-form correction (small, scoped lookup) -----------------
ADVERBIAL_FORM = {
    "प्रसन्न": "प्रसन्नतेने",
    "आनंद": "आनंदाने",
    "छान": "छानपणे",
    "सुखी": "सुखाने",
    "शांत": "शांतपणे",
    "नम्र": "नम्रपणे",
    "विनम्र": "विनम्रपणे",
}
# NOTE: झाले/झाला/झाली were removed from this set. "आनंद झाला" is already
# correct; turning it into "आनंदाने झाला" was a bug.
ADVERBIAL_CONTEXT_CUES = {"एकत्र", "करत", "येऊन", "राहत", "राहिले",
                          "बोलतो", "बोलते", "बोलला", "बोलली"}


def adjust_adverbial_form(word, next_word):
    if word in ADVERBIAL_FORM and next_word in ADVERBIAL_CONTEXT_CUES:
        return ADVERBIAL_FORM[word]
    return word


# --- Irregular verb clusters -----------------------------------------
IRREGULAR_VERB_CLUSTERS = [
    {"येत", "आला", "आली", "आले", "येईल", "येतो", "येते"},
    {"जात", "गेला", "गेली", "गेले", "जाईल", "जातो", "जाते"},
    {"करत", "केला", "केली", "केले", "करेल", "करतो", "करते"},
    {"होत", "झाला", "झाली", "झाले", "होईल", "होतो", "होते"},
    {"देत", "दिला", "दिली", "दिले", "देईल", "देतो", "देते"},
]


def same_irregular_verb(a, b):
    return any(a in cluster and b in cluster for cluster in IRREGULAR_VERB_CLUSTERS)


def is_likely_inflection(a, b):
    if same_irregular_verb(a, b):
        return True
    return common_prefix_ratio(a, b) >= 0.5


# Core nouns whose synonyms break the verb phrase around them
# (e.g. "पाऊस पडला" -> "वर्षाव पडला" is ungrammatical). No suggestions shown.
NO_SUGGEST = {"पाऊस"}


def suggest(word, top_k=3, min_embedding_sim=0.6):
    if word in NO_SUGGEST:
        return []
    if word in LOOKUP:
        single_word = [w for w in LOOKUP[word] if len(w.split()) == 1]
        two_word = [w for w in LOOKUP[word] if len(w.split()) == 2]
        candidates = single_word + two_word
        if candidates:
            return [{"word": w, "score": 1.0, "source": "dataset"} for w in candidates[:top_k]]
    if word in dict_lookup:
        return [{"word": w, "score": 0.9, "source": "dictionary"} for w in dict_lookup[word][:top_k]]
    # Very short words give spurious embedding matches -- skip them.
    if word not in wv or len(word) <= 3:
        return []
    candidate_pool = list(combined_lookup.keys())
    sims = [(c, wv.similarity(word, c)) for c in candidate_pool if c in wv and c != word]
    sims.sort(key=lambda x: -x[1])
    scored = []
    for cand, sim in sims[:15]:
        if sim < min_embedding_sim or is_likely_inflection(word, cand):
            continue
        feats = [pair_features(wv, word, cand)]
        if svm_pipeline.predict(feats)[0] == 1:
            scored.append({"word": cand, "score": float(sim), "source": "embedding"})
    return scored[:top_k]


@st.cache_resource(show_spinner=False)
def load_sbert():
    """Optional context model. Returns None if it can't be loaded."""
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer("l3cube-pune/indic-sentence-bert-nli")
    except Exception:
        return None


# --- Context rules -----------------------------------------------------
# (word, next word) -> suggestions that fit THAT combination. Applied per
# occurrence, so the same word can get different suggestions in different
# places (खूप प्रयत्न -> अथक प्रयत्न, but खूप हुशार -> अतिशय हुशार).
COLLOCATIONS = {
    ("खूप", "प्रयत्न"): ["अथक", "भरपूर"],
    ("खूप", "मेहनत"): ["अथक", "भरपूर"],
    ("खूप", "अभ्यास"): ["सखोल", "भरपूर"],
    ("खूप", "पाऊस"): ["मुसळधार", "जोरदार"],
    ("खूप", "ऊन"): ["कडक", "रखरखीत"],
    ("खूप", "थंडी"): ["कडाक्याची", "बोचरी"],
    ("खूप", "वारा"): ["जोरदार", "सोसाट्याचा"],
    ("खूप", "आनंद"): ["अपार", "अतीव"],
    ("खूप", "वेळ"): ["बराच", "भरपूर"],
}
# If the word before a replacement is already an intensifier, a suggestion
# that starts with "अत्यंत" would double it ("अतिशय अत्यंत ...").
INTENSIFIERS = {"अत्यंत", "अतिशय", "फार", "खूप", "अती", "भरपूर"}
SENT_END = set(".!?।")


def is_word_seg(seg):
    return bool(seg.strip()) and seg not in PUNCT_CHARS


def sentence_span(segs, idx):
    lo = idx
    while lo > 0 and segs[lo - 1] not in SENT_END:
        lo -= 1
    hi = idx
    while hi < len(segs) - 1 and segs[hi + 1] not in SENT_END:
        hi += 1
    return lo, hi


def rerank_with_context(segs, idx, alts, model):
    """Swap each candidate into this occurrence's sentence and score how
    close the new sentence stays to the original (sentence embeddings)."""
    lo, hi = sentence_span(segs, idx)
    sent = "".join(segs[lo:hi + 1]).strip()
    new_sents = []
    for a in alts:
        tmp = list(segs[lo:hi + 1])
        tmp[idx - lo] = a["word"]
        new_sents.append("".join(tmp).strip())
    emb = model.encode([sent] + new_sents, normalize_embeddings=True)
    sims = emb[1:] @ emb[0]
    for a, sc in zip(alts, sims):
        a["ctx"] = float(sc)
    return sorted(alts, key=lambda a: -a["ctx"])


def scan_essay(text, top_k=3, model=None):
    """One entry per OCCURRENCE of a flaggable word (not per unique word)."""
    segs = segment_preserving(text)
    widx = [i for i, sg in enumerate(segs) if is_word_seg(sg)]
    cache = {}
    occurrences = []
    for pos, i in enumerate(widx):
        tok = segs[i]
        if tok in MARATHI_STOPWORDS or len(tok) <= 2:
            continue
        if tok not in cache:
            cache[tok] = suggest(tok, top_k=top_k)
        base = [dict(a) for a in cache[tok]]
        prv = segs[widx[pos - 1]] if pos > 0 else ""
        nxt = segs[widx[pos + 1]] if pos + 1 < len(widx) else ""
        colloc = [{"word": w, "score": 1.0, "source": "context"}
                  for w in COLLOCATIONS.get((tok, nxt), [])]
        if not base and not colloc:
            continue
        if model is not None and base:
            base = rerank_with_context(segs, i, base, model)
        merged, seen_w = [], set()
        for a in colloc + base:
            if a["word"] not in seen_w:
                merged.append(a)
                seen_w.add(a["word"])
        occurrences.append({"idx": i, "word": tok, "prev": prv, "next": nxt,
                            "alternatives": merged[:top_k]})
    return occurrences


def build_enriched(original, chosen):
    """chosen: {segment index: replacement}. Returns (text, grammar_hints)."""
    segments = segment_preserving(original)
    enriched = list(segments)
    widx = [i for i, sg in enumerate(segments) if is_word_seg(sg)]

    for pos, i in enumerate(widx):
        if i not in chosen:
            continue
        new = chosen[i]
        prev = enriched[widx[pos - 1]] if pos > 0 else ""
        if prev in INTENSIFIERS and new.startswith("अत्यंत "):
            new = new[len("अत्यंत "):]
        enriched[i] = new

    # Grammar fixes touch only words the user replaced.
    for pos, i in enumerate(widx[:-1]):
        if i not in chosen:
            continue
        nxt = enriched[widx[pos + 1]]
        enriched[i] = adjust_adjective_gender(enriched[i], nxt)
        enriched[i] = adjust_adverbial_form(enriched[i], nxt)

    # Hints (not auto-applied) for words the user left unchanged.
    hints = []
    for pos, i in enumerate(widx[:-1]):
        w, nxt = enriched[i], enriched[widx[pos + 1]]
        if i not in chosen and w in ADVERBIAL_FORM and nxt in ADVERBIAL_CONTEXT_CUES:
            hints.append(f"'{w} {nxt}' → '{ADVERBIAL_FORM[w]} {nxt}'?")
    return "".join(enriched), hints, segments, enriched


def richness(text):
    """Vocabulary richness = unique words / total words (type-token ratio)."""
    toks = tokenize(text)
    total = len(toks)
    content = [t for t in toks if t not in MARATHI_STOPWORDS and len(t) > 2]
    counts = Counter(content)
    return {
        "ttr": len(set(toks)) / total if total else 0.0,
        "unique": len(set(toks)),
        "total": total,
        "repeated": sum(1 for c in counts.values() if c >= 2),
    }


def overused_words(text, min_count=2):
    content = [t for t in tokenize(text) if t not in MARATHI_STOPWORDS and len(t) > 2]
    return sorted(((w, c) for w, c in Counter(content).items() if c >= min_count),
                  key=lambda x: (-x[1], x[0]))


def highlight_html(segs, changed, css_class):
    out = []
    for i, seg in enumerate(segs):
        t = html.escape(seg).replace("\n", "<br>")
        out.append(f'<span class="{css_class}">{t}</span>' if i in changed else t)
    return "".join(out)


SOURCE_LABEL = {
    "dataset": " यापूर्वी वापरलेले",
    "dictionary": " शब्दकोश",
    "embedding": " सुचवलेले",
    "context": " संदर्भानुसार",
}

SAMPLE_ESSAYS = {
    "(नमुना निबंध निवडा)": "",
    "माझा आवडता ऋतू": (
        "माझा आवडता ऋतू पावसाळा आहे. पावसाळ्यात आकाशात मोठे काळे ढग येतात. "
        "खूप पाऊस पडतो आणि सगळीकडे हिरवळ दिसते. थंड वारा वाहतो. "
        "मला पावसात भिजायला खूप आवडते. आम्ही मित्र एकत्र खेळतो आणि आम्हाला खूप आनंद होतो. "
        "आई गरम चहा देते. पावसाळा मला खूप छान वाटतो."
    ),
    "माझा मित्र": (
        "माझा मित्र रोहन खूप हुशार आहे. तो नेहमी मेहनत करतो आणि सर्वांना मदत करतो. "
        "त्याचा स्वभाव खूप चांगला आहे. परीक्षेचा अभ्यास कठीण होता, पण त्याने खूप प्रयत्न केले. "
        "तो खेळातही चांगला आहे. आम्ही रोज शाळेत एकत्र जातो. "
        "मला रोहनसारखा मित्र मिळाला याचा खूप आनंद आहे."
    ),
    "माझी शाळा": (
        "आमची शाळा खूप मोठी आणि सुंदर आहे. शाळेचे मैदान खूप स्वच्छ आहे. "
        "आमचे शिक्षक खूप चांगले आहेत आणि ते आम्हाला नेहमी मदत करतात. "
        "वर्गात आम्ही शांत बसतो. मला गणित सोपे वाटते आणि विज्ञान खूप आवडते. "
        "शाळेत आम्ही छान खेळ खेळतो. मला माझी शाळा खूप आवडते."
    ),
    "सहलीचा दिवस": (
        "आम्ही सहलीला गेलो. प्रवास खूप छान झाला. रस्ता खूप लांब होता, पण सगळे आनंदी होते. "
        "आम्ही एक जुना किल्ला पाहिला. किल्ल्यावरून दिसणारे दृश्य खूप सुंदर होते. "
        "सूर्य मावळताना आकाश खूप सुंदर दिसत होते. थकलो होतो, तरी खूप मजा आली. "
        "तो दिवस खूप चांगला होता."
    ),
    "माझे गाव": (
        "माझे गाव खूप शांत आणि सुंदर आहे. गावाजवळ एक मोठी नदी आहे आणि तिचे पाणी स्वच्छ आहे. "
        "गावातील लोक खूप प्रेमळ आहेत. सकाळी पक्षी गातात आणि थंड वारा वाहतो. "
        "शेतकरी रोज शेतात मेहनत करतात. सणाच्या दिवशी सगळीकडे आनंद असतो. "
        "मला माझे गाव खूप आवडते."
    ),
}


def _load_sample():
    text = SAMPLE_ESSAYS.get(st.session_state.get("sample_choice"), "")
    if text:
        st.session_state["essay_text"] = text


# ---------------------------------------------------------------------
# Model accuracy (held-out evaluation) -- logic unchanged, moved above the
# layout so the Model Performance tab can call it.
# ---------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def evaluate_models():
    import numpy as np
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
    )

    rng = random.Random(42)
    words = [w for w in dict_lookup if w in wv]
    rng.shuffle(words)
    split = int(0.8 * len(words))
    train_words, test_words = words[:split], words[split:]

    def feats(a, b):
        return pair_features(wv, a, b)

    def build_pairs(word_list, neg_pool):
        # Negatives are drawn only from neg_pool, so test negatives come
        # from test words and never touch the training words.
        X, y = [], []
        for w in word_list:
            true_syns = set(dict_lookup[w])
            for s in dict_lookup[w]:
                if s in wv and s != w:
                    X.append(feats(w, s))
                    y.append(1)
            neg = rng.choice(neg_pool)
            tries = 0
            while (neg == w or neg in true_syns) and tries < 10:
                neg = rng.choice(neg_pool)
                tries += 1
            if neg != w and neg not in true_syns:
                X.append(feats(w, neg))
                y.append(0)
        return X, y

    X_tr, y_tr = build_pairs(train_words, train_words)
    X_te, y_te = build_pairs(test_words, test_words)

    def new_svm():
        return make_svm()

    svm = new_svm()
    svm.fit(X_tr, y_tr)
    pred = svm.predict(X_te)
    tn, fp, fn, tp = confusion_matrix(y_te, pred, labels=[0, 1]).ravel()

    cv_scores = cross_val_score(
        new_svm(), X_tr, y_tr,
        cv=StratifiedKFold(5, shuffle=True, random_state=42), scoring="f1"
    )

    return {
        "train_pairs": len(X_tr), "test_pairs": len(X_te),
        "accuracy": accuracy_score(y_te, pred),
        "precision": precision_score(y_te, pred, zero_division=0),
        "recall": recall_score(y_te, pred, zero_division=0),
        "f1": f1_score(y_te, pred, zero_division=0),
        "cv_f1_mean": float(cv_scores.mean()), "cv_f1_std": float(cv_scores.std()),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


# =====================================================================
# LAYOUT: sidebar + tabs
# =====================================================================
with st.sidebar:
    st.markdown("##  Settings")
    use_context = st.checkbox(
        " संदर्भानुसार पर्यायांचा क्रम लावा (IndicSBERT)",
        value=False,
        help="Reranks suggestions by how well each fits the sentence. "
             "First use downloads the model (needs internet and sentence-transformers).",
    )
    st.divider()
    st.markdown("##  System status")
    st.metric("fastText vectors loaded", f"{len(wv):,}")
    st.metric("Dataset lookup words", f"{len(LOOKUP):,}")
    st.metric("Dictionary headwords", f"{len(dict_lookup):,}")
    st.caption("SVM: RBF kernel · 8 features")
    st.divider()
    st.caption("NLP Mini Project · St. Francis Institute of Technology")

tab_an, tab_eval = st.tabs([" Analyze Essay", " Model Performance"])


# ====================================================== TAB 1: ANALYZE
with tab_an:
    left, right = st.columns([1, 1], gap="large")

    with left:
        st.markdown("####  तुमचा निबंध टाका/Enter your essay:")
        st.selectbox(
            " नमुना निबंध",
            list(SAMPLE_ESSAYS),
            key="sample_choice",
            on_change=_load_sample,
            help="Pick a ready example to try the app quickly, or type your own essay below.",
        )
        essay = st.text_area(
            "essay",
            key="essay_text",
            height=230,
            placeholder="उदा. आज खूप पाऊस पडला आणि मला खूप आनंद झाला.",
            label_visibility="collapsed",
        )
        scan_clicked = st.button(" शब्द तपासा", use_container_width=True)

    if scan_clicked and essay.strip():
        sbert = None
        if use_context:
            with st.spinner("संदर्भ मॉडेल लोड होत आहे..."):
                sbert = load_sbert()
            if sbert is None:
                st.warning("Context model could not be loaded; showing normal suggestions. "
                           "Install it with: pip install sentence-transformers")
        st.session_state["scan_results"] = scan_essay(essay, model=sbert)
        st.session_state["scan_essay_text"] = essay

    with right:
        st.markdown("####  वारंवार वापरलेले शब्द")
        if "scan_results" not in st.session_state:
            st.info("निबंध टाका आणि **शब्द तपासा** दाबा — येथे विश्लेषण दिसेल.")
        else:
            original_essay = st.session_state["scan_essay_text"]
            stats = richness(original_essay)
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Words", stats["total"])
            k2.metric("Unique", stats["unique"])
            k3.metric("Flagged spots", len(st.session_state["scan_results"]))
            k4.metric("Repeated", stats["repeated"])

            repeated = overused_words(original_essay)
            if repeated:
                top = repeated[:8]
                mx = top[0][1]
                rows = "".join(
                    f'<div class="rep-row"><span class="rep-word">{html.escape(w)}</span>'
                    f'<span class="rep-track"><span class="rep-bar" style="width:{int(100 * c / mx)}%"></span></span>'
                    f'<span class="rep-count">×{c}</span></div>'
                    for w, c in top
                )
                st.markdown(f'<div class="rep-box">{rows}</div>', unsafe_allow_html=True)
                st.caption("हे शब्द निबंधात पुन्हा पुन्हा आले आहेत — यांच्यासाठी वेगवेगळे पर्याय वापरून पहा.")
            else:
                st.caption("कोणताही शब्द वारंवार वापरलेला नाही.")

    if "scan_results" in st.session_state:
        results = st.session_state["scan_results"]
        original_essay = st.session_state["scan_essay_text"]
        essay_key = abs(hash(original_essay))

        st.divider()
        if not results:
            st.info("कोणतेही सुधारणा सुचवण्यासारखे शब्द सापडले नाहीत.")
        else:
            st.markdown("####  सुचवलेले पर्याय")
            st.caption("प्रत्येक ठिकाणासाठी पर्याय निवडा, किंवा 'बदलू नका' ठेवा. "
                       "एकच शब्द वेगवेगळ्या ठिकाणी वेगळ्या प्रकारे बदलता येतो.")

            chosen = {}
            grid = st.columns(3, gap="medium")
            for n, r in enumerate(results):
                with grid[n % 3]:
                    options = ["(बदलू नका)"] + [
                        f"{alt['word']}  —  {SOURCE_LABEL[alt['source']]}, स्कोअर {alt['score']:.2f}"
                        + (f", संदर्भ {alt['ctx']:.2f}" if "ctx" in alt else "")
                        for alt in r["alternatives"]
                    ]
                    ctx = " ".join(x for x in (r["prev"], f"‹{r['word']}›", r["next"]) if x)
                    choice = st.selectbox(
                        f"**{r['word']}**  ·  {ctx}",
                        options,
                        key=f"choice_{essay_key}_{r['idx']}",
                    )
                    if choice != "(बदलू नका)":
                        chosen[r["idx"]] = choice.split("  —  ")[0]

            st.divider()

            if st.button(" सुधारित निबंध तयार करा", use_container_width=True):
                enriched, hints, orig_segs, new_segs = build_enriched(original_essay, chosen)
                changed = set(chosen)

                before, after = richness(original_essay), richness(enriched)
                st.markdown("####  शब्दसंपत्ती गुण")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Richness score",
                          f"{before['ttr'] * 100:.0f}% → {after['ttr'] * 100:.0f}%",
                          f"{(after['ttr'] - before['ttr']) * 100:+.0f} pts")
                m2.metric("Unique words", f"{before['unique']} → {after['unique']}")
                m3.metric("Repeated words", f"{before['repeated']} → {after['repeated']}",
                          f"{after['repeated'] - before['repeated']:+d}", delta_color="inverse")
                m4.metric("Words changed", len(changed))
                st.caption("Richness score = unique words ÷ total words. "
                           "Higher means less repetition.")

                st.markdown("####  मूळ आणि सुधारित निबंध")
                c_old, c_new = st.columns(2, gap="large")
                c_old.markdown("**मूळ**")
                c_old.markdown(
                    f'<div class="result-card">{highlight_html(orig_segs, changed, "hl-old")}</div>',
                    unsafe_allow_html=True)
                c_new.markdown("**सुधारित**")
                c_new.markdown(
                    f'<div class="result-card">{highlight_html(new_segs, changed, "hl-new")}</div>',
                    unsafe_allow_html=True)
                if hints:
                    st.info(" व्याकरण सूचना (तपासा): " + "; ".join(hints))
                st.caption(
                    "टीप: फक्त तुम्ही निवडलेले शब्दच बदलले आहेत. "
                    "हे साधन शब्दसंपत्ती सुचवते — व्याकरणदृष्ट्या (लिंग/वचन/काळ) "
                    "बरोबर आहे का ते कृपया स्वतः तपासा."
                )


# ================================================ TAB 2: PERFORMANCE
with tab_eval:
    st.markdown("####  मॉडेल अचूकता (Model Accuracy)")
    st.caption("Model: RBF SVM, 8 features (C=1, γ=0.03)")
    st.caption(
        "Measured on held-out dictionary words (20% of words, never seen by the "
        "evaluation SVM during training). Runs once, then results are cached."
    )
    analysed = "scan_results" in st.session_state
    if not analysed:
        st.info("पहिले **Analyze Essay** टॅबमध्ये निबंध तपासा (शब्द तपासा), "
                "नंतर येथे मूल्यमापन चालवा.")

    if st.button("▶ Run evaluation", disabled=not analysed):
        with st.spinner("मूल्यमापन चालू आहे... (काही मिनिटे लागू शकतात)"):
            st.session_state["eval_result"] = evaluate_models()

    m = st.session_state.get("eval_result")
    if m is not None:
        st.markdown("**1. SVM classifier — is this candidate a genuine synonym?**")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy", f"{m['accuracy']*100:.1f}%")
        c2.metric("Precision", f"{m['precision']*100:.1f}%")
        c3.metric("Recall", f"{m['recall']*100:.1f}%")
        c4.metric("F1 score", f"{m['f1']*100:.1f}%")
        st.caption(
            f"5-fold cross-validated F1 on training data: "
            f"{m['cv_f1_mean']*100:.1f}% ± {m['cv_f1_std']*100:.1f}%  |  "
            f"{m['train_pairs']} training pairs, {m['test_pairs']} held-out test pairs"
        )
        st.dataframe(
            pd.DataFrame(
                [[m["tn"], m["fp"]], [m["fn"], m["tp"]]],
                index=["Actual: not synonym", "Actual: synonym"],
                columns=["Predicted: not synonym", "Predicted: synonym"],
            )
        )
        st.info(
            "Note: negatives are randomly chosen non-synonyms, which are easier "
            "than the near-miss candidates seen in real use, so treat F1 and "
            "precision as the main indicators rather than accuracy alone."
        )
    elif analysed:
        st.info("Click **Run evaluation** to compute metrics on the held-out split.")