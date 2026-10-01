import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download("stopwords", quiet=True)
STOP = set(stopwords.words("english"))
STEMMER = PorterStemmer()

HTML_JUNK = {"quot", "gt", "lt", "amp", "href", "nbsp"}
SOURCES = {"reuters", "ap", "afp"}

def preprocess(text, remove_stop=False, stem=False, remove_sources=False):
    text = str(text).lower()
    text = text.replace("\\", " ")            # line-break artefacts
    text = re.sub(r"#\d+;", " ", text)        # entities like #39;
    text = re.sub(r"http\S+|www\.\S+", " ", text)   # URLs
    text = re.sub(r"[^a-z\s]", " ", text)     # keep letters only
    words = text.split()
    words = [w for w in words if w not in HTML_JUNK]
    if remove_sources:
        words = [w for w in words if w not in SOURCES]
    if remove_stop:
        words = [w for w in words if w not in STOP]
    if stem:
        words = [STEMMER.stem(w) for w in words]
    return " ".join(words)
