import re
import unicodedata

URL = re.compile(r"https?://[^\s]+|www\.[^\s]+", re.IGNORECASE)
HANDLE = re.compile(r"(?<!\w)@[\w]+")

def normalise(text):
    text = unicodedata.normalize("NFKC", text).casefold()
    text = URL.sub(" url ", text)
    text = HANDLE.sub(" user ", text)
    return " ".join(text.split())
