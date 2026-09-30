"""Download the original benchmark using the mirror documented by TensorFlow."""
from pathlib import Path
from urllib.request import urlretrieve

url = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"
target = Path(__file__).resolve().parents[1] / "data/creditcard.csv"
target.parent.mkdir(exist_ok=True)
urlretrieve(url, target)
print(f"Downloaded {target}. Review the original dataset terms before redistribution.")
