import pandas as pd

df = pd.read_json(
    "hf://datasets/bingbangboom/stockfish-evaluation-SAN/stockfish_evaluations.jsonl",
    lines=True,
    nrows=1_000_000,
)

df.to_csv("stockfish_evaluations.csv", index=False)
