"""Run predict -> explain -> verify for one order without MCP.

Use this first to check that the weights, the data and the GPU all work:

    python examples/try_engine.py --order 7
"""

import argparse
import json

from precedent import Engine, load_task
from precedent import Engine, load_task
from precedent.env import load_env

load_env()   # reads TABPFN_TOKEN and HF_TOKEN from .env if present

parser = argparse.ArgumentParser()
parser.add_argument("--order", type=int, default=7)
parser.add_argument("--model", default="fast", choices=["fast", "base"])
parser.add_argument("--dataset", default="salt", choices=["salt", "csv"])
parser.add_argument("--csv")
parser.add_argument("--target")
args = parser.parse_args()

task = load_task(args.dataset, target=args.target, csv_path=args.csv)
engine = Engine(task)
print(f"{task.name}: {len(task.X_ctx)} past orders, {len(task.X_new)} new orders, target {task.target}\n")

p = engine.predict(order_id=args.order, model=args.model)
print(f"predict : '{p['predicted']}' ({p['confidence']:.0%} sure); true value '{p['true_value']}'")

ex = engine.explain(order_id=args.order, model=args.model)
print(f"explain : {ex['seconds']}s; top precedent holds {ex['top_precedent_share']:.0%} of the positive influence")
for pr in ex["precedents"][:5]:
    print(f"   #{pr['rank']} past order {pr['past_order']}  influence {pr['influence']:+.3f}  "
          f"value '{pr['value']}'  matches {len(pr['matching_fields'])} fields")

v = engine.verify(explanation_id=ex["explanation_id"])
print(f"verify  : {v['summary']}")
print(json.dumps({k: v[k] for k in ("fall_points", "random_fall_points", "verified", "strength")}, indent=2))
