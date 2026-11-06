# render_graph.py - draw the graph of any example as a PNG (and save the Mermaid text).
# Usage:  python render_graph.py 02_prompt_template
import importlib
import sys
from pathlib import Path

if len(sys.argv) < 2:
    print("Usage: python render_graph.py <example name>   e.g. 02_prompt_template")
    sys.exit(1)

name = sys.argv[1].removesuffix(".py")
module = importlib.import_module(name)  # importing does NOT run the example (it has a __main__ guard)

# Most examples define `graph` at module level. Human-in-the-loop only defines `builder`.
compiled = getattr(module, "graph", None) or module.builder.compile()
drawable = compiled.get_graph()

out_dir = Path("graphs")
out_dir.mkdir(exist_ok=True)

# 1) Text version of the graph (Mermaid). Works offline.
mermaid = drawable.draw_mermaid()
(out_dir / f"{name}.mmd").write_text(mermaid)
print(mermaid)

# 2) Picture version. Uses the free mermaid.ink service, so it needs internet.
try:
    (out_dir / f"{name}.png").write_bytes(drawable.draw_mermaid_png())
    print(f"\nSaved picture: graphs/{name}.png")
except Exception as e:
    print(f"\nCould not make the PNG ({e}).")
    print("Paste the Mermaid text above into https://mermaid.live to see the graph.")
