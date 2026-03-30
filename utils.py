import os, json, hashlib, requests
from collections import defaultdict, Counter
import plotly.graph_objects as go
import anthropic
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Config ────────────────────────────────────────────────────────────────────
API_URL    = "https://arron-sausagelike-rightwardly.ngrok-free.dev/predict"
API_KEY    = ""
CACHE_DIR  = "cache"

# ── Constants ─────────────────────────────────────────────────────────────────
LLM_QUERY_PROMPT_TEMPLATE = """
### Instruction: Below is an topic and several related news articles. Please generate 1 frame-generic and 5 frame-targeted (one for each frame, respectively) natural and concise queries (6 total) based on the following requirements:  
1. Each query should focus on the topic and be different from each other.
2. Each query should be as concise as possible while all the provided articles can answer it directly.
3. Ensure your output strictly adheres to the following JSON format: ["<content of query 1>", "<content of query 2>", ...] without any additional text or explanation.
4. The frame-generic query must not invoke to any of the specific frames described below:

### Frame descriptions:
Conflict: This frame emphasizes conflict between individuals, groups, or institutions as a means of capturing audience interest.
Human Interest: This frame brings a human face or an emotional angle to the presentation of an event, issue, or problem.
Economic: This frame reports an event, problem, or issue in terms of the consequences it will have economically on an individual, group, institution, region, or country.
Morality: This frame puts the event, problem, or issue in the context of religious tenets or moral prescriptions.
Responsibility: This frame presents an issue or problem in such a way as to attribute responsibility for its cause or solution to either the government or to an individual or group.

### Topic: {{topic}}

### Relevant articles:
"""

LLM_RAG_PROMPT = """
### Instruction: Write an accurate, engaging, and concise answer for the given question using only the provided search results and cite them properly using [1][2][3] etc. Ensure the answer adheres to the following strict requirements: 
1. The order of the provided documents is random, so consider them fairly without bias toward their position in the list. 
2. You must cite the provided documents below. If multiple documents support the answer, cite all of them.

### Question: {{query}}

### Search Results:
"""

LLM_NARRATIVE_ANALYSIS_PROMPT = """
### Instruction: According to the narrative policy framework, classify the characters or character groups in the following news article text into the following roles following the rules below the role descriptions:

1. Heroes: Characters that provide or promise to provide relief from the harm and presume to solve the problem
2. Victims: Characters that are harmed
3. Villains: Characters that do the harm

### Rules:
1. Attempt to identify the one character per role, where the role is dominated by that one character
2. If multiple characters equally dominate a role, list all of the in that role (MAX 5 characters per role)
3. If no character clearly dominates a role, return an empty list for that role
4. Write no explanation text of what you are doing. Return your result in the following JSON format: {"hero": ["<character 1>", "<character 2>", ...], "victim": [...], "villian": [...]}

### Article text:
"""


FRAMES = ["conflict", "economic", "human_interest", "morality", "responsibility"]
FRAME_COLORS = {
    "conflict":       "#e74c3c",
    "economic":       "#3498db",
    "human_interest": "#2ecc71",
    "morality":       "#9b59b6",
    "responsibility": "#f39c12",
}

DROP_COMBO = frozenset(["conflict", "human_interest", "responsibility"]) 

from dotenv import load_dotenv

load_dotenv(".env.local")
CLIENT = anthropic.Client(api_key=os.getenv("ANTHROPIC_API_KEY"))

# ── Helpers ───────────────────────────────────────────────────────────────────
def _frames_present(result: dict) -> frozenset[str]:
    """Return frozenset of frames present in a generic_framing result dict."""
    return frozenset(f for f in FRAMES if result.get(f, {}).get("present", False))

def _cache_key(text: str, model_type: str) -> str:
    """Stable hash of text + model_type — used as the cache filename."""
    digest = hashlib.sha256(f"{model_type}:{text}".encode()).hexdigest()[:16]
    return digest


def cache_get(text: str, model_type: str) -> dict | None:
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, f"{_cache_key(text, model_type)}.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def cache_set(text: str, model_type: str, result: dict) -> None:
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, f"{_cache_key(text, model_type)}.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2)


def generate_llm_queries(sources, topic):
    cache_key = f"queries:{topic}:{','.join(s['title'] for s in sources)}"
    cached = cache_get(cache_key, "llm_queries")
    if cached is not None:
        print("  → Cache hit: LLM queries")
        return cached

    prompt = LLM_QUERY_PROMPT_TEMPLATE.replace("{{topic}}", topic)
    for idx, source in enumerate(sources):
        prompt += f"[{idx + 1}] {source['title']}\n"
        prompt += f"{source['content']}\n\n"
        
    response = CLIENT.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=2048,
        messages=[{"role": "user", "content": prompt}]
    )

    raw = response.content[0].text
    raw = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    result = json.loads(raw)
    cache_set(cache_key, "llm_queries", result)
    return result


def generate_rag_response(articles, query):
    cache_key = f"rag:{query}:{','.join(a['title'] for a in articles)}"
    cached = cache_get(cache_key, "rag_response")
    if cached is not None:
        print(f"  → Cache hit: RAG response for '{query[:50]}...'")
        return cached["text"]

    prompt = LLM_RAG_PROMPT.replace("{{query}}", query)
    for idx, article in enumerate(articles):
        prompt += f"[{idx + 1}] {article['title']}\n"
        prompt += f"{article['content']}\n\n"

    response = CLIENT.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=2048,
        messages=[{"role": "user", "content": prompt}]
    )

    text = response.content[0].text
    cache_set(cache_key, "rag_response", {"text": text})
    return text


def call_framing_api(text: str, label: str = "", model_type: str = "") -> dict | None:
    print(f"  → Calling API for: {label}")
    try:
        resp = requests.post(
            API_URL,
            json={"text": text, "model_type": model_type, "api_key": API_KEY},
            timeout=30,
        )
        resp.raise_for_status()
        result = resp.json()
        return result
    except requests.exceptions.RequestException as e:
        print(f"  [ERROR] API call failed for '{label}': {e}")
        return None


def extract_present_frames(result: dict) -> list[str]:
    return [f for f in FRAMES if result.get(f, {}).get("present", False)]


def hex_to_rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"

    
def load_articles_and_topic(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def save_json(path: str, data: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=4)
    

def analyze_narrative_frame(text: str, label: str) -> dict | None:
    print(f"  → Analyzing narrative framing for: {label}")
    prompt = LLM_NARRATIVE_ANALYSIS_PROMPT + text
    response = CLIENT.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=2048,
        messages=[{"role": "user", "content": prompt}]
    )
    raw = response.content[0].text
    raw = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()

    if not raw:
        print(f"  → WARNING: Empty response for {label} | stop_reason={response.stop_reason}")
        return None

    print(f"  → DEBUG raw repr: {repr(raw[:500])}")
    return json.loads(raw)


def analyze_articles(articles: list[dict], model_type, do_narrative=True) -> list[dict]:
    results = []
    for idx, article in enumerate(articles):
        text = f"{article['title']}\n\n{article['content']}"
        label = f"Article [{idx + 1}]"

        result = {"generic_framing": {}, "narrative_framing": {}} 
        cached = cache_get(text, model_type)
        if cached:
            print(f"  → Cache hit: {label}")
            result = cached
        else:
            _generic_result = call_framing_api(text, label, model_type)
            if _generic_result is not None:
                result["generic_framing"] = _generic_result

            if do_narrative:
                _narrative_result = analyze_narrative_frame(text, label)
                if _narrative_result is not None:
                    result["narrative_framing"] = _narrative_result

        if do_narrative and (not result["narrative_framing"]):
            _narrative_result = analyze_narrative_frame(text, label)
            if _narrative_result is not None:
                result["narrative_framing"] = _narrative_result
            
        result["article_title"] = article["title"]
        result["article_content"] = article["content"]
        cache_set(text, model_type, result)
        results.append(result)
        
        if len(results) > 100:
            print("Finished:", len(results))
    return results


# ── Visualizations ────────────────────────────────────────────────────────────
def save_sankey(article_results: list[dict], topic: str, rag_result: dict, out_dir: str, model_name: str):
    n = len(article_results)
    article_labels = [f"Article [{i+1}]" for i in range(n)]
    frame_labels   = [f.replace("_", " ").title() for f in FRAMES]
    rag_labels     = [f"RAG: {f.replace('_', ' ').title()}" for f in FRAMES]

    all_labels = article_labels + frame_labels + rag_labels
    idx = {lbl: i for i, lbl in enumerate(all_labels)}

    source, target, value, link_colors = [], [], [], []
    frame_weight = defaultdict(float)

    for i, res in enumerate(article_results):
        if res is None:
            continue
        for frame in extract_present_frames(res):
            fl = frame.replace("_", " ").title()
            source.append(idx[f"Article [{i+1}]"])
            target.append(idx[fl])
            value.append(1)
            link_colors.append(hex_to_rgba(FRAME_COLORS[frame], 0.4))
            frame_weight[frame] += 1

    if rag_result:
        rag_present = extract_present_frames(rag_result)
        for frame in FRAMES:
            fl  = frame.replace("_", " ").title()
            rl  = f"RAG: {fl}"
            w   = frame_weight.get(frame, 0)
            rag = frame in rag_present
            if w > 0 or rag:
                flow = max(w, 0.5) if rag else w * 0.3
                source.append(idx[fl])
                target.append(idx[rl])
                value.append(max(flow, 0.1))
                link_colors.append(hex_to_rgba(FRAME_COLORS[frame], 0.6 if rag else 0.15))

    node_colors = (
        ["#b0bec5"] * n
        + [FRAME_COLORS[f] for f in FRAMES]
        + [hex_to_rgba(FRAME_COLORS[f], 0.5) for f in FRAMES]
    )

    fig = go.Figure(go.Sankey(
        arrangement="snap",
        node=dict(
            pad=20, thickness=22,
            line=dict(color="white", width=0.5),
            label=all_labels,
            color=node_colors,
            hovertemplate="%{label}<extra></extra>",
        ),
        link=dict(
            source=source, target=target, value=value, color=link_colors,
            hovertemplate="%{source.label} → %{target.label}: %{value:.1f}<extra></extra>",
        ),
    ))

    fig.update_layout(
        title=dict(
            text=f"Narrative Frame Shift | {topic} | {model_name}",
            font=dict(size=16, family="Arial, sans-serif"),
            x=0.01, xanchor="left",
        ),
        paper_bgcolor="white",
        font=dict(color="#333333", size=12, family="Arial, sans-serif"),
        margin=dict(l=20, r=20, t=80, b=20),
        height=600,
    )

    path = os.path.join(out_dir, f"frame_shift_sankey_{model_name}.png")
    fig.write_image(path, width=1000, height=600, scale=2)
    print(f"  Saved: {path}")


def save_bar(article_results: list[dict], topic: str, rag_result: dict, out_dir: str, model_name: str):
    valid = [r for r in article_results if r]
    frame_counts = {f: sum(1 for r in valid if f in extract_present_frames(r)) for f in FRAMES}
    article_pct  = [frame_counts[f] / max(len(valid), 1) * 100 for f in FRAMES]
    rag_pct      = [100 if (rag_result and f in extract_present_frames(rag_result)) else 0
                    for f in FRAMES]
    x_labels = [f.replace("_", " ").title() for f in FRAMES]
    colors   = [FRAME_COLORS[f] for f in FRAMES]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Articles (% present)", x=x_labels, y=article_pct,
        marker_color=[hex_to_rgba(c, 0.5) for c in colors],
        marker_line=dict(color=colors, width=1.5),
    ))
    fig.add_trace(go.Bar(
        name="RAG output (present)", x=x_labels, y=rag_pct,
        marker_color=colors,
        marker_pattern_shape="/",
        marker_pattern_fgcolor="white",
    ))

    fig.update_layout(
        barmode="group",
        title=dict(
            text=f"Articles vs RAG Output | {topic} | {model_name}",
            font=dict(size=15, family="Arial, sans-serif"),
            x=0.01, xanchor="left",
        ),
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(color="#333333", family="Arial, sans-serif"),
        yaxis=dict(
            title="% present", range=[0, 105],
            gridcolor="#eeeeee", gridwidth=1,
            linecolor="#cccccc", linewidth=1,
        ),
        xaxis=dict(linecolor="#cccccc", linewidth=1),
        legend=dict(
            bgcolor="rgba(255,255,255,0.8)", bordercolor="#cccccc", borderwidth=1,
            orientation="h", yanchor="top", y=0.98, xanchor="left", x=0.01,
        ),
        margin=dict(l=50, r=20, t=60, b=40),
        height=400,
    )

    path = os.path.join(out_dir, f"frame_presence_bar_{model_name}.png")
    fig.write_image(path, width=900, height=400, scale=2)
    print(f"  Saved: {path}")

# ── Summary ───────────────────────────────────────────────────────────────────
def print_summary(article_results: list[dict], rag_result: dict):
    valid = [r for r in article_results if r]
    print("\nFRAME SHIFT SUMMARY")
    print("─" * 65)
    for frame in FRAMES:
        art_count = sum(1 for r in valid if frame in extract_present_frames(r))
        art_pct   = art_count / len(valid) * 100 if valid else 0
        rag_has   = rag_result and frame in extract_present_frames(rag_result)
        bar       = "█" * art_count + "░" * (len(valid) - art_count)

        if art_count == 0 and rag_has:
            shift = "← INTRODUCED in RAG"
        elif art_count > 0 and not rag_has:
            shift = "← DROPPED in RAG"
        elif art_count > 0 and rag_has:
            shift = "← RETAINED"
        else:
            shift = ""

        print(f"  {frame.replace('_', ' ').title():<20} [{bar}] {art_pct:4.0f}%  |  "
              f"RAG: {'✓' if rag_has else '✗'}  {shift}")
    print()


def visualize_results(article_results: list[dict], topic: str, rag_result: dict, out_dir: str, model_name: str):
    os.makedirs(out_dir, exist_ok=True)
    print_summary(article_results, rag_result)
    save_sankey(article_results, topic, rag_result, out_dir, model_name)
    save_bar(article_results, topic, rag_result, out_dir, model_name)

STAKEHOLDER_CATEGORIES = [
    "federal government",
    "federal law enforcement",
    "local government",
    "local law enforcement",
    "citizens",
    "immigrants",
    "advocacy groups",
    "economy",
]

STAKEHOLDER_COLORS = {
    "federal government":      "#e74c3c",
    "federal law enforcement": "#c0392b",
    "local government":        "#1abc9c",
    "local law enforcement":   "#16a085",
    "citizens":                "#3498db",
    "immigrants":              "#9b59b6",
    "advocacy groups":         "#2ecc71",
    "economy":                 "#95a5a6",
}

ROLE_COLORS = {
    "hero":    "#f1c40f",
    "victim":  "#e67e22",
    "villain": "#e74c3c",
}


def save_stakeholder_sankey(
    article_results: list[dict],
    topic: str,
    rag_result: dict,
    out_dir: str,
    model_name: str,
    query_idx: int = 0,
):
    os.makedirs(out_dir, exist_ok=True)
    roles = ["hero", "victim", "villain"]

    color_dict = {
        "federal government":      "#e74c3c",  # red — stays
        "federal law enforcement": "#e67e22",  # orange — clearly distinct
        "local government":        "#1abc9c",
        "local law enforcement":   "#16a085",
        "citizens":                "#3498db",
        "immigrants":              "#9b59b6",
        "advocacy groups":         "#2ecc71",
        "economy":                 "#95a5a6",
    }
    role_colors = {
        "hero":    "#f39c12",
        "victim":  "#3498db",
        "villain": "#e74c3c",
    }

    # ── Collect flows ─────────────────────────────────────────────────────────
    # art_flows:  (stakeholder, art_role_node)  -> count
    # rag_flows:  (art_role_node, rag_role_node, originating_stakeholder) -> count
    art_flows: dict[tuple[str, str], int] = {}
    rag_flows: dict[tuple[str, str, str], int] = {}

    for art in article_results:
        for role in roles:
            for stakeholder in art.get("stakeholder_roles", {}).get(role, []):
                art_node = f"{role} [articles]"
                art_flows[(stakeholder, art_node)] = (
                    art_flows.get((stakeholder, art_node), 0) + 1
                )

    if rag_result:
        for role in roles:
            for stakeholder in rag_result.get("stakeholder_roles", {}).get(role, []):
                rag_node = f"{role} [RAG]"
                art_sources = [(s, a) for (s, a) in art_flows if s == stakeholder]
                if art_sources:
                    for (s, art_node) in art_sources:
                        key = (art_node, rag_node, stakeholder)
                        rag_flows[key] = rag_flows.get(key, 0) + 1
                else:
                    art_node = "unattributed [articles]"
                    art_flows[(stakeholder, art_node)] = (
                        art_flows.get((stakeholder, art_node), 0) + 1
                    )
                    key = (art_node, rag_node, stakeholder)
                    rag_flows[key] = rag_flows.get(key, 0) + 1

    if not art_flows:
        print("  No stakeholder data to plot.")
        return

    # ── Build node list ───────────────────────────────────────────────────────
    all_nodes_ordered = []
    seen = set()

    def add_node(n):
        if n not in seen:
            all_nodes_ordered.append(n)
            seen.add(n)

    for (src, tgt) in art_flows:
        add_node(src)
    for (src, tgt) in art_flows:
        add_node(tgt)
    for (src, tgt, _) in rag_flows:
        add_node(src)
    for (src, tgt, _) in rag_flows:
        add_node(tgt)

    node_idx = {n: i for i, n in enumerate(all_nodes_ordered)}

    # ── Node colors ───────────────────────────────────────────────────────────
    def node_color(name: str) -> str:
        base = name.replace(" [articles]", "").replace(" [RAG]", "")
        if base in role_colors:
            return role_colors[base]
        return color_dict.get(base, "#b0bec5")

    node_colors = [node_color(n) for n in all_nodes_ordered]

    # ── Assemble links ────────────────────────────────────────────────────────
    sources, targets, values, link_colors = [], [], [], []

    for (src, tgt), val in art_flows.items():
        sources.append(node_idx[src])
        targets.append(node_idx[tgt])
        values.append(val)
        # Color by stakeholder so you can trace identity left→right
        link_colors.append(hex_to_rgba(color_dict.get(src, "#b0bec5"), 0.4))

    for (src, tgt, stakeholder), val in rag_flows.items():
        sources.append(node_idx[src])
        targets.append(node_idx[tgt])
        values.append(val)
        # Same stakeholder color — now you can trace it through all three columns
        link_colors.append(hex_to_rgba(color_dict.get(stakeholder, "#b0bec5"), 0.4))

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig = go.Figure(go.Sankey(
        arrangement="snap",
        node=dict(
            pad=20,
            thickness=22,
            line=dict(color="white", width=0.5),
            label=all_nodes_ordered,
            color=node_colors,
            hovertemplate="%{label}<extra></extra>",
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            color=link_colors,
            hovertemplate="%{source.label} → %{target.label}: %{value:.1f}<extra></extra>",
        ),
    ))

    fig.update_layout(
        title=dict(
            text=f"Stakeholder Narrative Framing | {topic} | {model_name}",
            font=dict(size=16, family="Arial, sans-serif"),
            x=0.01,
            xanchor="left",
        ),
        paper_bgcolor="white",
        font=dict(color="#333333", size=12, family="Arial, sans-serif"),
        margin=dict(l=20, r=20, t=80, b=20),
        height=600,
    )

    out_path = os.path.join(
        out_dir, f"stakeholder_sankey_q{query_idx + 1}_{model_name}.png"
    )
    fig.write_image(out_path, width=1200, height=600, scale=2)
    print(f"  Saved: {out_path}")

def save_stakeholder_shift_diverging_bar(
    article_results: list[dict],
    topic: str,
    rag_results: list[dict],
    out_dir: str,
    model_name: str,
):
    """
    Diverging bar chart showing the delta (RAG rate - Article rate) per
    stakeholder × role combination.  Positive bars = RAG amplifies the role;
    negative bars = RAG suppresses it.
    """
    import os
    from collections import defaultdict
    import plotly.graph_objects as go
 
    os.makedirs(out_dir, exist_ok=True)
    roles = ["hero", "victim", "villain"]
 
    n_articles = len([a for a in article_results if a])
    n_rag      = len([r for r in rag_results  if r])
 
    color_dict = {
        "federal government":      "#e74c3c",
        "federal law enforcement": "#e67e22",
        "local government":        "#1abc9c",
        "local law enforcement":   "#16a085",
        "citizens":                "#3498db",
        "immigrants":              "#9b59b6",
        "advocacy groups":         "#2ecc71",
        "economy":                 "#95a5a6",
    }
 
    # ── Compute rates ────────────────────────────────────────────────────────
    art_rates: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for art in article_results:
        if not art:
            continue
        for role in roles:
            for stakeholder in art.get("stakeholder_roles", {}).get(role, []):
                art_rates[stakeholder][role] += 1 / n_articles
 
    rag_rates: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for rag in rag_results:
        if not rag:
            continue
        for role in roles:
            for stakeholder in rag.get("stakeholder_roles", {}).get(role, []):
                rag_rates[stakeholder][role] += 1 / n_rag
 
    all_stakeholders = sorted(
        set(list(art_rates.keys()) + list(rag_rates.keys())),
        key=lambda s: list(color_dict.keys()).index(s) if s in color_dict else 999,
    )
 
    # ── Role display config ──────────────────────────────────────────────────
    role_cfg = {
        "hero":    {"label": "Hero",    "col": 0},
        "victim":  {"label": "Victim",  "col": 1},
        "villain": {"label": "Villain", "col": 2},
    }
 
    fig = go.Figure()
 
    # ── One trace per role (so the legend shows roles, not stakeholders) ─────
    # We'll overlay individual colored bars per stakeholder inside each role panel
    # using subplots-style x-offsets on a single axis, grouped by role.
    #
    # Layout: x-axis has 3 clusters (one per role), within each cluster bars
    # are spaced by stakeholder.  y-axis is the delta (-1 … +1 as a fraction).
 
    bar_width = 0.08
    cluster_gap = 0.6          # space between role clusters
    within_gap  = 0.02         # space between bars inside a cluster
 
    n_stakeholders = len(all_stakeholders)
    cluster_width  = n_stakeholders * (bar_width + within_gap)
 
    role_centers = {
        role: i * (cluster_width + cluster_gap)
        for i, role in enumerate(roles)
    }
 
    # Build one trace per stakeholder (each bar is a single-element trace)
    # so that we can colour them independently and share a single legend entry
    # per stakeholder.
    legend_added = set()
 
    for s_idx, stakeholder in enumerate(all_stakeholders):
        color = color_dict.get(stakeholder, "#b0bec5")
        show_legend = stakeholder not in legend_added
        legend_added.add(stakeholder)
 
        xs, ys, texts, hovers = [], [], [], []
 
        for role in roles:
            art_val = art_rates[stakeholder].get(role, 0.0)
            rag_val = rag_rates[stakeholder].get(role, 0.0)
            delta   = rag_val - art_val
 
            if art_val == 0 and rag_val == 0:
                continue
 
            x_pos = (
                role_centers[role]
                - cluster_width / 2
                + s_idx * (bar_width + within_gap)
                + bar_width / 2
            )
 
            xs.append(x_pos)
            ys.append(delta)
            texts.append("")
            hovers.append(
                f"<b>{stakeholder.title()}</b> | {role.title()}<br>"
                f"Articles: {art_val:.0%}<br>"
                f"RAG: {rag_val:.0%}<br>"
                f"Δ: {delta:+.0%}"
            )
 
        if not xs:
            continue
 
        fig.add_trace(go.Bar(
            x=xs,
            y=ys,
            width=bar_width,
            marker_color=[
                color if d >= 0 else hex_to_rgba(color, 0.35)
                for d in ys
            ],
            marker_line=dict(color=color, width=1),
            name=stakeholder.title(),
            legendgroup=stakeholder,
            showlegend=show_legend,
            hovertemplate="%{customdata}<extra></extra>",
            customdata=hovers,
            text=texts,
        ))
 
    # ── Zero reference line ──────────────────────────────────────────────────
    x_min = min(role_centers.values()) - cluster_width / 2 - 0.1
    x_max = max(role_centers.values()) + cluster_width / 2 + 0.1
 
    fig.add_shape(
        type="line",
        x0=x_min, x1=x_max, y0=0, y1=0,
        line=dict(color="#333333", width=1.2),
    )
 
    # ── Role label annotations ───────────────────────────────────────────────
    for role in roles:
        fig.add_annotation(
            x=role_centers[role],
            y=1.06,
            yref="paper",
            text=f"<b>{role.title()}</b>",
            showarrow=False,
            font=dict(size=13, family="Arial, sans-serif", color="#333333"),
            xanchor="center",
        )
 
    # ── Layout ───────────────────────────────────────────────────────────────
    fig.update_layout(
        title=dict(
            text=(
                f"Stakeholder Role Shift (RAG − Articles) | "
                f"{topic} | {model_name}"
            ),
            font=dict(size=15, family="Arial, sans-serif"),
            x=0.01, xanchor="left",
            pad=dict(b=20),
        ),
        barmode="overlay",
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(color="#333333", size=12, family="Arial, sans-serif"),
        xaxis=dict(
            showticklabels=False,
            showgrid=False,
            zeroline=False,
            range=[x_min - 0.15, x_max + 0.05],
        ),
        yaxis=dict(
            title="Δ Rate (RAG − Articles)",
            tickformat=".0%",
            showgrid=True,
            gridcolor="#eeeeee",
            zeroline=False,
            linecolor="#cccccc",
            tick0=0,
            dtick=0.1,
        ),
        legend=dict(
            bgcolor="white",
            bordercolor="#cccccc",
            borderwidth=1,
            yanchor="middle",
            y=0.5,
            xanchor="left",
            x=1.02,
            title=dict(text="Stakeholder"),
        ),
        margin=dict(l=160, r=200, t=80, b=40),
        height=500,
    )
 
    path = os.path.join(out_dir, f"stakeholder_shift_diverging_{model_name}.png")
    fig.write_image(path, width=1100, height=500, scale=2)
    print(f"  Saved: {path}")


def run_visualizations_from_results(
    results_path: str,
    out_dir: str | None = None,
    model_name: str | None = None,
):
    """
    Load a results JSON and regenerate all visualizations:
    - Per-query: frame shift sankey, frame presence bar, stakeholder role bars
    - Aggregated: stakeholder role bars across all queries combined
    """
    with open(results_path) as f:
        data = json.load(f)

    topic           = data["topic"]
    model_name      = model_name or data.get("model_name", "unknown")
    article_results = data["article_results"]
    rag_results     = data["rag_results"]
    queries         = data.get("queries", [])

    article_gf = [a.get("generic_framing", a) for a in article_results]

    print(f"\n{'═'*65}")
    print(f"  Regenerating visualizations for: {os.path.basename(results_path)}")
    print(f"  Topic: {topic}")
    print(f"  Articles: {len(article_results)}  |  Queries: {len(rag_results)}")
    print(f"  Output: {out_dir}")
    print(f"{'═'*65}")

    # ── Per-query visualizations ──────────────────────────────────────────────
    for idx, rag_result in enumerate(rag_results):
        query_label = queries[idx] if idx < len(queries) else f"query_{idx + 1}"
        query_out   = os.path.join(out_dir, f"query_{idx + 1}")
        os.makedirs(query_out, exist_ok=True)

        print(f"\n  ── Query [{idx + 1}]: {query_label[:65]}")

        rag_gf = rag_result.get("generic_framing", rag_result)

        # Existing frame visualizations (Plotly)
        print_summary(article_gf, rag_gf)
        save_sankey(article_gf, topic, rag_gf, query_out, model_name)
        save_bar(article_gf, topic, rag_gf, query_out, model_name)

        save_stakeholder_sankey(
            article_results=article_results,
            topic=topic,
            rag_result=rag_result,
            out_dir=query_out,
            model_name=model_name,
            query_idx=idx,
        )

    save_stakeholder_shift_diverging_bar(
        article_results=article_results,
        topic=topic,
        rag_results=rag_results,
        out_dir=out_dir,
        model_name=model_name,
    )

    print(f"\n  ✓ Done. All visualizations saved to: {out_dir}\n")
