#!/usr/bin/env python3
"""
YES/NO.EVENTS Vector Search Module
TF-IDF and semantic search for markets and strategies
"""
import os
import sys
import json
import argparse
import math
import re
from pathlib import Path
from collections import defaultdict

# Activate venv
VENV_PATH = Path(__file__).parent / ".venv"
if VENV_PATH.exists():
    for pyver in ["python3.12", "python3.11", "python3.10"]:
        venv_site = VENV_PATH / "lib" / pyver / "site-packages"
        if venv_site.exists() and str(venv_site) not in sys.path:
            sys.path.insert(0, str(venv_site))
            break

BASE = Path(__file__).parent
DATA_DIR = BASE / "data"
VECTORS_DIR = DATA_DIR / "vectors"
STRATEGIES_DIR = BASE / "strategies"

# ══════════════════════════════════════════════════════════════════════════════
# TF-IDF VECTORIZER
# ══════════════════════════════════════════════════════════════════════════════

class TFIDFVectorizer:
    """Simple TF-IDF implementation without external dependencies"""

    def __init__(self):
        self.vocabulary = {}
        self.idf = {}
        self.documents = []
        self.doc_vectors = []

    def tokenize(self, text):
        """Convert text to lowercase tokens"""
        text = text.lower()
        # Remove special chars, keep alphanumeric and hyphens
        text = re.sub(r'[^a-z0-9\-\s]', ' ', text)
        tokens = text.split()
        # Filter short tokens
        return [t for t in tokens if len(t) > 1]

    def fit(self, documents):
        """Build vocabulary and compute IDF from documents"""
        self.documents = documents
        doc_count = len(documents)

        # Build vocabulary
        term_doc_freq = defaultdict(int)
        for doc in documents:
            tokens = set(self.tokenize(doc))
            for token in tokens:
                term_doc_freq[token] += 1

        # Compute IDF
        self.vocabulary = {term: i for i, term in enumerate(term_doc_freq.keys())}
        self.idf = {}
        for term, df in term_doc_freq.items():
            self.idf[term] = math.log((doc_count + 1) / (df + 1)) + 1

        # Compute document vectors
        self.doc_vectors = [self._vectorize(doc) for doc in documents]

    def _vectorize(self, text):
        """Convert text to TF-IDF vector"""
        tokens = self.tokenize(text)
        tf = defaultdict(int)
        for token in tokens:
            tf[token] += 1

        # Normalize TF
        max_tf = max(tf.values()) if tf else 1

        vector = {}
        for term, count in tf.items():
            if term in self.idf:
                vector[term] = (count / max_tf) * self.idf[term]

        return vector

    def transform(self, text):
        """Transform query to vector"""
        return self._vectorize(text)

    def cosine_similarity(self, vec1, vec2):
        """Compute cosine similarity between two sparse vectors"""
        # Get common terms
        common = set(vec1.keys()) & set(vec2.keys())

        if not common:
            return 0.0

        dot = sum(vec1[t] * vec2[t] for t in common)
        norm1 = math.sqrt(sum(v**2 for v in vec1.values()))
        norm2 = math.sqrt(sum(v**2 for v in vec2.values()))

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot / (norm1 * norm2)

    def search(self, query, top_k=10):
        """Search documents by query"""
        query_vec = self.transform(query)

        scores = []
        for i, doc_vec in enumerate(self.doc_vectors):
            sim = self.cosine_similarity(query_vec, doc_vec)
            scores.append((i, sim))

        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)

        return scores[:top_k]

# ══════════════════════════════════════════════════════════════════════════════
# MARKET SEARCH
# ══════════════════════════════════════════════════════════════════════════════

def load_markets():
    """Load current markets from trading module"""
    try:
        from trading import scan_elon_markets
        return scan_elon_markets()
    except Exception as e:
        print(f"Error loading markets: {e}")
        return {}

def build_market_index():
    """Build TF-IDF index for markets"""
    markets = load_markets()

    documents = []
    market_refs = []

    for event, mlist in markets.items():
        total_vol = sum(m['volume'] for m in mlist)

        for m in mlist:
            # Build rich text document for each market
            doc_parts = [
                f"event {event}",
                f"bracket {m['bracket']}",
                f"range {m['bracket']} tweets",
                m['title'],
            ]

            # Add volume descriptors
            vol = m['volume']
            if vol > 500000:
                doc_parts.append("very high volume massive liquidity")
            elif vol > 100000:
                doc_parts.append("high volume good liquidity")
            elif vol > 50000:
                doc_parts.append("medium volume decent liquidity")
            else:
                doc_parts.append("low volume thin liquidity")

            # Add odds descriptors
            odds = m['yes']
            if odds > 0.20:
                doc_parts.append("high probability likely favorite")
            elif odds > 0.10:
                doc_parts.append("moderate probability contender")
            elif odds > 0.05:
                doc_parts.append("low probability underdog")
            else:
                doc_parts.append("very low probability longshot")

            # Add bracket range descriptors
            bracket = m['bracket']
            try:
                parts = bracket.replace('+', '-9999').split('-')
                low = int(parts[0])
                if low < 200:
                    doc_parts.append("low tweets few posts quiet")
                elif low < 400:
                    doc_parts.append("moderate tweets medium activity")
                elif low < 600:
                    doc_parts.append("high tweets busy active")
                else:
                    doc_parts.append("very high tweets extremely active prolific")
            except:
                pass

            documents.append(' '.join(doc_parts))
            market_refs.append({
                'event': event,
                'bracket': m['bracket'],
                'id': m['id'],
                'yes': m['yes'],
                'volume': m['volume'],
                'title': m['title']
            })

    return documents, market_refs

def search_markets(query, top_k=10):
    """Search markets using TF-IDF"""
    documents, market_refs = build_market_index()

    if not documents:
        return []

    vectorizer = TFIDFVectorizer()
    vectorizer.fit(documents)

    results = vectorizer.search(query, top_k)

    output = []
    for idx, score in results:
        if score > 0.01:  # Filter very low scores
            market = market_refs[idx].copy()
            market['score'] = round(score, 4)
            output.append(market)

    return output

# ══════════════════════════════════════════════════════════════════════════════
# STRATEGY SEARCH
# ══════════════════════════════════════════════════════════════════════════════

def load_strategies():
    """Load all strategy files"""
    strategies = []

    for strat_type in ['signals', 'automations', 'backtests', 'models']:
        strat_dir = STRATEGIES_DIR / strat_type
        if not strat_dir.exists():
            continue

        for f in strat_dir.glob("*.md"):
            content = f.read_text()

            # Parse YAML frontmatter
            meta = {}
            body = content
            if content.startswith('---'):
                try:
                    _, fm, body = content.split('---', 2)
                    import yaml
                    meta = yaml.safe_load(fm) or {}
                except:
                    pass

            strategies.append({
                'path': str(f.relative_to(STRATEGIES_DIR)),
                'type': strat_type,
                'name': meta.get('name', f.stem),
                'status': meta.get('status', 'unknown'),
                'content': body,
                'meta': meta
            })

    return strategies

def build_strategy_index():
    """Build TF-IDF index for strategies"""
    strategies = load_strategies()

    documents = []
    strat_refs = []

    for s in strategies:
        # Build searchable document
        doc_parts = [
            s['name'],
            s['type'],
            s.get('meta', {}).get('markets', ''),
            s['content'][:2000],  # Limit content
        ]

        # Add status descriptors
        status = s['status']
        if status == 'live':
            doc_parts.append("active production running deployed")
        elif status == 'testing':
            doc_parts.append("beta experimental testing validation")
        elif status == 'draft':
            doc_parts.append("draft incomplete work in progress")

        documents.append(' '.join(doc_parts))
        strat_refs.append({
            'path': s['path'],
            'name': s['name'],
            'type': s['type'],
            'status': s['status']
        })

    return documents, strat_refs

def search_strategies(query, top_k=10):
    """Search strategies using TF-IDF"""
    documents, strat_refs = build_strategy_index()

    if not documents:
        return []

    vectorizer = TFIDFVectorizer()
    vectorizer.fit(documents)

    results = vectorizer.search(query, top_k)

    output = []
    for idx, score in results:
        if score > 0.01:
            strat = strat_refs[idx].copy()
            strat['score'] = round(score, 4)
            output.append(strat)

    return output

# ══════════════════════════════════════════════════════════════════════════════
# SIMILAR MARKETS
# ══════════════════════════════════════════════════════════════════════════════

def find_similar_markets(market_id, top_k=5):
    """Find markets similar to given market ID"""
    documents, market_refs = build_market_index()

    # Find target market
    target_idx = None
    for i, m in enumerate(market_refs):
        if m['id'] == market_id:
            target_idx = i
            break

    if target_idx is None:
        return []

    vectorizer = TFIDFVectorizer()
    vectorizer.fit(documents)

    target_vec = vectorizer.doc_vectors[target_idx]

    scores = []
    for i, doc_vec in enumerate(vectorizer.doc_vectors):
        if i != target_idx:
            sim = vectorizer.cosine_similarity(target_vec, doc_vec)
            scores.append((i, sim))

    scores.sort(key=lambda x: x[1], reverse=True)

    output = []
    for idx, score in scores[:top_k]:
        if score > 0.1:
            market = market_refs[idx].copy()
            market['similarity'] = round(score, 4)
            output.append(market)

    return output

# ══════════════════════════════════════════════════════════════════════════════
# SEMANTIC SEARCH (Keyword Expansion)
# ══════════════════════════════════════════════════════════════════════════════

SEMANTIC_SYNONYMS = {
    # Volume terms
    'liquid': ['volume', 'liquidity', 'depth', 'thick'],
    'thin': ['low volume', 'illiquid', 'sparse'],
    'active': ['high volume', 'busy', 'popular'],

    # Probability terms
    'favorite': ['high probability', 'likely', 'leading'],
    'longshot': ['low probability', 'unlikely', 'underdog'],
    'contender': ['moderate probability', 'possible'],

    # Time terms
    'weekend': ['saturday', 'sunday', 'wknd'],
    'weekday': ['monday', 'tuesday', 'wednesday', 'thursday', 'friday'],

    # Strategy terms
    'momentum': ['trend', 'direction', 'movement'],
    'reversal': ['correction', 'pullback', 'mean reversion'],
    'breakout': ['surge', 'spike', 'jump'],
    'signal': ['entry', 'trigger', 'alert'],
    'exit': ['close', 'sell', 'take profit'],
    'position': ['sizing', 'allocation', 'bet size'],

    # Analysis terms
    'edge': ['advantage', 'ev', 'expected value', 'mispricing'],
    'arbitrage': ['arb', 'risk free', 'guaranteed'],
}

def expand_query(query):
    """Expand query with semantic synonyms"""
    expanded = [query]
    query_lower = query.lower()

    for term, synonyms in SEMANTIC_SYNONYMS.items():
        if term in query_lower:
            expanded.extend(synonyms)
        for syn in synonyms:
            if syn in query_lower:
                expanded.append(term)
                expanded.extend(s for s in synonyms if s != syn)

    return ' '.join(expanded)

def semantic_search_markets(query, top_k=10):
    """Search markets with semantic expansion"""
    expanded = expand_query(query)
    return search_markets(expanded, top_k)

def semantic_search_strategies(query, top_k=10):
    """Search strategies with semantic expansion"""
    expanded = expand_query(query)
    return search_strategies(expanded, top_k)

# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def print_market_results(results, title="Search Results"):
    """Pretty print market search results"""
    print(f"\n{'═'*60}")
    print(f" {title}")
    print(f"{'═'*60}")

    if not results:
        print(" No results found")
        return

    print(f"\n {'Event':<12} {'Bracket':<12} {'Odds':>8} {'Volume':>10} {'Score':>8}")
    print(f" {'-'*54}")

    for r in results:
        odds = r['yes'] * 100
        vol_k = r['volume'] / 1000
        score = r.get('score', r.get('similarity', 0))
        print(f" {r['event']:<12} {r['bracket']:<12} {odds:>7.1f}% ${vol_k:>8.0f}k {score:>7.3f}")

    print()

def print_strategy_results(results, title="Strategy Search"):
    """Pretty print strategy search results"""
    print(f"\n{'═'*60}")
    print(f" {title}")
    print(f"{'═'*60}")

    if not results:
        print(" No strategies found")
        return

    status_icons = {'draft': '○', 'testing': '◐', 'live': '●', 'archived': '◌', 'unknown': '?'}

    print(f"\n {'Type':<12} {'Name':<25} {'Status':<10} {'Score':>8}")
    print(f" {'-'*58}")

    for r in results:
        icon = status_icons.get(r['status'], '?')
        print(f" {r['type']:<12} {r['name']:<25} {icon} {r['status']:<8} {r['score']:>7.3f}")

    print()

def main():
    parser = argparse.ArgumentParser(
        description="YES/NO.EVENTS Vector Search",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  ./search.sh markets "high volume 500 tweets"
  ./search.sh markets "weekend low odds"
  ./search.sh strats "momentum entry signal"
  ./search.sh strats "kelly position sizing"
  ./search.sh similar 1148943
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Markets search
    markets_parser = subparsers.add_parser("markets", help="Search markets")
    markets_parser.add_argument("query", help="Search query")
    markets_parser.add_argument("--top", "-n", type=int, default=10, help="Number of results")
    markets_parser.add_argument("--semantic", "-s", action="store_true", help="Use semantic expansion")
    markets_parser.add_argument("--json", action="store_true", help="JSON output")

    # Strategies search
    strats_parser = subparsers.add_parser("strats", help="Search strategies")
    strats_parser.add_argument("query", help="Search query")
    strats_parser.add_argument("--top", "-n", type=int, default=10, help="Number of results")
    strats_parser.add_argument("--semantic", "-s", action="store_true", help="Use semantic expansion")
    strats_parser.add_argument("--json", action="store_true", help="JSON output")

    # Similar markets
    similar_parser = subparsers.add_parser("similar", help="Find similar markets")
    similar_parser.add_argument("market_id", help="Market ID to compare")
    similar_parser.add_argument("--top", "-n", type=int, default=5, help="Number of results")
    similar_parser.add_argument("--json", action="store_true", help="JSON output")

    # Index info
    subparsers.add_parser("info", help="Show index statistics")

    args = parser.parse_args()

    try:
        if args.command == "markets":
            if args.semantic:
                results = semantic_search_markets(args.query, args.top)
            else:
                results = search_markets(args.query, args.top)

            if args.json:
                print(json.dumps(results, indent=2))
            else:
                print_market_results(results, f"Markets: '{args.query}'")

        elif args.command == "strats":
            if args.semantic:
                results = semantic_search_strategies(args.query, args.top)
            else:
                results = search_strategies(args.query, args.top)

            if args.json:
                print(json.dumps(results, indent=2))
            else:
                print_strategy_results(results, f"Strategies: '{args.query}'")

        elif args.command == "similar":
            results = find_similar_markets(args.market_id, args.top)

            if args.json:
                print(json.dumps(results, indent=2))
            else:
                print_market_results(results, f"Similar to {args.market_id}")

        elif args.command == "info":
            docs, refs = build_market_index()
            strat_docs, strat_refs = build_strategy_index()

            print(f"\n{'═'*40}")
            print(f" Vector Search Index Info")
            print(f"{'═'*40}")
            print(f"\n Markets indexed: {len(refs)}")
            print(f" Strategies indexed: {len(strat_refs)}")
            print(f"\n Events:")

            events = defaultdict(int)
            for r in refs:
                events[r['event']] += 1
            for e, c in sorted(events.items()):
                print(f"   {e}: {c} brackets")

            print(f"\n Strategy types:")
            types = defaultdict(int)
            for s in strat_refs:
                types[s['type']] += 1
            for t, c in sorted(types.items()):
                print(f"   {t}: {c} files")
            print()

        else:
            parser.print_help()

    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0

if __name__ == "__main__":
    exit(main())
