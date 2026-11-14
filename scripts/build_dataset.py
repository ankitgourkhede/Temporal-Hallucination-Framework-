"""Build the temporal benchmark corpus, benchmark CSV and classifier training CSV.

Source of truth: data/benchmark/facts.yaml.

    PYTHONPATH=. python scripts/build_dataset.py            # build + validate
    PYTHONPATH=. python scripts/build_dataset.py --check    # validate only, write nothing

Validation (the build fails on any error):
  * every document's text yields exactly its declared claims under the project's own
    claim extractor, so gold labels cannot drift from extractor behaviour;
  * every benchmark query parses to the intended destination, nationality, category,
    item and requested slot under the project's own query understanding;
  * every gold value is asserted by at least one current primary source.

Conflict type and difficulty are derived from the corpus rather than hand-labelled:
  stale_primary    an older primary source disagrees and was also last verified long ago
                   (an archived page: freshness scoring can see that it is old)
  unrevised_primary an older primary source disagrees but was crawled recently, so its
                   last_verified_ts looks fresh even though its content was never revised
  stale_secondary  a secondary source older than the gold source disagrees
  wrong_secondary  a secondary source at least as recent as the gold source disagrees
  scope_mismatch   a source scoped to other nationalities disagrees (filtered by S1-S3,
                   visible to the S0 baseline)
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from trustrag.online.retrieval.evidence import extract_claims, scope_matches  # noqa: E402

FACTS = ROOT / "data/benchmark/facts.yaml"
CORPUS_DIR = ROOT / "data/benchmark_corpus"
BENCHMARK_CSV = ROOT / "data/benchmark/benchmark.csv"
CLASSIFIER_CSV = ROOT / "data/train/queries_labeled.csv"
DATASHEET = ROOT / "data/benchmark/DATASHEET.md"
CONFIG = ROOT / "trustrag/config.benchmark.yaml"

PRIMARY_TYPES = {"government", "embassy", "intl_body"}
CATEGORY_SLOTS = {
    "visa_policy": ["visa_required", "visa_type", "max_stay_days", "passport_validity_months"],
    "vaccination_requirement": ["vaccine_required", "vaccine_name", "certificate_required"],
    "customs_regulation": ["item_allowed", "duty_free_limit", "declaration_required"],
}
CATEGORY_INTENT = {
    "visa_policy": "travel_entry",
    "vaccination_requirement": "health_safety",
    "customs_regulation": "customs_regulations",
}
CATEGORY_VOLATILITY = {
    "visa_policy": "High",
    "vaccination_requirement": "Medium",
    "customs_regulation": "Low",
}
NATIONALITY_CODES = {
    "India": "IN", "United Kingdom": "GB", "United States": "US",
    "Australia": "AU", "Canada": "CA", "Japan": "JP",
}
NATIONALITY_ADJ = {
    "India": "Indian", "United Kingdom": "British", "United States": "American",
}
SPLIT_RATIOS = (("train", 0.4), ("dev", 0.2), ("test", 0.4))
UNREVISED_WINDOW_DAYS = 60


def days_between(left: str, right: str) -> int:
    from datetime import date

    return abs((date.fromisoformat(left) - date.fromisoformat(right)).days)


# --------------------------------------------------------------------------- values

def parse_value(slot: str, value: Any) -> Any:
    """Turn a YAML gold value into the extractor's normalized representation."""
    if slot == "duty_free_limit":
        amount, code = str(value).split()
        return (float(amount), code.upper())
    return value


def value_to_csv(value: Any) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, tuple):
        return f"{value[0]:g} {value[1]}"
    return str(value)


def same_value(left: Any, right: Any) -> bool:
    if isinstance(left, (tuple, list)) or isinstance(right, (tuple, list)):
        return tuple(left) == tuple(right) if isinstance(left, (tuple, list)) and isinstance(right, (tuple, list)) else False
    return left == right


# --------------------------------------------------------------------------- model

@dataclass
class Doc:
    id: str
    destination: str
    category: str
    source_type: str
    domain: str
    scope: str
    content_date: str
    snapshot_date: str
    title: str
    text: str
    claims: dict[str, Any]          # slot -> value (item_allowed -> {item: value})
    in_force: bool = True           # False for official proposals / announced-but-future rules

    @property
    def is_primary(self) -> bool:
        return self.source_type in PRIMARY_TYPES

    @property
    def url(self) -> str:
        return f"https://{self.domain}/{self.id.replace('_', '-')}"

    def claim_for(self, slot: str, item: str | None) -> Any:
        if slot == "item_allowed":
            return self.claims.get("item_allowed", {}).get(item) if item else None
        return self.claims.get(slot)

    def in_scope_for(self, slot: str, nationality: str) -> bool:
        return scope_matches(slot, self.scope, nationality)


@dataclass
class Question:
    qid: str
    fact_id: str
    destination: str
    nationality: str
    category: str
    slot: str
    item: str | None
    gold: Any
    queries: list[str]
    split: str | None = None        # fixed split (e.g. blind batches are always "test")
    batch: str = "core_v1"
    gold_doc: Doc | None = None
    conflicts: list[tuple[str, Doc]] = field(default_factory=list)


def load(path: Path) -> tuple[dict[str, Any], list[Doc], list[Question]]:
    spec = yaml.safe_load(path.read_text(encoding="utf-8"))
    snapshot = str(spec["current_snapshot"])
    docs: list[Doc] = []
    questions: list[Question] = []
    for destination, block in spec["destinations"].items():
        for raw in block.get("docs", []):
            claims = {}
            for slot, value in (raw.get("claims") or {}).items():
                if slot == "item_allowed":
                    claims[slot] = dict(value)
                else:
                    claims[slot] = parse_value(slot, value)
            docs.append(Doc(
                id=raw["id"], destination=destination, category=raw["category"],
                source_type=raw["source_type"], domain=raw["domain"], scope=str(raw.get("scope", "ALL")),
                content_date=str(raw["content_date"]), snapshot_date=str(raw.get("snapshot_date", snapshot)),
                title=raw["title"], text=" ".join(str(raw["text"]).split()), claims=claims,
                in_force=bool(raw.get("in_force", True)),
            ))
        for fact in block.get("facts", []):
            truth = {slot: parse_value(slot, value) for slot, value in (fact.get("truth") or {}).items()}
            items = fact.get("items") or {}
            for index, q in enumerate(fact["questions"]):
                slot, item = q["slot"], q.get("item")
                gold = items[item] if slot == "item_allowed" else truth[slot]
                suffix = f"{slot}_{item}" if item else slot
                questions.append(Question(
                    qid=f"{fact['id']}__{suffix}".replace("-", "_"), fact_id=fact["id"],
                    destination=destination, nationality=fact["nationality"], category=fact["category"],
                    slot=slot, item=item, gold=gold, queries=list(q["queries"]),
                    split=fact.get("split"), batch=str(fact.get("batch", "core_v1")),
                ))
    ids = [doc.id for doc in docs]
    duplicates = [key for key, count in Counter(ids).items() if count > 1]
    if duplicates:
        raise SystemExit(f"Duplicate document ids: {duplicates}")
    return spec, docs, questions


# --------------------------------------------------------------------------- validation

def all_items() -> list[str]:
    query_rules = yaml.safe_load((ROOT / "trustrag/policies/query.yaml").read_text(encoding="utf-8"))
    return list(query_rules["entities"]["items"])


def validate_docs(docs: list[Doc]) -> list[str]:
    errors = []
    items = all_items()
    for doc in docs:
        slots = CATEGORY_SLOTS[doc.category]
        document = {"document_id": doc.id, "source_domain": doc.domain, "source_type": doc.source_type,
                    "is_primary": doc.is_primary, "nationality_scope": doc.scope}
        chunk = {"chunk_id": doc.id, "text": doc.text}
        extracted: dict[str, Any] = {}
        for item in [None, *items]:
            for claim in extract_claims(chunk=chunk, document=document, slots=slots,
                                        destination=doc.destination, item=item):
                if claim["slot"] == "item_allowed":
                    extracted.setdefault("item_allowed", {})[item] = claim["value_norm"]
                else:
                    extracted[claim["slot"]] = claim["value_norm"]
        declared = doc.claims
        for slot in sorted(set(extracted) | set(declared)):
            got, want = extracted.get(slot), declared.get(slot)
            if slot == "item_allowed":
                if (got or {}) != (want or {}):
                    errors.append(f"{doc.id}: item_allowed extracted {got} but declared {want}")
            elif got is None or want is None or not same_value(got, want):
                errors.append(f"{doc.id}: {slot} extracted {got!r} but declared {want!r}")
        if len(doc.text.split()) > 150:
            errors.append(f"{doc.id}: text longer than one 180-word chunk")
    return errors


def analyse_conflicts(questions: list[Question], docs: list[Doc]) -> list[str]:
    errors = []
    for q in questions:
        relevant = [d for d in docs if d.destination == q.destination and d.category == q.category]
        supporting = [d for d in relevant if d.is_primary and d.in_force and d.in_scope_for(q.slot, q.nationality)
                      and d.claim_for(q.slot, q.item) is not None
                      and same_value(d.claim_for(q.slot, q.item), q.gold)]
        if not supporting:
            errors.append(f"{q.qid}: no in-scope primary source asserts gold value {q.gold!r}")
            continue
        q.gold_doc = max(supporting, key=lambda d: d.snapshot_date)
        for doc in relevant:
            value = doc.claim_for(q.slot, q.item)
            if value is None or same_value(value, q.gold):
                continue
            if not doc.in_scope_for(q.slot, q.nationality):
                kind = "scope_mismatch"
            elif doc.is_primary and not doc.in_force:
                kind = "proposed_change"
            elif doc.is_primary:
                if doc.content_date >= q.gold_doc.content_date:
                    errors.append(f"{q.qid}: current primary {doc.id} contradicts the gold value")
                    continue
                recently_crawled = days_between(doc.snapshot_date, q.gold_doc.snapshot_date) <= UNREVISED_WINDOW_DAYS
                kind = "unrevised_primary" if recently_crawled else "stale_primary"
            elif doc.content_date < q.gold_doc.content_date:
                kind = "stale_secondary"
            else:
                kind = "wrong_secondary"
            q.conflicts.append((kind, doc))
    return errors


def difficulty(q: Question) -> str:
    kinds = {kind for kind, _ in q.conflicts}
    in_scope = [doc for kind, doc in q.conflicts if kind != "scope_mismatch"]
    if not kinds:
        return "easy"
    if kinds & {"stale_primary", "unrevised_primary", "proposed_change"} or len(in_scope) >= 2:
        return "hard"
    return "medium"


def validate_queries(questions: list[Question]) -> list[str]:
    from trustrag.online.query.intent import QueryUnderstanding
    from trustrag.pipeline import TrustRAGPipeline

    analyzer = QueryUnderstanding.from_config(CONFIG)
    # Validate query *authoring* against the deterministic keyword rules only. Trained
    # classifiers are a system under test: their mistakes belong in the results, and
    # rewording benchmark queries to suit them would tune the benchmark to the model.
    analyzer.classifier_models.models.clear()
    errors = []
    for q in questions:
        for text in q.queries:
            policy = analyzer.summary(text, as_of="2026-11-09", live_check=False, mode="benchmark")
            entities = policy.get("entities", {})
            problems = []
            if policy.get("action") != "RETRIEVE":
                problems.append(f"action={policy.get('action')}")
            if policy.get("category") != q.category:
                problems.append(f"category={policy.get('category')}")
            if entities.get("destination") != q.destination:
                problems.append(f"destination={entities.get('destination')}")
            if entities.get("nationality") != q.nationality:
                problems.append(f"nationality={entities.get('nationality')}")
            if q.item and entities.get("item") != q.item:
                problems.append(f"item={entities.get('item')}")
            requested = TrustRAGPipeline._requested_slot(text, policy)
            if requested not in {q.slot, None} or (requested is None and q.slot != "passport_validity_months"):
                problems.append(f"requested_slot={requested}")
            if problems:
                errors.append(f"{q.qid}: {text!r} -> {', '.join(problems)}")
    return errors


# --------------------------------------------------------------------------- outputs

def assign_splits(questions: list[Question], seed: int) -> dict[str, str]:
    """Stratified by category and difficulty; one split per paraphrase group."""
    rng = random.Random(seed)
    strata: dict[tuple[str, str], list[Question]] = defaultdict(list)
    splits = {}
    for q in questions:
        if q.split:
            if q.split not in {"train", "dev", "test"}:
                raise SystemExit(f"{q.qid}: split must be train, dev or test")
            splits[q.qid] = q.split
        else:
            strata[(q.category, difficulty(q))].append(q)
    for key in sorted(strata):
        group = sorted(strata[key], key=lambda q: q.qid)
        rng.shuffle(group)
        n = len(group)
        n_train = round(n * SPLIT_RATIOS[0][1])
        n_dev = round(n * SPLIT_RATIOS[1][1])
        if n >= 2 and n - n_train - n_dev <= 0:
            n_train = max(0, n_train - 1)
        for index, q in enumerate(group):
            splits[q.qid] = "train" if index < n_train else "dev" if index < n_train + n_dev else "test"
    return splits


def gold_answer(q: Question) -> str:
    who = f"{NATIONALITY_ADJ.get(q.nationality, q.nationality)} travellers"
    dest = q.destination
    gold = q.gold
    if q.slot == "max_stay_days":
        return f"{who} can stay in {dest} for up to {gold} days."
    if q.slot == "visa_required":
        return f"{who} {'need' if gold else 'do not need'} a visa for {dest}."
    if q.slot == "visa_type":
        label = {"evisa": "an e-visa", "visa_free": "visa-free entry", "visa_on_arrival": "a visa on arrival",
                 "embassy_visa": "an embassy visa"}[gold]
        return f"{who} enter {dest} with {label}."
    if q.slot == "passport_validity_months":
        return f"Passports must be valid for at least {gold} months on entry to {dest}."
    if q.slot == "vaccine_required":
        return f"A vaccination is {'required' if gold else 'not required'} for {who} entering {dest}."
    if q.slot == "vaccine_name":
        return f"The vaccine checked for {dest} is {gold}."
    if q.slot == "certificate_required":
        return f"A vaccination certificate is {'required' if gold else 'not required'} to enter {dest}."
    if q.slot == "duty_free_limit":
        return f"The duty-free allowance for {dest} is {gold[1]} {gold[0]:,.0f}."
    if q.slot == "declaration_required":
        return f"A customs declaration is {'required' if gold else 'not required'} in {dest}."
    if q.slot == "item_allowed":
        return f"Bringing {q.item} into {dest} is {gold}."
    raise ValueError(q.slot)


def benchmark_rows(questions: list[Question], spec: dict[str, Any], seed: int) -> list[dict[str, str]]:
    splits = assign_splits(questions, seed)
    rows = []
    for q in questions:
        kinds = sorted({kind for kind, _ in q.conflicts})
        for index, text in enumerate(q.queries, start=1):
            rows.append({
                "query_id": f"{q.qid}__p{index}",
                "query": text,
                "nationality": q.nationality,
                "destination": q.destination,
                "category": q.category,
                "volatility": CATEGORY_VOLATILITY[q.category],
                "gold_slot": q.slot,
                "gold_value": value_to_csv(q.gold),
                "gold_answer": gold_answer(q),
                "gold_source_url": q.gold_doc.url,
                "as_of_date": str(spec["as_of_date"]),
                "difficulty": difficulty(q),
                "has_conflict": "1" if kinds else "0",
                "conflict_type": "+".join(kinds) if kinds else "none",
                "paraphrase_group": q.qid,
                "split": splits[q.qid],
                "item": q.item or "",
                "batch": q.batch,
                "conflicting_sources": ";".join(f"{kind}:{doc.id}" for kind, doc in q.conflicts),
            })
    return rows


def write_corpus(docs: list[Doc]) -> None:
    if CORPUS_DIR.exists():
        shutil.rmtree(CORPUS_DIR)
    for doc in docs:
        folder = CORPUS_DIR / doc.destination.lower().replace(" ", "_")
        folder.mkdir(parents=True, exist_ok=True)
        (folder / f"{doc.id}.md").write_text(f"# {doc.title}\n\n{doc.text}\n", encoding="utf-8")
        meta = {
            "doc_key": doc.id, "title": doc.title, "url": doc.url, "source_domain": doc.domain,
            "source_type": doc.source_type, "country": doc.destination, "category": doc.category,
            "nationality_scope": doc.scope, "content_date": doc.content_date,
            "snapshot_date": doc.snapshot_date, "synthetic": True,
        }
        (folder / f"{doc.id}.meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


# --------------------------------------------------------------------------- classifier data

CLASSIFIER_TEMPLATES = {
    ("travel_entry", "visa_policy"): [
        "what are the entry requirements for {dest}",
        "visa rules for {dest} for {nat} passport holders",
        "{nat} passport visa-free countries list includes {dest}?",
        "how do I apply for an e-visa to {dest}",
        "how many days of stay are allowed in {dest}",
        "is there a visa on arrival in {dest}",
        "passport requirements to enter {dest}",
        "what documents do I need for entry to {dest}",
        "can I extend my stay in {dest} after the visa expires",
        "tourist visa processing time for {dest}",
    ],
    ("health_safety", "vaccination_requirement"): [
        "health requirements for travelling to {dest}",
        "which vaccinations should I get before {dest}",
        "is there any outbreak notice for {dest} travellers",
        "covid rules for arrivals in {dest}",
        "vaccination certificate rules in {dest}",
        "do children need a vaccine to travel to {dest}",
        "yellow fever vaccination advice for {dest}",
        "health requirement checklist before flying to {dest}",
    ],
    ("customs_regulations", "customs_regulation"): [
        "customs rules when landing in {dest}",
        "what do I have to declare at {dest} customs",
        "restricted items list for {dest}",
        "duty on electronics bought abroad when entering {dest}",
        "can I carry {item} through customs in {dest}",
        "import rules for {item} in {dest}",
        "is {item} prohibited at {dest} customs",
        "how much duty do I pay on gifts in {dest}",
    ],
    ("out_of_scope", ""): [
        "best beaches in {dest}",
        "cheap hotels in {dest}",
        "what is the weather like in {dest} in December",
        "top restaurants to try in {dest}",
        "things to do in {dest} with kids",
        "flight price from Delhi to {dest}",
        "sightseeing itinerary for 3 days in {dest}",
        "best hotel near the airport in {dest}",
    ],
}
CLASSIFIER_DESTINATIONS = [
    "Vietnam", "Thailand", "Singapore", "Malaysia", "Indonesia", "Japan", "Nepal", "Kenya",
    "Australia", "Canada", "the UK", "the USA", "India", "China",
]
CLASSIFIER_NATIONALITIES = ["Indian", "British", "American", "Australian", "Canadian"]
CLASSIFIER_ITEMS = ["drone", "medicine", "alcohol", "tobacco", "food", "e-cigarette", "cash"]


def classifier_rows(seed: int, per_template: int = 4) -> list[dict[str, str]]:
    rng = random.Random(seed)
    rows = []
    for (intent, category), templates in CLASSIFIER_TEMPLATES.items():
        for t_index, template in enumerate(templates):
            group = f"clf_{intent}_{t_index:02d}"
            for dest in rng.sample(CLASSIFIER_DESTINATIONS, per_template):
                rows.append({
                    "query": template.format(dest=dest, nat=rng.choice(CLASSIFIER_NATIONALITIES),
                                             item=rng.choice(CLASSIFIER_ITEMS)),
                    "intent": intent,
                    "category": category,
                    "paraphrase_of": group,
                })
    return rows


# --------------------------------------------------------------------------- datasheet

def datasheet(docs: list[Doc], questions: list[Question], rows: list[dict[str, str]], clf: list[dict[str, str]]) -> str:
    def table(counter: Counter, header: str) -> str:
        lines = [f"| {header} | count |", "| --- | ---: |"]
        lines += [f"| {key} | {value} |" for key, value in sorted(counter.items())]
        return "\n".join(lines)

    split_q = Counter(r["split"] for r in rows if r["query_id"].endswith("__p1"))
    lines = [
        "# Temporal benchmark datasheet",
        "",
        "Generated by `scripts/build_dataset.py` from `data/benchmark/facts.yaml`. Do not edit by hand.",
        "",
        "**All documents are synthetic.** They are modelled on real travel-policy changes but were not",
        "verified against live government pages; URLs use the reserved `.example` TLD.",
        "",
        "## Size",
        "",
        f"- Corpus documents: {len(docs)} ({sum(d.is_primary for d in docs)} primary, "
        f"{sum(not d.is_primary for d in docs)} secondary) across {len({d.destination for d in docs})} destinations",
        f"- Benchmark questions (paraphrase groups): {len(questions)}; benchmark rows: {len(rows)}",
        f"- Questions per split: " + ", ".join(f"{k} {v}" for k, v in sorted(split_q.items())),
        f"- Non-Indian nationality share of rows: "
        f"{sum(r['nationality'] != 'India' for r in rows) / len(rows):.1%}",
        f"- Classifier training rows: {len(clf)} in {len({r['paraphrase_of'] for r in clf})} paraphrase groups",
        "",
        "## Benchmark rows",
        "",
        table(Counter(r["category"] for r in rows), "category"),
        "",
        table(Counter(r["difficulty"] for r in rows), "difficulty"),
        "",
        table(Counter(r["conflict_type"] for r in rows), "conflict_type"),
        "",
        table(Counter(r["split"] for r in rows), "split"),
        "",
        "## Corpus",
        "",
        table(Counter(d.source_type for d in docs), "source_type"),
        "",
        "## Conflict definitions",
        "",
        "- `stale_primary`: an archived primary page disagrees; its last verification is old, so freshness scoring can see it.",
        "- `unrevised_primary`: an official page whose content was never revised but which was crawled recently;",
        "  primary freshness uses `last_verified_ts`, so it looks fresh. Gold comes from a newer primary notice",
        "  (often an embassy), so authority alone can favour the outdated page.",
        "- `stale_secondary`: a secondary source older than the gold source disagrees (classic temporal hallucination trap).",
        "- `proposed_change`: a recent official page announces a change that is not in force yet",
        "  (`in_force: false` in facts.yaml); the long-standing rule is still the gold answer.",
        "- `wrong_secondary`: a secondary source at least as recent as the gold source disagrees (misinformation, not staleness).",
        "- `scope_mismatch`: a source for other nationalities disagrees; S1-S3 filter it by nationality scope, S0 does not.",
        "",
        "Difficulty: `easy` = no conflicting source; `medium` = one in-scope conflict or only scope conflicts;",
        "`hard` = a stale or unrevised primary source, or two or more in-scope conflicting sources.",
        "",
        "Customs questions use the PRIMARY_ONLY retrieval strategy (low volatility), so S1-S3 never see",
        "secondary customs sources; secondary customs conflicts only test the S0 baseline.",
        "",
        "## Known limitations",
        "",
        "- All questions use one `as_of_date`; retrieval does not yet filter documents dated after `as_of`,",
        "  so point-in-time (time-travel) questions are not included.",
        "- Gold values are only as correct as `facts.yaml`; replace with verified snapshots before claiming",
        "  real-world accuracy.",
        "- Test split is small (see above); report bootstrap CIs and treat McNemar p-values with caution.",
    ]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- main

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="validate only; do not write files")
    parser.add_argument("--skip-query-check", action="store_true", help="skip query-understanding validation")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    spec, docs, questions = load(FACTS)
    errors = validate_docs(docs) + analyse_conflicts(questions, docs)
    if not args.skip_query_check:
        errors += validate_queries(questions)
    if errors:
        print(f"{len(errors)} validation error(s):", *errors, sep="\n  ")
        raise SystemExit(1)

    rows = benchmark_rows(questions, spec, args.seed)
    clf = classifier_rows(args.seed)
    summary = hashlib.sha256(FACTS.read_bytes()).hexdigest()[:12]
    print(f"OK: {len(docs)} documents, {len(questions)} questions, {len(rows)} benchmark rows, "
          f"{len(clf)} classifier rows (facts.yaml sha256 {summary})")
    if args.check:
        return
    write_corpus(docs)
    write_csv(BENCHMARK_CSV, rows)
    write_csv(CLASSIFIER_CSV, clf)
    DATASHEET.write_text(datasheet(docs, questions, rows, clf), encoding="utf-8")
    print(f"Wrote {CORPUS_DIR.relative_to(ROOT)}/, {BENCHMARK_CSV.relative_to(ROOT)}, "
          f"{CLASSIFIER_CSV.relative_to(ROOT)}, {DATASHEET.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
