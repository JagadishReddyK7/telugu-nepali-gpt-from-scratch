"""
Synthetic comparative-reasoning dataset generator (Phase 3.1).

    python -m telugu.reasoning.generate
    python -m nepali.reasoning.generate

This module is the language-agnostic *engine*. Everything language-specific --
entity names, attribute vocabulary, sentence templates, answer words, digit
script -- lives in ``<lang>/reasoning/lexicon.py``. The two languages share the
generation logic exactly as they share ``common/model.py``: the code is common,
the data is not. No name, template or example crosses between the two corpora.

WHY SYNTHETIC, AND WHY TEMPLATES
--------------------------------
The spec requires a reasoning set whose ground truth is *known*, which rules out
scraping. Every example here is built from a random total order over entities
(or random numeric values), rendered into premises through templates, and the
answer is computed from the order -- so the label is correct by construction,
and ``tests/test_reasoning.py`` re-derives it independently.

FIVE TASK FAMILIES
------------------
    superlative_chain   A > B, B > C            -> who is largest / smallest?
    numeric_compare     values for 2-3 entities -> who is larger / smaller / equal?
    chain_relation      A > B, B > C            -> relation between A and C?
                        A > B, A > C            -> relation between B and C?  (undetermined)
    yes_no              A > B, B > C            -> is A larger than C?  yes / no
    ordering            A > B, B > C            -> full order, largest to smallest

Each premise chain has 3 or 4 entities (2-3 hops). Premises are presented in
shuffled order and each relation is randomly rendered as "X is more than Y" or
"Y is less than X", so the model cannot read the answer off surface order.

HOW TRAIN / TEST LEAKAGE IS AVOIDED
-----------------------------------
1. **Held-out entity names.** Names and object nouns are partitioned into three
   disjoint pools. A test example never mentions an entity seen in training.
2. **Held-out templates.** Every template family with two or more phrasings
   reserves its last phrasing for the test split alone. Half of the test set
   uses a held-out phrasing and is flagged ``template_held_out`` so accuracy on
   seen vs unseen wording can be reported separately.
3. **Exact deduplication** of prompt strings across all three splits.
4. **Deterministic seeding**, so the released scripts regenerate the identical
   dataset.
"""

from __future__ import annotations

import itertools
import json
import logging
import random
import unicodedata
from dataclasses import dataclass, field, asdict
from pathlib import Path

logger = logging.getLogger(__name__)

TASKS = ("superlative_chain", "numeric_compare", "chain_relation", "yes_no",
         "ordering")

DEVANAGARI_DIGITS = str.maketrans("0123456789", "०१२३४५६७८९")


def render_number(n: int, digits: str) -> str:
    """Render an integer in the lexicon's digit script.

    Args:
        n: The number.
        digits: ``"ascii"`` or ``"devanagari"``.

    Returns:
        The digit string.
    """
    s = str(n)
    return s.translate(DEVANAGARI_DIGITS) if digits == "devanagari" else s


def normalize_answer(text: str) -> str:
    """Canonical form used for exact-match comparison.

    NFC-normalizes, strips, collapses whitespace, and drops a trailing
    sentence terminator (``.`` or the Devanagari danda ``।``) so that a model
    answering "राम।" is not marked wrong against "राम".

    Args:
        text: Raw answer text.

    Returns:
        The normalized string.
    """
    t = unicodedata.normalize("NFC", text).strip()
    t = " ".join(t.split())
    while t and t[-1] in ".।":
        t = t[:-1].rstrip()
    return t


@dataclass
class Example:
    """One reasoning example, JSON-serializable.

    Attributes:
        id: Stable identifier ``<split>-<index>``.
        split: ``train`` / ``validation`` / ``test``.
        task: One of :data:`TASKS`.
        attribute: Attribute key from the lexicon (``height``, ``price`` ...).
        n_entities: Entities mentioned.
        hops: Number of relations that must be chained to answer.
        entities: The entity names, in the order they were assigned.
        prompt: Premises + question + answer prefix. The model's input.
        answer: Gold answer, exactly as the model should emit it.
        choices: Candidate answers for the log-likelihood multiple-choice
            evaluation; always contains ``answer``.
        template_held_out: True if any template used is test-only.
        premise_order: The hidden total order, largest first (for auditing).
    """

    id: str
    split: str
    task: str
    attribute: str
    n_entities: int
    hops: int
    entities: list[str]
    prompt: str
    answer: str
    choices: list[str]
    template_held_out: bool
    premise_order: list[str] = field(default_factory=list)


class TemplatePicker:
    """Chooses a phrasing from a template family, honouring the held-out rule.

    The *last* entry of any family with two or more phrasings is the held-out
    one. For train / validation it is never used. For test, the picker decides
    ONCE per example (with probability ``held_out_rate``) whether the example is
    a held-out-template example; if so, every family with a reserved phrasing
    uses it, so the example's wording is entirely unseen rather than a mix.

    Args:
        rng: Random generator.
        split: Which split is being generated.
        held_out_rate: Fraction of test examples built from held-out phrasings.
    """

    def __init__(self, rng: random.Random, split: str,
                 held_out_rate: float = 0.5):
        self.rng = rng
        self.split = split
        self.held_out_mode = (split == "test" and rng.random() < held_out_rate)
        self.used_held_out = False

    def pick(self, family: list[str]) -> str:
        """Return one phrasing from ``family``."""
        if len(family) == 1:
            return family[0]
        seen, held = family[:-1], family[-1]
        if self.held_out_mode:
            self.used_held_out = True
            return held
        return self.rng.choice(seen)


class ReasoningGenerator:
    """Builds the three splits for one language from its lexicon.

    Args:
        lexicon: The language's lexicon dict (see ``<lang>/reasoning/lexicon.py``
            for the schema).
        seed: Master seed; every split derives its own generator from it.
    """

    def __init__(self, lexicon: dict, seed: int):
        self.lex = lexicon
        self.seed = seed
        self.pools = self._split_entities()
        self._seen_prompts: set[str] = set()

    # ------------------------------------------------------------ entities

    def _split_entities(self) -> dict[str, dict[str, list[str]]]:
        """Partition names and objects into disjoint train/val/test pools.

        Roughly 65% / 15% / 20%. The split is a deterministic function of the
        seed, so the pools are reproducible and are recorded in ``stats.json``.
        """
        rng = random.Random(self.seed)
        out: dict[str, dict[str, list[str]]] = {}
        for kind in ("names", "objects"):
            items = list(dict.fromkeys(self.lex[kind]))   # dedupe, keep order
            rng.shuffle(items)
            n = len(items)
            n_test = max(4, int(round(0.20 * n)))
            n_val = max(4, int(round(0.15 * n)))
            out[kind] = {
                "test": items[:n_test],
                "validation": items[n_test:n_test + n_val],
                "train": items[n_test + n_val:],
            }
        return out

    def _entities(self, rng: random.Random, split: str, domain: str,
                  n: int) -> list[str]:
        """Sample ``n`` distinct entities of ``domain`` from the split's pool."""
        kind = "names" if domain == "person" else "objects"
        pool = self.pools[kind][split]
        # Compound names contain simple ones ("రామ" in "రామకృష్ణ", "राम" in
        # "राम प्रसाद"). Two entities in one example must not nest, or the
        # premises become ambiguous to read; resample until none does.
        for _ in range(100):
            ents = rng.sample(pool, n)
            if not any(a != b and a in b for a in ents for b in ents):
                return ents
        raise RuntimeError("could not sample non-nested entities")

    # ------------------------------------------------------------- helpers

    def _attr(self, rng: random.Random, tasks_needing_values: bool = False) -> tuple[str, dict]:
        """Pick a random attribute; optionally only ones with numeric values."""
        keys = [k for k, a in self.lex["attributes"].items()
                if not tasks_needing_values or "value" in a]
        k = rng.choice(keys)
        return k, self.lex["attributes"][k]

    def _relation_sentence(self, tp: TemplatePicker, rng: random.Random,
                           attr: dict, bigger: str, smaller: str) -> str:
        """Render 'bigger > smaller' as either a 'more' or a 'less' sentence."""
        if rng.random() < 0.5:
            return tp.pick(attr["rel_more"]).format(a=bigger, b=smaller)
        return tp.pick(attr["rel_less"]).format(a=smaller, b=bigger)

    def _chain_premises(self, tp: TemplatePicker, rng: random.Random,
                        attr: dict, order: list[str]) -> list[str]:
        """Adjacent-pair relations for a total order, shuffled."""
        sents = [self._relation_sentence(tp, rng, attr, order[i], order[i + 1])
                 for i in range(len(order) - 1)]
        rng.shuffle(sents)
        return sents

    def _compose(self, tp: TemplatePicker, rng: random.Random,
                 premises: list[str], question: str) -> str:
        """Join intro (optional), premises and question into the prompt body."""
        intro = tp.pick(self.lex["intro"])
        parts = ([intro] if intro else []) + premises + [question]
        return " ".join(p.strip() for p in parts if p.strip())

    def _finish(self, split: str, idx: int, task: str, attr_key: str,
                order: list[str], entities: list[str], hops: int, body: str,
                answer: str, choices: list[str], tp: TemplatePicker) -> Example | None:
        """Assemble an :class:`Example`, or None if the prompt is a duplicate."""
        prompt = f"{body} {self.lex['answer_prefix']}"
        key = normalize_answer(prompt)
        if key in self._seen_prompts:
            return None
        self._seen_prompts.add(key)
        # Gold answer must be among the choices, and choices are unique.
        choices = list(dict.fromkeys([answer] + [c for c in choices if c != answer]))
        return Example(id=f"{split}-{idx}", split=split, task=task,
                       attribute=attr_key, n_entities=len(entities), hops=hops,
                       entities=entities, prompt=prompt, answer=answer,
                       choices=choices, template_held_out=tp.used_held_out,
                       premise_order=order)

    # --------------------------------------------------------------- tasks

    def gen_superlative_chain(self, rng, split, idx) -> Example | None:
        """A > B, B > C  ->  who is the largest / smallest?"""
        tp = TemplatePicker(rng, split)
        akey, attr = self._attr(rng)
        n = rng.choice([3, 3, 4])
        order = self._entities(rng, split, attr["domain"], n)
        premises = self._chain_premises(tp, rng, attr, order)
        if rng.random() < 0.5:
            q, ans = tp.pick(attr["q_most"]), order[0]
        else:
            q, ans = tp.pick(attr["q_least"]), order[-1]
        body = self._compose(tp, rng, premises, q)
        return self._finish(split, idx, "superlative_chain", akey, order, order,
                            n - 1, body, ans, list(order), tp)

    def gen_numeric_compare(self, rng, split, idx) -> Example | None:
        """Explicit values for 2-3 entities -> larger / smaller / equal."""
        tp = TemplatePicker(rng, split)
        akey, attr = self._attr(rng, tasks_needing_values=True)
        n = rng.choice([2, 2, 3])
        ents = self._entities(rng, split, attr["domain"], n)
        lo, hi = attr["value_range"]
        digits = self.lex["digits"]

        if n == 2 and rng.random() < 0.15:
            v = rng.randint(lo, hi)
            values = [v, v]                                   # a tie
        else:
            values = rng.sample(range(lo, hi + 1), n)        # distinct
        premises = [tp.pick(attr["value"]).format(a=e, v=render_number(v, digits))
                    for e, v in zip(ents, values)]
        rng.shuffle(premises)

        ranked = sorted(zip(values, ents), reverse=True)
        order = [e for _, e in ranked]
        if n == 2:
            a, b = ents
            want_more = rng.random() < 0.5
            q = tp.pick(attr["q_pair_more"] if want_more else attr["q_pair_less"]
                        ).format(a=a, b=b)
            if values[0] == values[1]:
                ans = self.lex["equal"]
            else:
                ans = order[0] if want_more else order[-1]
            choices = [a, b, self.lex["equal"]]
        else:
            want_more = rng.random() < 0.5
            q = tp.pick(attr["q_most"] if want_more else attr["q_least"])
            ans = order[0] if want_more else order[-1]
            choices = list(ents)
        body = self._compose(tp, rng, premises, q)
        return self._finish(split, idx, "numeric_compare", akey, order, ents,
                            1, body, ans, choices, tp)

    def gen_chain_relation(self, rng, split, idx) -> Example | None:
        """Relation between the two endpoints of a chain, or undetermined."""
        tp = TemplatePicker(rng, split)
        akey, attr = self._attr(rng)
        ents = self._entities(rng, split, attr["domain"], 3)
        x, y = None, None

        if rng.random() < 0.25:
            # Branching structure: A > B and A > C (or B > A and C > A). The
            # relation between B and C is NOT determined by the premises. A
            # model that has learned "chain the inequalities" must recognise
            # that there is no chain here.
            top, b, c = ents
            if rng.random() < 0.5:
                premises = [self._relation_sentence(tp, rng, attr, top, b),
                            self._relation_sentence(tp, rng, attr, top, c)]
                order = [top]           # only the apex is fixed
            else:
                premises = [self._relation_sentence(tp, rng, attr, b, top),
                            self._relation_sentence(tp, rng, attr, c, top)]
                order = [top]
            rng.shuffle(premises)
            x, y = (b, c) if rng.random() < 0.5 else (c, b)
            ans = self.lex["undetermined"]
            hops = 2
        else:
            order = list(ents)
            premises = self._chain_premises(tp, rng, attr, order)
            a, c = order[0], order[-1]
            # Ask in either direction so the model must invert when needed.
            if rng.random() < 0.5:
                x, y, ans = a, c, attr["ans_more"].format(a=a, b=c)
            else:
                x, y, ans = c, a, attr["ans_less"].format(a=c, b=a)
            hops = 2

        q = tp.pick(attr["q_relation"]).format(a=x, b=y)
        choices = [attr["ans_more"].format(a=x, b=y),
                   attr["ans_less"].format(a=x, b=y),
                   self.lex["undetermined"]]
        body = self._compose(tp, rng, premises, q)
        return self._finish(split, idx, "chain_relation", akey, order, ents,
                            hops, body, ans, choices, tp)

    def gen_yes_no(self, rng, split, idx) -> Example | None:
        """Is X larger than Y?  ->  yes / no, over a 3- or 4-entity chain."""
        tp = TemplatePicker(rng, split)
        akey, attr = self._attr(rng)
        n = rng.choice([3, 3, 4])
        order = self._entities(rng, split, attr["domain"], n)
        premises = self._chain_premises(tp, rng, attr, order)

        i, j = rng.sample(range(n), 2)          # any pair, either direction
        x, y = order[i], order[j]
        truth = i < j                            # x precedes y  <=>  x is larger
        want_more = rng.random() < 0.5
        q = tp.pick(attr["q_yesno_more"] if want_more else attr["q_yesno_less"]
                    ).format(a=x, b=y)
        is_true = truth if want_more else (not truth)
        ans = self.lex["yes"] if is_true else self.lex["no"]
        body = self._compose(tp, rng, premises, q)
        return self._finish(split, idx, "yes_no", akey, order, order,
                            abs(i - j), body, ans, [self.lex["yes"], self.lex["no"]], tp)

    def gen_ordering(self, rng, split, idx) -> Example | None:
        """Full ordering of a chain, largest-first or smallest-first."""
        tp = TemplatePicker(rng, split)
        akey, attr = self._attr(rng)
        n = rng.choice([3, 3, 4])
        order = self._entities(rng, split, attr["domain"], n)
        premises = self._chain_premises(tp, rng, attr, order)
        sep = self.lex["list_separator"]
        if rng.random() < 0.5:
            q, ans_list = tp.pick(attr["q_order_desc"]), order
        else:
            q, ans_list = tp.pick(attr["q_order_asc"]), order[::-1]
        ans = sep.join(ans_list)
        choices = [sep.join(p) for p in itertools.permutations(order)]
        body = self._compose(tp, rng, premises, q)
        return self._finish(split, idx, "ordering", akey, order, order,
                            n - 1, body, ans, choices, tp)

    # ---------------------------------------------------------------- driver

    def generate_split(self, split: str, n: int) -> list[Example]:
        """Generate ``n`` unique examples for one split, tasks balanced.

        Args:
            split: ``train`` / ``validation`` / ``test``.
            n: Target number of examples.

        Returns:
            The examples, in generation order.
        """
        offset = {"train": 0, "validation": 1, "test": 2}[split]
        rng = random.Random(self.seed * 1000 + offset)
        gens = {
            "superlative_chain": self.gen_superlative_chain,
            "numeric_compare": self.gen_numeric_compare,
            "chain_relation": self.gen_chain_relation,
            "yes_no": self.gen_yes_no,
            "ordering": self.gen_ordering,
        }
        out: list[Example] = []
        attempts = 0
        while len(out) < n:
            task = TASKS[len(out) % len(TASKS)]      # round-robin -> balanced
            ex = gens[task](rng, split, len(out))
            attempts += 1
            if ex is not None:
                out.append(ex)
            if attempts > 50 * n:
                raise RuntimeError(
                    f"{split}: could not reach {n} unique examples after "
                    f"{attempts} attempts -- entity pool too small")
        logger.info("%s: %d examples (%d attempts, %d duplicates skipped)",
                    split, len(out), attempts, attempts - len(out))
        return out


# ---------------------------------------------------------------- statistics

def dataset_stats(splits: dict[str, list[Example]], pools: dict,
                  lexicon: dict, tokenizer=None) -> dict:
    """Summaries for the report: sizes, task mix, template variety, lengths.

    Args:
        splits: Mapping split name -> examples.
        pools: Entity pools from :class:`ReasoningGenerator`.
        lexicon: The lexicon, for counting templates.
        tokenizer: Optional SentencePieceProcessor for token-length stats.

    Returns:
        JSON-serializable dict.
    """
    def count(examples, key):
        c: dict = {}
        for e in examples:
            k = str(getattr(e, key))
            c[k] = c.get(k, 0) + 1
        return dict(sorted(c.items()))

    n_templates = 0
    n_held_out = 0
    for attr in lexicon["attributes"].values():
        for k, v in attr.items():
            if isinstance(v, list):
                n_templates += len(v)
                n_held_out += int(len(v) >= 2)
    n_templates += len(lexicon["intro"])
    n_held_out += int(len(lexicon["intro"]) >= 2)

    out = {
        "seed": None,
        "sizes": {s: len(v) for s, v in splits.items()},
        "entity_pools": {k: {s: len(v) for s, v in p.items()}
                         for k, p in pools.items()},
        "entity_pools_disjoint": all(
            not (set(p["train"]) & set(p["test"]))
            and not (set(p["train"]) & set(p["validation"]))
            and not (set(p["validation"]) & set(p["test"]))
            for p in pools.values()),
        "attributes": list(lexicon["attributes"].keys()),
        "templates_total": n_templates,
        "template_families_with_held_out_phrasing": n_held_out,
        "per_split": {},
    }
    for s, exs in splits.items():
        d = {
            "task": count(exs, "task"),
            "attribute": count(exs, "attribute"),
            "n_entities": count(exs, "n_entities"),
            "hops": count(exs, "hops"),
            "template_held_out": count(exs, "template_held_out"),
            "answer_kinds": count_answer_kinds(exs, lexicon),
            "mean_prompt_chars": round(sum(len(e.prompt) for e in exs) / len(exs), 1),
            "mean_answer_chars": round(sum(len(e.answer) for e in exs) / len(exs), 1),
        }
        if tokenizer is not None:
            pl = [len(tokenizer.encode(e.prompt)) for e in exs]
            al = [len(tokenizer.encode(e.answer)) for e in exs]
            d["prompt_tokens"] = {"mean": round(sum(pl) / len(pl), 1),
                                  "max": max(pl), "min": min(pl)}
            d["answer_tokens"] = {"mean": round(sum(al) / len(al), 1),
                                  "max": max(al), "min": min(al)}
            d["total_tokens_with_bos_eos"] = sum(pl) + sum(al) + 2 * len(exs)
        out["per_split"][s] = d
    return out


def count_answer_kinds(exs: list[Example], lexicon: dict) -> dict:
    """How many answers are names vs yes/no vs equal vs undetermined vs lists."""
    kinds: dict[str, int] = {}
    special = {lexicon["yes"]: "yes", lexicon["no"]: "no",
               lexicon["equal"]: "equal", lexicon["undetermined"]: "undetermined"}
    for e in exs:
        if e.answer in special:
            k = special[e.answer]
        elif lexicon["list_separator"] in e.answer:
            k = "ordering_list"
        elif e.answer in e.entities:
            k = "entity_name"
        else:
            k = "relation_sentence"
        kinds[k] = kinds.get(k, 0) + 1
    return dict(sorted(kinds.items()))


# ------------------------------------------------------------------------ io

def write_jsonl(path: Path, examples: list[Example]) -> None:
    """Write examples as one JSON object per line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for e in examples:
            f.write(json.dumps(asdict(e), ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    """Read a JSONL file of examples."""
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_samples_md(path: Path, splits: dict[str, list[Example]],
                     language: str, per_task: int = 3) -> None:
    """Write a Markdown file of example prompts per task, for the report.

    Args:
        path: Destination.
        splits: Split name -> examples.
        language: Language label for the heading.
        per_task: Examples shown per task per split.
    """
    L = [f"# Synthetic reasoning data — {language}", "",
         "Generated by `common/reasoning_gen.py` with this language's lexicon.",
         "Each block shows the model's input (`prompt`) and the gold `answer`.",
         "Test examples use held-out entity names; those marked *held-out",
         "template* also use a phrasing never seen in training.", ""]
    for split, exs in splits.items():
        L += [f"## {split}", ""]
        for task in TASKS:
            shown = [e for e in exs if e.task == task][:per_task]
            L += [f"### {task}", ""]
            for e in shown:
                flag = " *(held-out template)*" if e.template_held_out else ""
                L += [f"- **{e.id}**{flag} · {e.attribute} · {e.n_entities} entities · {e.hops} hops",
                      f"  - prompt: {e.prompt}",
                      f"  - answer: **{e.answer}**", ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


def build_dataset(lexicon: dict, language: str, out_dir: Path, seed: int,
                  sizes: dict[str, int], tokenizer_path: str | None = None,
                  report_dir: Path | None = None) -> dict:
    """Generate, write and summarize all three splits for one language.

    Args:
        lexicon: The language's lexicon.
        language: Language name, used in file stems and headings.
        out_dir: Where ``{train,validation,test}.jsonl`` and ``stats.json`` go.
        seed: Master seed.
        sizes: Split name -> example count.
        tokenizer_path: Optional SentencePiece model for token-length stats.
        report_dir: If given, ``stats.json`` and ``samples.md`` are copied here.

    Returns:
        The stats dict.
    """
    gen = ReasoningGenerator(lexicon, seed)
    splits = {s: gen.generate_split(s, n) for s, n in sizes.items()}

    sp = None
    if tokenizer_path and Path(tokenizer_path).exists():
        import sentencepiece as spm
        sp = spm.SentencePieceProcessor(model_file=tokenizer_path)

    stats = dataset_stats(splits, gen.pools, lexicon, sp)
    stats["seed"] = seed
    stats["language"] = language
    stats["entity_pool_members"] = gen.pools

    out_dir.mkdir(parents=True, exist_ok=True)
    for s, exs in splits.items():
        write_jsonl(out_dir / f"{s}.jsonl", exs)
    (out_dir / "stats.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    write_samples_md(out_dir / "samples.md", splits, language)

    if report_dir is not None:
        report_dir.mkdir(parents=True, exist_ok=True)
        (report_dir / f"{language}_reasoning_data.json").write_text(
            json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
        write_samples_md(report_dir / f"{language}_reasoning_samples.md",
                         splits, language)
    for s, n in stats["sizes"].items():
        logger.info("  %-10s %6d examples -> %s", s, n, out_dir / f"{s}.jsonl")
    return stats
