# Writing blind benchmark cases

This guide is for whoever writes the next batch of benchmark cases (`blind_v1`). The batch
exists to test one claim fairly: **weighting trust signals differently per topic (S2) gives
better answers than one fixed weighting (S1).** The cases must be written *without* seeing how
any system behaves. Otherwise the result can't be trusted, whichever way it goes.

## The rules

1. **Do not run the systems.** Don't run `make benchmark-experiments`, `make fit`,
   `make ablation-freshness`, `experiments/sensitivity.py`, or the API/demo, and don't open
   anything under `results/`. The only command you need is the validator (below). It checks
   your writing, not the systems' answers.
2. **Write cases that reflect how conflicts really happen**, not cases meant to make any
   system win or lose. For every question, fill in a `pattern:` note saying which real-world
   situation it models (a real example is best, e.g. "UK raised the allowance from £145 to £390
   in 2009 and old blogs kept the old figure").
3. **Use new destinations.** A new document added to an existing destination (Vietnam,
   Thailand, …) changes the labels of the existing questions there. Use countries not yet in
   `facts.yaml`, e.g. Germany, UAE, South Korea, Turkey, Egypt, Maldives, Qatar, Germany, France, Brazil.
4. **Put every fact in the blind batch and the test split**: add `split: test` and
   `batch: blind_v1` to each fact. Nothing in this batch is ever used for development or fitting.
5. **Commit the batch before anyone evaluates it.** Once it's evaluated, it can't be edited.
   Errors found afterwards get fixed in a new batch (`blind_v2`).

## What to write

Aim for **at least 10 questions per topic** (visa, vaccination, customs), each with 2–3
paraphrased queries. Use whatever mix of the situations below *you* think is realistic for that
topic. Write down your reasoning in the `pattern:` notes; don't adjust the mix later.

| Situation | What the documents look like | Correct (gold) answer |
|---|---|---|
| **A. Rule changed; old official page never updated** | Newer official notice (embassy or ministry) with the new rule; older official page, recently crawled, still shows the old rule | New rule |
| **B. Change announced but not yet in force** | Recent official proposal, draft or "from next year" page with a new value (`in_force: false`); older official page with the rule in force today | Rule in force today |
| **C. Recent but unreliable source** | Recent blog, forum or news post with a wrong value; older official page with the right value | Official value |
| **D. Old rule repeated by many** | Several old secondary sources repeating a superseded rule; one current official source | Current official value |
| **E. No conflict** | One or more sources that all agree | That value |

Notes:
- Customs questions retrieve **official sources only**, so customs cases need their conflict
  between official pages (types A and B). Blogs and forums in customs only affect the vanilla
  baseline.
- The gold answer must be stated by an in-force official source (government, embassy or
  international body) that is in scope for the traveller's nationality. The validator enforces this.

## How to write documents the validator accepts

The validator runs the project's extractor on each document and checks that it finds exactly
the values you declared in `claims`. Short, plain sentences work best:

- **One topic per document, one customs item per document.**
- Yes/no: use "is required", "is not required", "must obtain", "visa-free", "exempt".
  Avoid "do not need" (the extractor doesn't read it as "no").
- Stay length: "up to 30 days" or "up to 6 months". Don't put another "N days" or "N months"
  in the same document.
- Money: currency code **before** the amount, e.g. `AED 3,000`, `INR 50,000`. Supported codes:
  USD EUR GBP INR SGD THB AUD CAD JPY VND. Write other amounts in words ("ten thousand US dollars").
- Customs items: name the item literally (`drone`, `e-cigarette`, `medicine`, `food`, `alcohol`,
  `tobacco`) with exactly one of "prohibited", "restricted"/"requires approval", or "allowed".
- Vaccines: `yellow fever`, `polio` or `covid-19`.
- Avoid hedging words in official documents unless the real page uses them ("may", "usually",
  "generally", "typically"); they lower a source's quality score.
- Dates: `content_date` is when the content was written or last changed. Leave out
  `snapshot_date` for pages crawled now. For an archived page, set `snapshot_date` to when it
  was last seen.

Queries:
- Name the destination country, and the traveller's nationality as an adjective ("Indian",
  "British", "American"). Leave it out to mean Indian travellers.
- Don't put a nationality adjective in front of "customs" ("Australian customs" is read as an
  Australian traveller); write "customs in Australia".
- Include the question type: "how long / stay / days" (stay length), "e-visa / visa type"
  (visa type), "visa" (visa needed?), "duty-free" (allowance), "declare" (declaration),
  "bring / carry / allowed / prohibited" (item), "certificate / proof", "which vaccine",
  "vaccine". Customs questions also need "customs", "duty", "declare" or "import".
- A new destination must be added to `trustrag/policies/query.yaml` under
  `entities: destinations:`.

## Template

```yaml
  Germany:
    docs:
      - id: de_customs_allowance_gov_2025
        category: customs_regulation
        source_type: government
        domain: zoll.de.example
        scope: ALL
        content_date: "2025-05-01"
        title: Germany duty-free allowance for air travellers
        text: >-
          Travellers arriving in Germany by air from outside the EU can bring goods worth up to
          EUR 430 duty-free.
        claims: {duty_free_limit: 430 EUR}
      - id: de_customs_proposal_2026
        category: customs_regulation
        source_type: government
        domain: bundesfinanzministerium.de.example
        scope: ALL
        content_date: "2026-09-15"
        in_force: false
        title: Draft revision of the traveller allowance
        text: >-
          Draft regulation: the duty-free allowance for air travellers arriving in Germany is
          proposed to rise to EUR 700.
        claims: {duty_free_limit: 700 EUR}
    facts:
      - id: de_customs
        nationality: India
        category: customs_regulation
        split: test
        batch: blind_v1
        pattern: "B: announced allowance increase not yet in force"
        truth: {duty_free_limit: 430 EUR}
        questions:
          - slot: duty_free_limit
            queries:
              - What is the duty-free allowance when arriving in Germany?
              - How much can I bring into Germany duty free?
```

(The values above are illustrative: replace them with your own case.)

## Checking and submitting

```bash
PYTHONPATH=. python scripts/build_dataset.py --check
```

Fix every error it reports. It only checks documents and query wording; it never runs the
systems. When it prints `OK`, commit `facts.yaml` (and `query.yaml`, if you added destinations)
on its own branch and hand it over. The evaluation is then run once:

```bash
make dataset index-benchmark
make ablation-freshness && make fit benchmark-experiments report
```

The blind batch is reported separately from `core_v1` (the `batch` column in
`data/benchmark/benchmark.csv`).
