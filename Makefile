.PHONY: setup dataset index-benchmark index search train-classifiers benchmark-check fit experiments report api demo-mock demo test

setup:
	python -m pip install -r requirements.txt

dataset:
	PYTHONPATH=. python scripts/build_dataset.py

index-benchmark:
	PYTHONPATH=. python -m trustrag.cli index --corpus data/benchmark_corpus --config trustrag/config.benchmark.yaml

index:
	PYTHONPATH=. python -m trustrag.cli index --corpus data/corpus --config trustrag/config.yaml

search:
	@test -n "$(Q)" || (echo 'Usage: make search Q="Vietnam visa duration"' && exit 2)
	PYTHONPATH=. python -m trustrag.cli search "$(Q)" --top-k $(or $(K),5) --config trustrag/config.yaml

train-classifiers:
	PYTHONPATH=. python -m trustrag.online.query.train --csv data/train/queries_labeled.csv --config trustrag/config.yaml

benchmark-check:
	PYTHONPATH=. python -m experiments.check_benchmark --benchmark data/benchmark/benchmark.csv --classifier-data data/train/queries_labeled.csv

fit:
	PYTHONPATH=. python -m experiments.fit --benchmark data/benchmark/benchmark.csv --config trustrag/config.benchmark.yaml

experiments:
	PYTHONPATH=. python -m experiments.run_all --smoke --config trustrag/config.yaml

benchmark-experiments:
	PYTHONPATH=. python -m experiments.run_all --benchmark data/benchmark/benchmark.csv --config trustrag/config.benchmark.yaml

report:
	PYTHONPATH=. python -m experiments.report

api:
	PYTHONPATH=. uvicorn trustrag.api:app --host 127.0.0.1 --port 8000

demo-mock:
	PYTHONPATH=. streamlit run ui/app.py

demo:
	PYTHONPATH=. streamlit run ui/app.py

test:
	PYTHONPATH=. pytest -q
