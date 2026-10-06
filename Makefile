PYTHON ?= python3
PERIODS ?= data/raw/Periods

.PHONY: reproduce data results test clean

reproduce: data results

data:
	$(PYTHON) scripts/build_data.py --periods $(PERIODS)

results:
	$(PYTHON) scripts/run_audits.py

test:
	$(PYTHON) -m pytest -q

clean:
	rm -rf data/built
