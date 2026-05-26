PYTHON ?= python

.PHONY: help activate train test strain stest train-all htrain htest htrain-all

help:
	@printf '%s\n' \
	  'Available targets:' \
	  '  make activate     - Print the conda activation command' \
	  '  make train        - Run training (edit scripts/train.py to change config)' \
	  '  make test         - Run testing (edit scripts/test.py to change config)' \
	  '  make strain       - Run smoke training (edit scripts/strain.py to change config)' \
	  '  make stest        - Run smoke testing (edit scripts/stest.py to change config)' \
	  '  make train-all    - Run all models across datasets with skip for missing data' \
	  '  make htrain       - Show training script help' \
	  '  make htest        - Show testing script help' \
	  '  make htrain-all   - Show batch training script help' \
	  '  make help         - Show this help message'

activate:
	@printf '%s\n' 'conda activate HSI-X-Classify-Backbone'

train:
	$(PYTHON) scripts/train.py $(ARGS)

test:
	$(PYTHON) scripts/test.py $(ARGS)

strain:
	$(PYTHON) scripts/strain.py $(ARGS)

stest:
	$(PYTHON) scripts/stest.py $(ARGS)

train-all:
	$(PYTHON) scripts/train_all.py $(ARGS)

htrain:
	$(PYTHON) scripts/train.py -h

htest:
	$(PYTHON) scripts/test.py -h

htrain-all:
	$(PYTHON) scripts/train_all.py -h
