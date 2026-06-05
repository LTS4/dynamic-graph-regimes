.PHONY: clean lint test_environment requirements format env

#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))

ENV_NAME = dynamic-graph-regimes

REQ_FILE = requirements.yaml

ifeq (, $(shell which conda))
	HAS_CONDA = False
else
	HAS_CONDA = True
endif

BASE = py312_base

#################################################################################
# PROJECT RULES                                                                 #
#################################################################################

# Put here repeated commands
# E.g:
# ## Make Dataset
# data: src/data/main
# 	python -m src.data.main && touch data


#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Lint using pylint
lint: test_environment
	pylint src

## Format with black and sort imports with isort
format:
	black src
	isort src

## Set up python interpreter environment
env: $(REQ_FILE)
ifeq (True, $(HAS_CONDA))
ifeq ("$(shell conda env list | grep $(ENV_NAME))", "")
ifeq ("$(shell conda env list | grep $(BASE))", "")
	conda env create --file $(REQ_FILE) --name $(ENV_NAME)
else
	conda create --name $(ENV_NAME) --clone $(BASE)
	conda env update --name $(ENV_NAME) --file $(REQ_FILE) --prune
endif
else
	conda env update --name $(ENV_NAME) --file $(REQ_FILE) --prune
endif
else
	@echo "Conda not installed, manually create environment"
	exit 1
endif

## Test python environment is setup correctly
test_environment:
ifneq ($(ENV_NAME), $(CONDA_PREFIX))
	@echo "Environment not activated"
	exit 1
endif


#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

# Inspired by <http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html>
# sed script explained:
# /^##/:
#	 * save line in hold space
#	 * purge line
#	 * Loop:
#		 * append newline + line to hold space
#		 * go to next line
#		 * if line starts with doc comment, strip comment character off and loop
#	 * remove target prerequisites
#	 * append hold space (+ newline) to line
#	 * replace newline plus comments by `---`
#	 * print line
# Separate expressions are necessary because labels cannot be delimited by
# semicolon; see <http://stackoverflow.com/a/11799865/1968>
.PHONY: help
help:
	@echo "$$(tput bold)Available rules:$$(tput sgr0)"
	@echo
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| LC_ALL='C' sort --ignore-case \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| more $(shell test $(shell uname) = Darwin && echo '--no-init --raw-control-chars')
