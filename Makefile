.PHONY:

#################################################################################
# GLOBALS                                                                       #
#################################################################################

ROOT_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST)))/)
PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
BUCKET = /
BUCKET_ENV = /
PROFILE = default
PROJECT_NAME = ArXiv Recommendation System
PYTHON_INTERPRETER = python3.8

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Export Path
path:
	@bash -c "export PATH=/$(ROOT_DIR):$(PATH)"
	@bash -c "source ~/.bashrc"
	@echo $(PATH)

## Activate environment
active:
	poetry shell

## Install dependecies
install: active
	$(PYTHON_INTERPRETER) poetry install

## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

#################################################################################
# PROJECT RULES                                                                 #
#################################################################################

# MAIN

## Start main app
app: active
	$(PYTHON_INTERPRETER) src/recommend.py

## Make data
data: active
	$(PYTHON_INTERPRETER) src/data.py

## Make figures
visual: active
	$(PYTHON_INTERPRETER) src/visual.py

# Test model
test_ident: active
	$(PYTHON_INTERPRETER) src/recommend.py test

# Train model
train_ident: active
	$(PYTHON_INTERPRETER) src/recommend.py train

#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

# Inspired by <http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html>
# sed script explained:
# /^##/:
# 	* save line in hold space
# 	* purge line
# 	* Loop:
# 		* append newline + line to hold space
# 		* go to next line
# 		* if line starts with doc comment, strip comment character off and loop
# 	* remove target prerequisites
# 	* append hold space (+ newline) to line
# 	* replace newline plus comments by `---`
# 	* print line
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
