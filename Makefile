# Stages are added as they are implemented. Each target = one command.
.PHONY: test data train quant eval report all
test:
	python -m pytest -q
data train quant eval report all:
	@echo "not implemented yet (see CLAUDE.md section 12)"; exit 1
