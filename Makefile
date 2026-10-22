.PHONY: report notebook test

report:            ## build all charts into figures/ from the offline sample
	python -m energy_eda report

notebook:          ## re-execute the notebook in place, keeping outputs
	python -m nbconvert --to notebook --execute --inplace notebooks/energy_eda.ipynb

test:
	python -m pytest -q
