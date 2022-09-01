render:
	jupyter nbconvert --to notebook --inplace --execute --output-dir=./rendered_notebooks/ --ExecutePreprocessor.timeout=None --ExecutePreprocessor.allow_errors=True notebooks/*.ipynb
	jupyter nbconvert --to html --execute --output-dir=./rendered_notebooks/ --ExecutePreprocessor.timeout=None --ExecutePreprocessor.allow_errors=True notebooks/*.ipynb

format:
	jupytext --set-formats notebooks//ipynb,python_scripts//py:percent notebooks/*.ipynb

sync:
	jupytext --sync notebooks/*.ipynb