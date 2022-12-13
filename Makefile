render:
	jupyter nbconvert --to notebook --inplace --execute --output-dir=./rendered_notebooks/ --ExecutePreprocessor.timeout=None --ExecutePreprocessor.allow_errors=True book/notebooks/*.ipynb
	jupyter nbconvert --to html --execute --output-dir=./rendered_notebooks/ --ExecutePreprocessor.timeout=None --ExecutePreprocessor.allow_errors=True book/notebooks/*.ipynb

format:
	jupytext --set-formats book/notebooks//ipynb,python_scripts//py:percent book/notebooks/*.ipynb

sync:
	jupytext --sync book/notebooks/*.ipynb

book:
	jupyter-book build book

book-clean:
	jupyter-book clean book