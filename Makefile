.PHONY: install run test lint clean

install:
	pip install -r requirements.txt

run:
	streamlit run app.py

test:
	pytest tests/ -v

lint:
	python -m py_compile app.py utils/models.py utils/charts.py utils/report.py
	echo "No syntax errors."

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete
	find . -name ".DS_Store" -delete

setup: install
	@echo "VIASTRA is ready. Run: make run"
