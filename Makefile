.PHONY: build clean run

build:
	uv run python -m PyInstaller \
		--clean \
		--onedir \
		--console \
		--paths=src \
		--copy-metadata gdown \
		--collect-submodules klinikatlas_search \
		--collect-submodules transformers \
		--add-data "src/klinikatlas_search/data;klinikatlas_search/data" \
		prototype_exec.py

run:
	uv run python prototype_exec.py

clean:
	uv run python -c "import shutil; shutil.rmtree('build', ignore_errors=True); shutil.rmtree('dist', ignore_errors=True)"