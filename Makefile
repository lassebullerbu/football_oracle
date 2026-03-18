
default: pytest

# default: pylint pytest

# pylint:
# 	find . -iname "*.py" -not -path "./tests/test_*" | xargs -n1 -I {}  pylint --output-format=colorized {}; true

test:
	@pytest -v tests/test_pipeline.py
# ----------------------------------
#         LOCAL SET UP
# ----------------------------------

install:
	@pip install -r requirements.txt
	@pip install . -e
# ----------------------------------
#         HEROKU COMMANDS
# ----------------------------------

# ----------------------------------
#         PIPELINE & RUN
# ----------------------------------

# (mubiao main.py) proprecesing & training model
run_pipeline:
	@python main.py

streamlit:
	-@streamlit run app.py


# ----------------------------------
#    LOCAL INSTALL COMMANDS
# ----------------------------------
install_package:
	@pip install . -U
# ----------------------------------
#         DOCKER COMMANDS
# ----------------------------------
# if needed: build and run  local Docker
docker_build:
	docker build -t football_oracle .

docker_run:
	docker run -it -p 8501:8501 football_oracle

# ----------------------------------
#    CLEANING
# ----------------------------------
clean:
	@rm -fr */__pycache__
	@rm -fr __init__.py
	@rm -fr build
	@rm -fr dist
	@rm -fr *.dist-info
	@rm -fr *.egg-info
	-@rm model.joblib
