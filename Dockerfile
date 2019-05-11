FROM python:3.6.8
MAINTAINER Matthew Emery <me@matthewemery.ca>
EXPOSE 5000

RUN pip install poetry

COPY pyproject.* .

RUN poetry install -n

COPY horsekickerpy/ horsekickerpy/
COPY app.py app.py

CMD ["poetry", "run", "python", "app.py"]
