FROM python:3.11-slim
WORKDIR /service
ARG REQUIREMENTS_FILE=requirements.txt
COPY requirements.txt requirements-public.txt ./
RUN pip install --no-cache-dir -r ${REQUIREMENTS_FILE}
COPY app ./app
COPY data ./data
EXPOSE 8000
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
