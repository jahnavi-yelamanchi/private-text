FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

WORKDIR /service
COPY requirements.runtime.txt .
RUN pip install --no-cache-dir --extra-index-url https://pypi.nvidia.com -r requirements.runtime.txt

COPY app ./app
ENV PRIVATE_TEXT_ARTIFACTS_PATH=/models
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
