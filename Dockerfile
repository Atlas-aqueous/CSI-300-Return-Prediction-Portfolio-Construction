FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV OMP_NUM_THREADS=4
ENV MKL_NUM_THREADS=4
ENV OPENBLAS_NUM_THREADS=4
ENV PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple \
    numpy \
    pandas \
    scikit-learn \
    joblib \
    xgboost

COPY . /app

RUN chmod +x /app/init.sh /app/train.sh /app/test.sh

CMD ["sh", "/app/test.sh"]
