FROM python:3.9-slim-bullseye

# Java 11 (PySpark) + outils de build (swig)
# JAVA_HOME doit être défini AVANT pip install (pyspark en a besoin)
RUN apt-get update && apt-get install -y \
    openjdk-11-jdk-headless \
    build-essential \
    swig \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/lib/jvm/java-11-openjdk-$(dpkg --print-architecture) /usr/lib/jvm/java-11

ENV JAVA_HOME=/usr/lib/jvm/java-11
ENV PATH=$JAVA_HOME/bin:$PATH

WORKDIR /app

COPY requirements.txt .
# Upgrader pip (meilleur resolver, moins de backtracking)
RUN pip install --upgrade pip && \
    pip install --no-cache-dir --timeout 300 -r requirements.txt

COPY . .

# Variables runtime placées APRÈS pip install pour ne pas invalider son cache
ENV PYTHONUNBUFFERED=1
ENV DOCKER=1
ENV GIT_PYTHON_REFRESH=quiet

RUN mkdir -p data reports mlruns

EXPOSE 5000 8080

RUN chmod +x entrypoint.sh

CMD ["./entrypoint.sh"]
