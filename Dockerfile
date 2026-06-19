FROM python:3.9-slim-bullseye

# Java 11 (PySpark) + outils de build (auto-sklearn / swig)
RUN apt-get update && apt-get install -y \
    openjdk-11-jdk-headless \
    build-essential \
    swig \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/lib/jvm/java-11-openjdk-$(dpkg --print-architecture) /usr/lib/jvm/java-11

ENV JAVA_HOME=/usr/lib/jvm/java-11
ENV PATH=$JAVA_HOME/bin:$PATH
ENV PYTHONUNBUFFERED=1
ENV DOCKER=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p data reports mlruns

EXPOSE 5000 8080

RUN chmod +x entrypoint.sh

CMD ["./entrypoint.sh"]
