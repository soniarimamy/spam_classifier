"""
Preprocessing avec PySpark :
  - Tokenisation du texte
  - Suppression des stop-words
  - TF-IDF (HashingTF + IDF)
  - Retourne X (numpy) et y (numpy) pour auto-sklearn
"""
import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF
from pyspark.ml.linalg import SparseVector, DenseVector


NUM_FEATURES = 500  # dimension du vecteur TF-IDF


def _vector_to_array(v) -> np.ndarray:
    if isinstance(v, SparseVector):
        return v.toArray()
    if isinstance(v, DenseVector):
        return np.array(v)
    return np.array(v)


def build_spark_session() -> SparkSession:
    return (
        SparkSession.builder
        .appName("SpamClassifier")
        .master("local[*]")
        .config("spark.driver.memory", "2g")
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )


def preprocess(csv_path: str = "data/messages.csv"):
    """
    Charge le CSV, applique le pipeline PySpark TF-IDF,
    renvoie (X_train, X_test, y_train, y_test, X_all, y_all, df_raw).
    """
    spark = build_spark_session()
    spark.sparkContext.setLogLevel("ERROR")

    print("[preprocess] Chargement des données avec PySpark …")
    df = spark.read.csv(csv_path, header=True, inferSchema=True)
    df = df.dropna(subset=["text", "label"])
    df = df.withColumn("label", df["label"].cast("integer"))
    print(f"  {df.count()} lignes chargées")

    # ── Pipeline NLP PySpark ──────────────────────────────────────────────────
    tokenizer   = Tokenizer(inputCol="text",           outputCol="words")
    remover     = StopWordsRemover(inputCol="words",   outputCol="filtered")
    hashing_tf  = HashingTF(inputCol="filtered",       outputCol="raw_tf",
                            numFeatures=NUM_FEATURES)
    idf         = IDF(inputCol="raw_tf",               outputCol="features")

    pipeline = Pipeline(stages=[tokenizer, remover, hashing_tf, idf])
    print("[preprocess] Entraînement du pipeline TF-IDF PySpark …")
    pipeline_model = pipeline.fit(df)
    processed      = pipeline_model.transform(df)

    # ── Conversion Spark → NumPy ──────────────────────────────────────────────
    print("[preprocess] Conversion Spark → NumPy …")
    pdf = processed.select("text", "features", "label").toPandas()

    X = np.vstack(pdf["features"].apply(_vector_to_array).values)
    y = pdf["label"].values.astype(int)

    # Sauvegarde texte brut pour EvidentlyAI
    df_raw = pdf[["text", "label"]].copy()

    spark.stop()
    print(f"[preprocess] Matrice features : {X.shape}  labels : {y.shape}")
    return X, y, df_raw
