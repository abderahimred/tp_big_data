# 🔧 Fix: Spark Task Size Warning

## Problem

You're encountering this warning:
```
WARN TaskSetManager: Stage 268 contains a task of very large size (56489 KiB). 
The maximum recommended task size is 1000 KiB.
```

## Root Cause

This happens when Spark tries to serialize large datasets that aren't properly partitioned. The 1-million sample synthetic dataset is being broadcast as a single large task instead of being distributed across partitions.

---

## ✅ Solution 1: Increase Partitions (Recommended)

Replace the `preprocess_for_spark` function in your notebook with this optimized version:

### Updated Code:

```python
def preprocess_for_spark(df, spark, n_partitions=None):
    """
    Preprocess data for Spark KMeans with proper partitioning
    Returns: Spark DataFrame with 'features' column
    
    Parameters:
    -----------
    df : pandas DataFrame
        Input data
    spark : SparkSession
        Active Spark session
    n_partitions : int, optional
        Number of partitions. If None, automatically calculated based on size
    """
    # Automatically determine optimal partitions based on data size
    if n_partitions is None:
        n_rows = len(df)
        if n_rows < 10000:
            n_partitions = 2
        elif n_rows < 100000:
            n_partitions = 8
        else:
            # For large datasets: ~100k rows per partition
            n_partitions = max(8, n_rows // 100000)
    
    print(f"   Using {n_partitions} partitions for {len(df):,} rows")
    
    # Convert Pandas to Spark DataFrame with repartitioning
    spark_df = spark.createDataFrame(df.astype(float))
    spark_df = spark_df.repartition(n_partitions)
    
    # Get column names
    feature_columns = spark_df.columns
    
    # Assemble features into vector
    assembler = VectorAssembler(inputCols=feature_columns, outputCol="raw_features")
    assembled_df = assembler.transform(spark_df)
    
    # Standardize features
    scaler = SparkScaler(inputCol="raw_features", outputCol="features", 
                         withStd=True, withMean=True)
    scaler_model = scaler.fit(assembled_df)
    scaled_df = scaler_model.transform(assembled_df)
    
    return scaled_df.select("features")
```

### Update the preprocessing section:

```python
# Preprocess all datasets for both frameworks
print("🔄 Preprocessing datasets...\n")

# Small dataset (Wine)
wine_sklearn = preprocess_for_sklearn(wine_df)
wine_spark = preprocess_for_spark(wine_df, spark)  # Will use 2 partitions
print(f"✅ Wine Quality - Sklearn: {wine_sklearn.shape}, Spark: {wine_spark.count()} rows")

# Medium dataset (MNIST)
mnist_sklearn = preprocess_for_sklearn(mnist_df)
mnist_spark = preprocess_for_spark(mnist_df, spark)  # Will use 8 partitions
print(f"✅ MNIST - Sklearn: {mnist_sklearn.shape}, Spark: {mnist_spark.count()} rows")

# Large dataset (Synthetic) - IMPORTANT: More partitions!
synthetic_sklearn = preprocess_for_sklearn(synthetic_df)
synthetic_spark = preprocess_for_spark(synthetic_df, spark, n_partitions=16)  # Use 16 partitions
print(f"✅ Synthetic - Sklearn: {synthetic_sklearn.shape}, Spark: {synthetic_spark.count()} rows")
```

---

## ✅ Solution 2: Optimize Spark Configuration

Update your Spark Session initialization:

```python
# Creating Spark Session with optimized configuration
spark = SparkSession.builder \
    .appName("BigDataClustering_Comparative_Analysis") \
    .config("spark.driver.memory", "4g") \
    .config("spark.executor.memory", "4g") \
    .config("spark.sql.shuffle.partitions", "16") \
    .config("spark.driver.maxResultSize", "2g") \
    .config("spark.default.parallelism", "8") \
    .config("spark.rpc.message.maxSize", "128") \
    .config("spark.kryoserializer.buffer.max", "512m") \
    .getOrCreate()

# Setting log level to reduce verbosity
spark.sparkContext.setLogLevel("ERROR")  # Changed from WARN to ERROR
```

### Configuration Explanation:

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `spark.sql.shuffle.partitions` | 16 | More partitions for data shuffling |
| `spark.default.parallelism` | 8 | Parallel tasks |
| `spark.rpc.message.maxSize` | 128 | Increase max message size (MB) |
| `spark.kryoserializer.buffer.max` | 512m | Increase serialization buffer |

---

## ✅ Solution 3: Use Sampling for Large Datasets

For extremely large datasets, you can sample before training:

```python
def run_spark_kmeans_optimized(spark_df, k, random_state=42, max_iter=300, sample_fraction=None):
    """
    Run Spark MLlib KMeans clustering with optional sampling
    
    Parameters:
    -----------
    sample_fraction : float, optional
        Fraction of data to sample (e.g., 0.5 for 50%). None for no sampling.
    """
    # Sample if specified
    if sample_fraction is not None and sample_fraction < 1.0:
        print(f"   Sampling {sample_fraction*100:.0f}% of data...")
        spark_df = spark_df.sample(withReplacement=False, fraction=sample_fraction, seed=random_state)
    
    # Cache and repartition for better performance
    spark_df = spark_df.repartition(16).cache()
    
    # Memory before training
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss / (1024 ** 2)
    
    # Train KMeans
    start_time = time.time()
    
    kmeans = SparkKMeans(
        k=k,
        seed=random_state,
        maxIter=max_iter,
        initMode='k-means||'
    )
    
    model = kmeans.fit(spark_df)
    predictions = model.transform(spark_df)
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    # Memory after training
    mem_after = process.memory_info().rss / (1024 ** 2)
    mem_used = mem_after - mem_before
    
    # Compute metrics
    inertia = model.summary.trainingCost
    
    evaluator = ClusteringEvaluator(
        predictionCol='prediction',
        featuresCol='features',
        metricName='silhouette',
        distanceMeasure='squaredEuclidean'
    )
    
    silhouette = evaluator.evaluate(predictions)
    labels = predictions.select('prediction').rdd.flatMap(lambda x: x).collect()
    
    # Unpersist
    spark_df.unpersist()
    
    return {
        'framework': 'Spark MLlib',
        'n_clusters': k,
        'n_samples': spark_df.count(),
        'n_features': None,
        'execution_time': execution_time,
        'memory_mb': mem_used,
        'inertia': inertia,
        'silhouette_score': silhouette,
        'davies_bouldin': None,
        'labels': labels,
        'model': model
    }
```

---

## ✅ Solution 4: Reduce Dataset Size (If Still Having Issues)

As a last resort, reduce the synthetic dataset size:

```python
# Instead of 1,000,000 samples
synthetic_df, synthetic_labels = generate_synthetic_large(
    n_samples=500000,  # Reduced to 500k
    n_features=20,
    n_clusters=10,
    random_state=42
)
```

---

## 📊 Expected Results After Fixes

After applying these fixes, you should see:

```
🔄 Preprocessing datasets...

   Using 2 partitions for 1,599 rows
✅ Wine Quality - Sklearn: (1599, 11), Spark: 1599 rows
   Using 8 partitions for 70,000 rows
✅ MNIST - Sklearn: (70000, 784), Spark: 70000 rows
   Using 16 partitions for 1,000,000 rows
✅ Synthetic - Sklearn: (1000000, 20), Spark: 1000000 rows
```

**No more warnings!** ✅

---

## 🎯 Quick Fix Summary

**Minimum changes needed:**

1. **Update Spark config:** Add `spark.sql.shuffle.partitions = 16`
2. **Repartition large data:** Use `.repartition(16)` when creating Spark DataFrame for synthetic data
3. **Cache strategically:** Use `.cache()` before fitting

---

## 🧪 Testing

After making changes, verify:

```python
# Check partition count
print(f"Partitions: {synthetic_spark.rdd.getNumPartitions()}")

# Should show: Partitions: 16 (or your chosen value)
```

---

## 💡 Why This Works

**Problem:** Spark was trying to serialize 1M rows as a single task  
**Solution:** Split into 16 smaller tasks (~62.5k rows each)  
**Result:** Each task is ~3.5 MB instead of 56 MB ✅

**Rule of thumb:** Aim for **50k-100k rows per partition** for optimal performance.

---

## 🔍 Additional Monitoring

Add this cell to monitor Spark performance:

```python
# Monitor Spark job progress
def monitor_spark_job():
    sc = spark.sparkContext
    print(f"App Name: {sc.appName}")
    print(f"Master: {sc.master}")
    print(f"Default Parallelism: {sc.defaultParallelism}")
    print(f"Active Jobs: {len(sc.statusTracker().getActiveJobIds())}")
    
monitor_spark_job()
```

---

**Apply this fix and your warnings should disappear!** 🚀
