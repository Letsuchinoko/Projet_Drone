try:
    import tensorflow as tf
    from tensorflow.keras import layers
    print(f"✅ TensorFlow {tf.__version__} installé correctement!")
except ImportError:
    print("❌ TensorFlow non installé - Lancez: pip install tensorflow")    