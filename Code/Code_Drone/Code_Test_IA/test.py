print("=== TEST TENSORFLOW ===")

try:
    import tensorflow as tf
    print(f"✅ TensorFlow importé: version {tf.__version__}")
    
    # Test des sous-modules
    keras = tf.keras
    print("✅ Keras accessible")
    
    layers = tf.keras.layers
    print("✅ Layers accessible")
    
    print("🎉 TensorFlow complètement fonctionnel!")
    
except ImportError as e:
    print(f"❌ TensorFlow non installé: {e}")
except Exception as e:
    print(f"❌ Erreur TensorFlow: {e}")