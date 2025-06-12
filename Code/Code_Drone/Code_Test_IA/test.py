# Test complet
python -c "
import tensorflow as tf
print('TensorFlow version:', tf.__version__)
from tensorflow import keras
print('Keras version:', keras.__version__)
from tensorflow.keras import layers
print('✅ Tout fonctionne!')
"