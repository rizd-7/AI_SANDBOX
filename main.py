import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.datasets import mnist
import numpy as np
import random
import matplotlib.pyplot as plt

# 1. Chargement et prétraitement du dataset MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()
num_classes = 10

x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)

val_split = 0.1
num_val = int(val_split * x_train.shape[0])
x_val = x_train[:num_val]
y_val = y_train[:num_val]
x_train2 = x_train[num_val:]
y_train2 = y_train[num_val:]

# 2. Analyse du dataset
img_shape = x_train.shape[1:3]
dataset_size = x_train.shape[0]


# 3. Celle qui a été utilisée pour le modèle basé sur EMNIST 
param_instance = {
    'num_conv_layers': 2,                # Deux couches convolutionnelles
    'filters': [32, 64],                # Filtres pour chaque couche conv
    'kernel_size': [(3, 3), (3, 3)],    # Taille du noyau pour chaque couche conv
    'pooling_layers': 2,                # Deux couches de pooling
    'dense_units': [64, 32],            # Deux couches denses avant la sortie
    'dropout_rate': 0.0,                # Pas de dropout dans le modèle
    'activation': 'relu',              # Fonction d'activation utilisée partout sauf en sortie
    'learning_rate': None,             # Non spécifié explicitement, donc par défaut dans Adam
    'batch_size': None,                # Non précisé dans ce code
    'optimizer': 'adam',               # Optimiseur utilisé
    'l2_reg': 0.0,                      # Pas de régularisation L2 utilisée
    'batch_norm': False,               # Pas de batch normalization dans le modèle
    'pooling_type': 'max_pooling',     # Type de pooling utilisé
}



def expand_param_instance(param_instance):
    def around(value, scale=[0.5, 1, 1.5]):
        """Génère des variations autour d'une valeur numérique."""
        return sorted(set(int(v) for v in [value * s for s in scale] if v > 0))

    def perturb_list(values):
        """Ajoute des variations autour d'une liste de valeurs (e.g., filtres, dense_units)."""
        new_values = set()
        for v in values:
            new_values.update(around(v))
        return sorted(new_values)

    expanded_space = {
        'num_conv_layers': list(range(max(1, param_instance['num_conv_layers'] - 1),
                                      param_instance['num_conv_layers'] + 2)),
        'filters': perturb_list(param_instance['filters']),
        'kernel_size': list(set([
            param_instance['kernel_size'][0],  # assume same for all
            (param_instance['kernel_size'][0][0] + 1, param_instance['kernel_size'][0][1] + 1),
            (max(1, param_instance['kernel_size'][0][0] - 1), max(1, param_instance['kernel_size'][0][1] - 1))
        ])),
        'pooling_layers': list(range(1, param_instance['pooling_layers'] + 2)),
        'dense_units': perturb_list(param_instance['dense_units']),
        'dropout_rate': sorted(set([param_instance['dropout_rate'], 0.1] if param_instance['dropout_rate'] == 0.0 else [param_instance['dropout_rate'], 0.0])),
        'activation': ['relu', 'tanh', 'elu', 'selu'],
        'learning_rate': [0.001, 0.0001] if not param_instance['learning_rate'] else [param_instance['learning_rate']],
        'batch_size': [32, 64, 128],
        'optimizer': ['adam', 'sgd', 'rmsprop', 'nadam'],
        'l2_reg': [0.0, 0.001, 0.01],
        'batch_norm': [True, False],
        'pooling_type': ['max_pooling', 'avg_pooling', 'global_avg_pooling'],
    }

    return expanded_space



def generate_param_space(img_shape, dataset_size):
    height, width = img_shape
    total_pixels = height * width

    # Règles basées sur la taille de l'image
    if total_pixels <= 32 * 32:
        filters = [16, 32, 64]
        kernel_sizes = [3]
        pooling_layers = [1, 2]
        dense_units = [64, 128]
    elif total_pixels <= 64 * 64:
        filters = [32, 64, 128]
        kernel_sizes = [3, 5]
        pooling_layers = [1, 2, 3]
        dense_units = [128, 256]
    else:
        filters = [64, 128, 256, 512]
        kernel_sizes = [3, 5, 7]
        pooling_layers = [2, 3]
        dense_units = [256, 512]

    # Règles basées sur la taille du dataset
    if dataset_size <= 10000:
        batch_sizes = [16, 32]
        learning_rates = [1e-3, 1e-4]
        dropout_rates = [0.1, 0.2, 0.3]
    elif dataset_size <= 60000:
        batch_sizes = [32, 64, 128]
        learning_rates = [1e-2, 1e-3, 1e-4]
        dropout_rates = [0.2, 0.3, 0.5]
    else:
        batch_sizes = [64, 128, 256]
        learning_rates = [1e-2, 1e-3]
        dropout_rates = [0.3, 0.5, 0.6]

    # Règles croisées pour la régularisation L2
    if dataset_size < 20000 and total_pixels <= 32 * 32:
        l2_reg = [0.0, 0.001]
    else:
        l2_reg = [0.0, 0.001, 0.01]

    # Base de connaissance complète
    param_space = {
        'num_conv_layers': [1, 2, 3, 4],
        'filters': filters,
        'kernel_size': kernel_sizes,
        'pooling_layers': pooling_layers,
        'dense_units': dense_units,
        'dropout_rate': dropout_rates,
        'activation': ['relu', 'tanh', 'elu', 'selu'],
        'learning_rate': learning_rates,
        'batch_size': batch_sizes,
        'optimizer': ['adam', 'sgd', 'rmsprop', 'nadam'],
        'l2_reg': l2_reg,
        'batch_norm': [True, False],
        'pooling_type': ['max_pooling', 'avg_pooling', 'global_avg_pooling'],
    }

    return param_space


# 3. Génération de l'espace de recherche des hyperparamètres
expanded_space = expand_param_instance(param_instance)
rule_based_space = generate_param_space(img_shape, dataset_size)

# Fusion intelligente : on garde uniquement les valeurs compatibles avec les règles
param_space = {}
for key in expanded_space:
    if key in rule_based_space:
        # Intersection des valeurs compatibles
        param_space[key] = list(set(expanded_space[key]) & set(rule_based_space[key]))
        if not param_space[key]:  # Si l'intersection est vide, on garde l'espace des règles
            param_space[key] = rule_based_space[key]
    else:
        # Si la clé n'a pas de contrainte dans les règles, on garde tout l'espace étendu
        param_space[key] = expanded_space[key]

print("Espace de recherche des hyperparamètres filtré automatiquement :")
for key, values in param_space.items():
    print(f"- {key}: {values}")





# 4. Construction du modèle
def build_model(params):
    model = models.Sequential([layers.Input(shape=(28, 28, 1))])

    for i in range(params['num_conv_layers']):
        model.add(layers.Conv2D(
            filters=params['filters'] * (2**i),
            kernel_size=params['kernel_size'],
            activation=params['activation'],
            padding='same',
            kernel_regularizer=tf.keras.regularizers.l2(params['l2_reg'])
        ))
        if params['batch_norm']:
            model.add(layers.BatchNormalization())
        if i < params['pooling_layers']:
            if params['pooling_type'] == 'max_pooling':
                model.add(layers.MaxPooling2D((2, 2)))
            else:
                model.add(layers.AveragePooling2D((2, 2)))

    model.add(layers.Flatten())
    model.add(layers.Dense(
        params['dense_units'],
        activation=params['activation'],
        kernel_regularizer=tf.keras.regularizers.l2(params['l2_reg'])
    ))
    if params['batch_norm']:
        model.add(layers.BatchNormalization())
    model.add(layers.Dropout(params['dropout_rate']))
    model.add(layers.Dense(num_classes, activation='softmax'))

    if params['optimizer'] == 'adam':
        opt = optimizers.Adam(learning_rate=params['learning_rate'])
    elif params['optimizer'] == 'sgd':
        opt = optimizers.SGD(learning_rate=params['learning_rate'], momentum=0.9)
    else:
        opt = optimizers.RMSprop(learning_rate=params['learning_rate'])

    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# 5. Recherche aléatoire
n_iterations = 5
results = []

for i in range(n_iterations):
    params = {k: random.choice(v) for k, v in param_space.items()}
    print(f"\n--- Iteration {i+1}/{n_iterations} with params: {params}")

    model = build_model(params)
    history = model.fit(
        x_train2, y_train2,
        epochs=10,
        batch_size=params['batch_size'],
        validation_data=(x_val, y_val),
        verbose=2
    )

    best_val_acc = max(history.history['val_accuracy'])
    results.append({'params': params, 'history': history.history, 'best_val_acc': best_val_acc})

# 6. Affichage des performances
for idx, res in enumerate(results):
    plt.figure()
    plt.plot(res['history']['accuracy'], label='Train Accuracy')
    plt.plot(res['history']['val_accuracy'], label='Val Accuracy')
    plt.title(f"Run {idx+1} - best val acc: {res['best_val_acc']:.4f}")
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

# 7. Évaluation finale
best_run = max(results, key=lambda x: x['best_val_acc'])
print("\nBest hyperparameters found:", best_run['params'])

best_model = build_model(best_run['params'])
best_model.fit(x_train, y_train,
               epochs=10,
               batch_size=best_run['params']['batch_size'],
               verbose=2)
test_loss, test_acc = best_model.evaluate(x_test, y_test, verbose=0)
print(f"Test accuracy: {test_acc:.4f}")
