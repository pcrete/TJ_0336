import tensorflow as tf
from utils.metrics import loss

def to_tensor(X_train, y_train, X_valid, y_valid):
    X_train = tf.convert_to_tensor(X_train, dtype=tf.float32)
    y_train = tf.convert_to_tensor(y_train, dtype=tf.float32)

    X_valid = tf.convert_to_tensor(X_valid, dtype=tf.float32)
    y_valid = tf.convert_to_tensor(y_valid, dtype=tf.float32)
    
    return X_train, y_train, X_valid, y_valid

def train(model,
          X_train, y_train,
          X_valid, y_valid,
          training_losses=[], 
          testing_losses=[], 
          epoch_counts=0, 
          previous_testing_loss=100, 
          batch_size=10000, 
          n_steps=20, 
          lr=0.01, 
          n_epochs=3000):

    dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(batch_size)
    opt = tf.keras.optimizers.Adam(learning_rate=lr)
    
    try:
        for _ in range(n_epochs):
            for X, y in dataset:
                with tf.GradientTape() as tape:
                    y_ = model(X)
                    l = loss(y, y_)
                
                training_losses.append(l.numpy())
                testing_losses.append(loss(y_valid, model(X_valid)).numpy())

                weight = model.trainable_variables
                gradient = tape.gradient(l, weight)
                opt.apply_gradients(zip(gradient, weight))
            
            if epoch_counts % n_steps == 0:
                print(f"Epoch: {epoch_counts} \t training_losses: {training_losses[-1]} \t testing_losses: {testing_losses[-1]}")
            
                if testing_losses[-1] > previous_testing_loss: 
                    print('early stopping')
                    return model

                previous_testing_loss = testing_losses[-1]
                
            epoch_counts += 1
            
        return model
    except KeyboardInterrupt:
        print('stop training')
        return model
        pass