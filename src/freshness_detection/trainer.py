from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

def train_model(model, data, epochs=75, batch_size=32):
    # Checkpoint to save the best model based on validation accuracy
    best_model_checkpoint = ModelCheckpoint(
        filepath="best_model.h5", 
        monitor="val_accuracy", 
        verbose=1, 
        save_best_only=True, 
        mode="max"
    )

    # Checkpoint to save every model after each epoch
    individual_model_checkpoint = ModelCheckpoint(
        filepath="model_epoch_{epoch:02d}.h5",  
        verbose=1,
        save_weights_only=False  
    )

    # Early stopping to prevent overfitting
    early_stopping = EarlyStopping(
        monitor="val_accuracy", 
        patience=10, 
        restore_best_weights=True
    )

    # Train the model with checkpoints
    model.fit(
        data, 
        batch_size=batch_size, 
        epochs=epochs, 
        validation_data=data, 
        callbacks=[best_model_checkpoint, individual_model_checkpoint, early_stopping]
    )