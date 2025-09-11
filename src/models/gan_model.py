import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd
from typing import Tuple, List, Optional
import logging
import matplotlib.pyplot as plt
from datetime import datetime
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StockGAN:
    """GAN model for stock price forecasting"""
    
    def __init__(self, 
                 sequence_length: int = 60,
                 n_features: int = 20,
                 latent_dim: int = 100,
                 generator_lr: float = 0.0002,
                 discriminator_lr: float = 0.0002):
        """
        Initialize Stock GAN
        
        Args:
            sequence_length: Length of input sequences
            n_features: Number of input features
            latent_dim: Dimension of latent space
            generator_lr: Generator learning rate
            discriminator_lr: Discriminator learning rate
        """
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.latent_dim = latent_dim
        self.generator_lr = generator_lr
        self.discriminator_lr = discriminator_lr
        
        # Build models
        self.generator = self._build_generator()
        self.discriminator = self._build_discriminator()
        
        # Compile models
        self._compile_models()
        
        # Training history
        self.history = {
            'generator_loss': [],
            'discriminator_loss': [],
            'discriminator_accuracy': []
        }
    
    def _build_generator(self) -> keras.Model:
        """Build the generator model"""
        
        # Input layers
        noise_input = layers.Input(shape=(self.latent_dim,), name='noise_input')
        condition_input = layers.Input(shape=(self.sequence_length, self.n_features), name='condition_input')
        
        # Process noise
        noise = layers.Dense(256)(noise_input)
        noise = layers.BatchNormalization()(noise)
        noise = layers.LeakyReLU(alpha=0.2)(noise)
        noise = layers.Dropout(0.3)(noise)
        
        noise = layers.Dense(512)(noise)
        noise = layers.BatchNormalization()(noise)
        noise = layers.LeakyReLU(alpha=0.2)(noise)
        noise = layers.Dropout(0.3)(noise)
        
        # Reshape noise for sequence generation
        noise = layers.Dense(self.sequence_length * 128)(noise)
        noise = layers.Reshape((self.sequence_length, 128))(noise)
        
        # Process condition (historical features)
        condition = layers.LSTM(128, return_sequences=True)(condition_input)
        condition = layers.Dropout(0.3)(condition)
        condition = layers.LSTM(128, return_sequences=True)(condition)
        condition = layers.Dropout(0.3)(condition)
        
        # Combine noise and condition
        combined = layers.Concatenate()([noise, condition])
        
        # Generate sequences
        x = layers.LSTM(256, return_sequences=True)(combined)
        x = layers.Dropout(0.3)(x)
        x = layers.BatchNormalization()(x)
        
        x = layers.LSTM(128, return_sequences=True)(x)
        x = layers.Dropout(0.3)(x)
        x = layers.BatchNormalization()(x)
        
        x = layers.LSTM(64, return_sequences=True)(x)
        x = layers.Dropout(0.2)(x)
        
        # Output layer - predict price sequences
        output = layers.TimeDistributed(layers.Dense(1, activation='linear'))(x)
        output = layers.Reshape((self.sequence_length,))(output)
        
        model = keras.Model([noise_input, condition_input], output, name='generator')
        
        logger.info("Generator model built")
        return model
    
    def _build_discriminator(self) -> keras.Model:
        """Build the discriminator model"""
        
        # Input layers
        sequence_input = layers.Input(shape=(self.sequence_length,), name='sequence_input')
        condition_input = layers.Input(shape=(self.sequence_length, self.n_features), name='condition_input')
        
        # Process price sequence
        sequence = layers.Reshape((self.sequence_length, 1))(sequence_input)
        sequence = layers.Conv1D(64, 3, padding='same')(sequence)
        sequence = layers.LeakyReLU(alpha=0.2)(sequence)
        sequence = layers.Dropout(0.3)(sequence)
        
        sequence = layers.Conv1D(128, 3, padding='same')(sequence)
        sequence = layers.LeakyReLU(alpha=0.2)(sequence)
        sequence = layers.Dropout(0.3)(sequence)
        
        sequence = layers.GlobalMaxPooling1D()(sequence)
        
        # Process condition
        condition = layers.LSTM(64)(condition_input)
        condition = layers.Dropout(0.3)(condition)
        
        # Combine sequence and condition
        combined = layers.Concatenate()([sequence, condition])
        
        # Classification layers
        x = layers.Dense(256)(combined)
        x = layers.LeakyReLU(alpha=0.2)(x)
        x = layers.Dropout(0.3)(x)
        
        x = layers.Dense(128)(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        x = layers.Dropout(0.3)(x)
        
        x = layers.Dense(64)(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        x = layers.Dropout(0.2)(x)
        
        # Output layer - real/fake classification
        output = layers.Dense(1, activation='sigmoid')(x)
        
        model = keras.Model([sequence_input, condition_input], output, name='discriminator')
        
        logger.info("Discriminator model built")
        return model
    
    def _compile_models(self):
        """Compile the models"""
        
        # Optimizers
        generator_optimizer = keras.optimizers.Adam(
            learning_rate=self.generator_lr,
            beta_1=0.5,
            beta_2=0.9
        )
        
        discriminator_optimizer = keras.optimizers.Adam(
            learning_rate=self.discriminator_lr,
            beta_1=0.5,
            beta_2=0.9
        )
        
        # Compile discriminator
        self.discriminator.compile(
            optimizer=discriminator_optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        # Build combined model for generator training
        self.discriminator.trainable = False
        
        noise_input = layers.Input(shape=(self.latent_dim,))
        condition_input = layers.Input(shape=(self.sequence_length, self.n_features))
        
        generated_sequence = self.generator([noise_input, condition_input])
        validity = self.discriminator([generated_sequence, condition_input])
        
        self.combined = keras.Model([noise_input, condition_input], validity, name='combined')
        self.combined.compile(
            optimizer=generator_optimizer,
            loss='binary_crossentropy'
        )
        
        self.discriminator.trainable = True
        
        logger.info("Models compiled successfully")
    
    def train(self, 
              X_train: np.ndarray, 
              y_train: np.ndarray,
              epochs: int = 1000,
              batch_size: int = 32,
              save_interval: int = 100,
              model_dir: str = 'models') -> dict:
        """
        Train the GAN model
        
        Args:
            X_train: Training features (batch_size, sequence_length, n_features)
            y_train: Training targets (batch_size, sequence_length)
            epochs: Number of training epochs
            batch_size: Batch size
            save_interval: Interval to save models
            model_dir: Directory to save models
            
        Returns:
            Training history
        """
        
        # Create model directory
        os.makedirs(model_dir, exist_ok=True)
        
        # Labels for training
        real_labels = np.ones((batch_size, 1))
        fake_labels = np.zeros((batch_size, 1))
        
        logger.info(f"Starting GAN training for {epochs} epochs")
        
        for epoch in range(epochs):
            
            # ---------------------
            #  Train Discriminator
            # ---------------------
            
            # Sample real data
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            real_sequences = y_train[idx]
            real_conditions = X_train[idx]
            
            # Generate fake data
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            fake_sequences = self.generator.predict([noise, real_conditions], verbose=0)
            
            # Train discriminator
            d_loss_real = self.discriminator.train_on_batch([real_sequences, real_conditions], real_labels)
            d_loss_fake = self.discriminator.train_on_batch([fake_sequences, real_conditions], fake_labels)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            
            # ---------------------
            #  Train Generator
            # ---------------------
            
            # Generate noise
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            
            # Sample conditions
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            conditions = X_train[idx]
            
            # Train generator (via combined model)
            g_loss = self.combined.train_on_batch([noise, conditions], real_labels)
            
            # Save training progress
            self.history['generator_loss'].append(g_loss)
            self.history['discriminator_loss'].append(d_loss[0])
            self.history['discriminator_accuracy'].append(d_loss[1])
            
            # Print progress
            if epoch % 100 == 0:
                logger.info(f"Epoch {epoch}/{epochs} - D_loss: {d_loss[0]:.4f}, D_acc: {d_loss[1]:.4f}, G_loss: {g_loss:.4f}")
            
            # Save models at intervals
            if epoch % save_interval == 0 and epoch > 0:
                self.save_models(f"{model_dir}/epoch_{epoch}")
                
        logger.info("Training completed")
        return self.history
    
    def predict(self, 
                X_condition: np.ndarray, 
                n_predictions: int = 1,
                use_noise: bool = True) -> np.ndarray:
        """
        Generate predictions using the trained generator
        
        Args:
            X_condition: Condition data (batch_size, sequence_length, n_features)
            n_predictions: Number of predictions per condition
            use_noise: Whether to use noise (for stochastic predictions)
            
        Returns:
            Generated sequences
        """
        batch_size = X_condition.shape[0]
        predictions = []
        
        for _ in range(n_predictions):
            if use_noise:
                noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            else:
                noise = np.zeros((batch_size, self.latent_dim))
            
            pred = self.generator.predict([noise, X_condition], verbose=0)
            predictions.append(pred)
        
        return np.array(predictions)
    
    def save_models(self, filepath_prefix: str):
        """Save generator and discriminator models"""
        self.generator.save(f"{filepath_prefix}_generator.h5")
        self.discriminator.save(f"{filepath_prefix}_discriminator.h5")
        logger.info(f"Models saved with prefix: {filepath_prefix}")
    
    def load_models(self, filepath_prefix: str):
        """Load generator and discriminator models"""
        try:
            self.generator = keras.models.load_model(f"{filepath_prefix}_generator.h5")
            self.discriminator = keras.models.load_model(f"{filepath_prefix}_discriminator.h5")
            logger.info(f"Models loaded from prefix: {filepath_prefix}")
        except Exception as e:
            logger.error(f"Error loading models: {e}")
    
    def plot_training_history(self, save_path: str = None):
        """Plot training history"""
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        # Loss plot
        axes[0].plot(self.history['generator_loss'], label='Generator Loss')
        axes[0].plot(self.history['discriminator_loss'], label='Discriminator Loss')
        axes[0].set_title('GAN Training Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # Accuracy plot
        axes[1].plot(self.history['discriminator_accuracy'], label='Discriminator Accuracy')
        axes[1].set_title('Discriminator Accuracy')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Training history plot saved to {save_path}")
        else:
            plt.show()

class TimeSeriesGAN(StockGAN):
    """Specialized GAN for time series forecasting"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def forecast_next_steps(self, 
                           X_condition: np.ndarray,
                           n_steps: int = 5,
                           n_samples: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forecast next n steps with uncertainty quantification
        
        Args:
            X_condition: Historical condition data
            n_steps: Number of steps to forecast
            n_samples: Number of Monte Carlo samples
            
        Returns:
            Tuple of (mean predictions, std predictions)
        """
        batch_size = X_condition.shape[0]
        all_predictions = []
        
        # Generate multiple predictions for uncertainty estimation
        for _ in range(n_samples):
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            pred = self.generator.predict([noise, X_condition], verbose=0)
            all_predictions.append(pred[:, -n_steps:])  # Take last n_steps
        
        all_predictions = np.array(all_predictions)  # (n_samples, batch_size, n_steps)
        
        # Calculate mean and std
        mean_pred = np.mean(all_predictions, axis=0)
        std_pred = np.std(all_predictions, axis=0)
        
        return mean_pred, std_pred

def main():
    """Example usage"""
    # Create sample data
    sequence_length = 60
    n_features = 10
    n_samples = 1000
    
    # Generate synthetic time series data
    np.random.seed(42)
    X_train = np.random.randn(n_samples, sequence_length, n_features)
    y_train = np.random.randn(n_samples, sequence_length)
    
    # Initialize GAN
    gan = StockGAN(
        sequence_length=sequence_length,
        n_features=n_features,
        latent_dim=100
    )
    
    # Print model summaries
    print("Generator Architecture:")
    gan.generator.summary()
    print("\nDiscriminator Architecture:")
    gan.discriminator.summary()
    
    # Train the model (small example)
    history = gan.train(
        X_train, y_train,
        epochs=100,
        batch_size=32,
        save_interval=50
    )
    
    # Generate predictions
    test_conditions = X_train[:5]  # Use first 5 samples as test
    predictions = gan.predict(test_conditions, n_predictions=3)
    
    print(f"Generated predictions shape: {predictions.shape}")
    
    # Plot training history
    gan.plot_training_history()

if __name__ == "__main__":
    main()