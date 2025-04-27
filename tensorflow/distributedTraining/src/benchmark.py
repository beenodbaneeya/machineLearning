import time
import json
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress INFO and WARNING logs
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard
from model import create_simple_nn
from data_utils import load_and_preprocess_data
from config import BenchmarkConfig
from utils import save_results


class DeviceBenchmark:
    def __init__(self, device_type):
        self.device_type = device_type
        self.results = {
            'device' : device_type,
            'metrics' : {},
            'timing' : {}
        }

    def setup_strategy(self):
        if self.device_type == 'GPU':
            strategy = tf.distribute.MirroredStrategy()
            print(f"Using {strategy.num_replicas_in_sync} GPU(s)")
            return strategy
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
            return None
        
    def train(self):
        start_time = time.time()

        strategy = self.setup_strategy()

        self.results['timing']['warmup'] = 0.0

        if strategy:
            with strategy.scope():
                model = create_simple_nn()
                train_dataset, test_dataset = load_and_preprocess_data()
        else:
            model = create_simple_nn()
            train_dataset, test_dataset = load_and_preprocess_data()

        # Setup callbacks
        callbacks = [
            TensorBoard(log_dir=BenchmarkConfig.GPU_LOG_DIR if self.device_type == 'GPU' else BenchmarkConfig.CPU_LOG_DIR)
        ]

        # Warm-up phase
        if self.device_type == 'GPU':
            print("\nRunning GPU warmup (1 epoch on small subset)...")
            warmup_start = time.time()

            warmup_dataset = train_dataset.take(1)

            model.fit(
                warmup_dataset,
                epochs=1,
                verbose=0
            )

            self.results['timing']['warmup'] = time.time() - warmup_start
            print(f"Warmup completed in {self.results['timing']['warmup']:.2f}s")

            # Main training
            train_start = time.time()

            history = model.fit(
                train_dataset,
                epochs=BenchmarkConfig.EPOCHS,
                validation_data=test_dataset,
                callbacks=callbacks,
                verbose=2
            )

            self.results['timing']['training'] = time.time() - train_start
            self.results['timing']['total'] = time.time() - start_time

            self.results['metrics'] = {
                'accuracy': history.history['accuracy'][-1],
                'val_accuracy': history.history['val_accuracy'][-1],
                'loss': history.history['loss'][-1],
                'val_loss': history.history['val_loss'][-1]
            }

            model_path = os.path.join(
                BenchmarkConfig.SAVE_DIR,
                f"{self.device_type.lower()}_model.keras"
            )

            model.save(model_path)
            self.results['model_path'] = model_path

            return self.results

def run_benchmark():
    results = {}


    # GPU benchmark
    if tf.config.list_physical_devices('GPU'):
        print("\n=== Running GPU Benchmark ===")
        gpu_bench = DeviceBenchmark('GPU')
        results['gpu'] = gpu_bench.train()
    else:
        results['gpu'] = None

    # CPU benchmark
    print("\n=== Running CPU benchmark ===")
    cpu_bench = DeviceBenchmark('CPU')
    results ['cpu'] = cpu_bench.train()


    # Save and compare results 
    save_results(results)
    print("\nBenchmark completed!")

    if results['gpu'] and results['cpu']:
        # Ensure results['gpu'] is not None
        if results['gpu']['timing']['training'] and results['cpu']['timing']['training']:
            speedup = results['cpu']['timing']['training'] / results['gpu']['timing']['training']
            print(f"\nGPU was {speedup:.1f}x faster than CPU")
        else:
            print("\nTraining time for GPU or CPU is missing.")
    else:
        print("\nSpeedup could not be calculated (GPU benchmark may have failed or no GPU available).")

    return results

    
if __name__ == "__main__":
    run_benchmark()
