import os
import tensorflow as tf

print(f'\n=== GPU Verification ===')
print(f'SLURM_PROCID: {os.getenv("SLURM_PROCID")}')
print(f'SLURM_LOCALID: {os.getenv("SLURM_LOCALID")}')

gpus = tf.config.list_physical_devices('GPU')
print(f'Visible GPUs: {len(gpus)}')
for i, gpu in enumerate(gpus):
    print(f'GPU {i}: {gpu}')

if len(gpus) > 1:
    print('\nTesting multi-GPU communication...')
    strategy = tf.distribute.MirroredStrategy()

    def compute_fn():
        t = tf.ones([10])
        return t

    # Run the function on all replicas
    per_replica_t = strategy.run(compute_fn)

    # Now you can reduce
    reduced = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_t, axis=None)

    print(f'NCCL reduce test: {reduced.numpy()}')
    print(f'Number of replicas: {strategy.num_replicas_in_sync}')
    print('\nMulti-GPU test passed!')
else:
    print('\nWarning: Only 1 GPU detected')
