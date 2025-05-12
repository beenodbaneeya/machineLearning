import torch
import torch.nn as nn
import torch.optim as optim
from torch.profiler import profile, ProfilerActivity
from model import SimpleModel
from dataloader import get_mnist_dataloader

def train(epochs=5, batch_size=32, num_workers=4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    train_loader = get_mnist_dataloader(batch_size, num_workers)

    prof = None
    try:
        for epoch in range(epochs):
            if epoch == 2:  # Profile only epoch 2
                print("Starting profiler...")
                prof = profile(
                    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                    record_shapes=True,
                    with_stack=True
                )
                prof.start()

            model.train()
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

                if batch_idx % 100 == 0:
                    print(f"Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item()}")

            # Stop profiler ONLY after epoch 2
            if epoch == 2 and prof is not None:
                prof.stop()
                print("Profiler stopped. Exporting trace...")
                prof.export_chrome_trace(f"trace_single_gpu_workers_{num_workers}.json")
    finally:
        pass

if __name__ == "__main__":
    train(num_workers=4)
