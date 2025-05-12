import os
import torch
import torch.nn as nn
from torch.profiler import profile, ProfilerActivity
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group  
from model import SimpleModel
from dataloader import get_mnist_dataloader

def ddp_setup():
    init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

def train(num_workers=4, batch_size=32, epochs=5):
    ddp_setup()
    rank = int(os.environ["LOCAL_RANK"])
    
    try:  
        model = SimpleModel().cuda()
        model = DDP(model, device_ids=[rank])
        optimizer = torch.optim.Adam(model.parameters())
        
        train_set = get_mnist_dataloader(batch_size, num_workers).dataset
        sampler = DistributedSampler(train_set)
        train_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True
        )

        prof = None
        for epoch in range(epochs):
            sampler.set_epoch(epoch)
            
            if epoch == 2 and rank == 0:
                prof = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True)
                prof.start()

            model.train()
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.cuda(), target.cuda()
                optimizer.zero_grad()
                output = model(data)
                loss = nn.functional.cross_entropy(output, target)
                loss.backward()
                optimizer.step()

                if batch_idx % 100 == 0 and rank == 0:
                    print(f"Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item()}")

            if epoch == 2 and rank == 0 and prof is not None:
                prof.stop()
                prof.export_chrome_trace(f"trace_ddp_workers_four{num_workers}.json")
    
    finally:  # Cleanup
        destroy_process_group()  

if __name__ == "__main__":
    train(num_workers=4)
