#log model histograms goal: inspect weight distributions and how they evolve. in torch
import torch
from tensorboardX import SummaryWriter

class simplenn(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(10,50)
        self.fc2 = torch.nn.Linear(50,2)
    def forward(self,x):
        x = torch.relu(fc1(x))
        x = fc2(x)
        return x


model = simplenn()
writer = SummaryWriter('runs/hists_experiment')

for epoch in range(20):
    #fake training step
    for param in model.parameters():
        param.data += 0.01 * torch.randn_like(param)

    for name,param in model.named_parameters():
        writer.add_histogram(name, param.detach().cpu().numpy(), epoch)
        writer.add_scalar('metric/dummy_epoch', epoch, epoch)

writer.close()
