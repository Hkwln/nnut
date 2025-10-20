from tensorboardX import SummaryWriter
import math, time, random


writer = SummaryWriter('runs/experiment_1')

for step in range(100):
    loss = math.exp(-step/20.0)+ random.uniform(-0.02, 0.02)
    acc = 1-loss + math.exp(random.uniform(-0.01, 0.01))
    writer.add_scalar('train/loss', loss, step)
    writer.add_scalar('train/acc', acc, step)

writer.close()
