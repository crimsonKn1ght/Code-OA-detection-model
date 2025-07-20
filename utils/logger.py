import os
from tensorboardX import SummaryWriter
import datetime

def get_writer(log_dir):
    
    run_dir = os.path.join(log_dir, f"exp_{datetime.datetime.now():%Y%m%d_%H%M%S}")
    return SummaryWriter(log_dir=run_dir)
