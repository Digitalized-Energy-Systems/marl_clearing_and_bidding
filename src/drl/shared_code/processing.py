
import torch


def batch_to_tensors(batch: tuple, device, continuous: bool):
    """ Transform numpy arrays into torch tensors. """
    obss, acts, rewards, next_obss, dones = batch

    obss = torch.tensor(obss, dtype=torch.float).to(device)
    next_obss = torch.tensor(
        next_obss, dtype=torch.float).to(device)
    if continuous is True:
        acts = torch.tensor(acts, dtype=torch.float).to(device)
    else:
        acts = torch.tensor(acts, dtype=torch.long).to(device)
    rewards = torch.tensor(rewards, dtype=torch.float).to(device)
    dones = torch.BoolTensor(dones).to(device)

    return obss, acts, rewards, next_obss, dones
