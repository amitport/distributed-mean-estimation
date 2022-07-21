import torch
import cmprs.config as config


def lognormal_generator(d):
    lognormal_distribution = torch.distributions.LogNormal(0, 1)
    return lambda n_clients: lognormal_distribution.sample([n_clients, d]).to(config.device)


def normal_generator(d):
    dist = torch.distributions.Normal(0, 1)
    return lambda n_clients: dist.sample([n_clients, d]).to(config.device)


VEC_GENERATORS = {
    'lognormal': lognormal_generator,
    'normal': normal_generator,
}
