import json
from concurrent.futures import ProcessPoolExecutor as Pool

from fonts import FONTS_BIT_TUPLES
from functions import *
from optimization import MomentumOptimizer, AdamOptimizer
from autoencoder import Autoencoder
def get_activation_func(activation_type: str)-> callable:
    if activation_type == "sigmoid":
        return sigmoid, sigmoid_derivative, sigmoid_normalization
    elif activation_type == "hiperbolic":
        return hiperbolic, hiperbolic_derivative, hiperbolic_normalization
    else:
        raise Exception("Invalid activation type")

def get_optimizer(optimizer_type: str, amount_of_layers: int)-> callable:
    if optimizer_type == "momentum":
        return MomentumOptimizer(amount_of_layers)
    elif optimizer_type == "adam":
        return AdamOptimizer(amount_of_layers)
    else:
        raise Exception("Invalid optimizer type")

def run_test(test_config: dict, test_eval: dict):

    _encoder_layers = test_config['architecture']['encoder']
    _latent_space_dim = test_config['architecture']['latent_space']
    _decoder_layers = test_config['architecture']['decoder']
    amount_of_layers = len(_encoder_layers) + 1 + len(_decoder_layers) + 1
    (activation_function, derivation_function, normalization_function) = get_activation_func(
        test_config['architecture']['activation_function'])
    complete_layers = _encoder_layers + [_latent_space_dim] + _decoder_layers
    results = {'layers': _encoder_layers, 'data': [], 'epochs': test_config['training']['epochs']}

    def runnable(i):
        data = [(font, font) for font in FONTS_BIT_TUPLES]
        autoencoder = Autoencoder(
            encoder_layers=_encoder_layers,
            latent_space_dim=_latent_space_dim,
            decoder_layers=_decoder_layers,
            data=data,
            activation_function=activation_function,
            derivation_function=derivation_function,
            normalization_function=normalization_function,
            optimization=get_optimizer(test_config['architecture']['optimizer'], amount_of_layers)
        )
        print(f"Training {complete_layers}{i}...")
        errors = autoencoder.train(test_config['training']['epochs'], test_config['training']['error_cutoff'])
        print(f"Finished {complete_layers}{i}...")
        results['data'].append({'iteration': i,'errors': errors})

    for i in range(test_eval['repetitions']):
        runnable(i)
    return results



def unzip(args):
    return run_test(*args)

if __name__ == '__main__':
    config = json.load(open("./autoencoder_test_config.json", "r"))
    output = {'tests': []}
    with Pool(max_workers = len(config['tests'])) as pool:
        for exec_result in pool.map(unzip, [(test, config['evaluation']) for test in config['tests']]):
            output['tests'].append(exec_result)
        json.dump(output, open(f"{config['output']}", "w"))