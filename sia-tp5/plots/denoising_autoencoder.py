import json

from src.denoising_autoencoder import DenoisingAutoencoder
from data.fonts import FONTS_BIT_TUPLES
from src.autoencoder import Autoencoder
from src.functions import *
import plotly.graph_objects as go
from plots.various_plots import plot_error
from plotly.subplots import make_subplots
from latent_space import plot_latent_space
from src.optimization import MomentumOptimizer, AdamOptimizer
from concurrent.futures import ProcessPoolExecutor as Pool
from multiprocessing import Pool as MPool

LEARNING_CONSTANT = 10 ** -3  # -> *
BETA = 0.3
SAVE_WEIGHTS = True
ERROR_PLOT_TITLE = "Error by Steps. Adam"


def add_heatmap_trace(fig, original, created, colorscale, i):
    input_letter = np.reshape(np.array(original), [7, 5])
    output_letter = np.reshape(created, [7, 5])
    fig.add_trace(go.Heatmap(z=np.flipud(input_letter),
                             coloraxis="coloraxis"),
                  row=1 + i // 4, col=1 + 2 * (i % 4))
    fig.add_trace(go.Heatmap(z=np.flipud(output_letter),
                             coloraxis="coloraxis"),
                  row=1 + i // 4, col=2 + 2 * (i % 4))

def gaussian_noise(tuple, mean, std):
    new_tuple = []
    for bit in tuple:
        new_tuple.append(bit + np.random.normal(mean, std))
    return new_tuple

def salt_and_pepper(tuple, prob):
    new_tuple = []
    for bit in tuple:
        if np.random.random() < prob:
            new_tuple.append(1 - bit)
        else:
            new_tuple.append(bit)
    return new_tuple

def poisson(figure, figure_size, mean):
    noisemap = np.ones((figure_size)) * mean #mean
    noisy = figure + np.random.poisson(noisemap)
    return noisy

S_P_NOISE = 0.04

def get_noise_func(noise_type: dict)-> callable:
    if noise_type['type'] == "gaussian":
        return lambda font: gaussian_noise(font, noise_type['mean'], noise_type['std'])
    elif noise_type['type'] == "salt_and_pepper":
        return lambda font: salt_and_pepper(font, noise_type['prob'])
    elif noise_type['type'] == "poisson":
        return lambda font: poisson(font, len(font), noise_type['mean'])
    else:
        raise Exception("Invalid noise type")

def get_activation_func(activation_type: str)-> callable:
    if activation_type == "sigmoid":
        return (sigmoid, sigmoid_derivative, sigmoid_normalization)
    elif activation_type == "hiperbolic":
        return (hiperbolic, hiperbolic_derivative, hiperbolic_normalization)
    else:
        raise Exception("Invalid activation type")

def get_optimizer(optimizer_type: str, amount_of_layers: int)-> callable:
    if optimizer_type == "momentum":
        return MomentumOptimizer(amount_of_layers)
    elif optimizer_type == "adam":
        return AdamOptimizer(amount_of_layers)
    else:
        raise Exception("Invalid optimizer type")

def diff(tuple1, tuple2):
    return np.sum(np.abs(np.array(tuple1) - np.array(tuple2)))

def runnable(i, noise_type, name, epochs, error_cutoff, _encoder_layers, _latent_space_dim, _decoder_layers, activation_function, derivation_function, normalization_function, optimization):
    noisy_data = [(font, font) for font in FONTS_BIT_TUPLES]
    autoencoder = denoising_autoencoder(
        encoder_layers=_encoder_layers,
        latent_space_dim=_latent_space_dim,
        decoder_layers=_decoder_layers,
        data=noisy_data,
        activation_function=activation_function,
        derivation_function=derivation_function,
        normalization_function=normalization_function,
        optimization=optimization
    )
    print(f"Training {name}{i}...")
    autoencoder.train(epochs, error_cutoff)
    extra_noisy_data = [(noise_type(font), font) for font in FONTS_BIT_TUPLES]
    print(f"Plotting {name}{i}...")
    fig = make_subplots(rows=8, cols=8)
    colorscale = [[0, 'white'], [1, 'black']]

    correct = 0
    for i, (noisy, data) in enumerate(extra_noisy_data):
        result = autoencoder.run_input(noisy)
        discretized_result = (round(result[0]) + 1) / 2
        add_heatmap_trace(fig, noisy, discretized_result, colorscale, i)
        correct += 1 if diff(discretized_result, data) < 5 else 0
    
    test_results = {}
    test_results['correct'] = correct
    test_results['total'] = len(extra_noisy_data)
    test_results['accuracy'] = correct / len(extra_noisy_data)
    return { 'fig': fig, 'test_results': test_results }

def unzip2(args):
    return runnable(*args)

def run_test(test_config: dict, test_eval: dict, output_file: str):
    results = []
    noise_type = get_noise_func(test_config['noise_type'])
    _encoder_layers = test_config['architecture']['encoder']
    _latent_space_dim = test_config['architecture']['latent_space']
    _decoder_layers = test_config['architecture']['decoder']
    amount_of_layers = len(_encoder_layers) + 1 + len(_decoder_layers) + 1
    (activation_function, derivation_function, normalization_function) = get_activation_func(test_config['architecture']['activation_function'])
    optimization = get_optimizer(test_config['architecture']['optimizer'], amount_of_layers)
    funcs = []
    
    def runnable(i):
        noisy_data = [(noise_type(font), font) for font in FONTS_BIT_TUPLES]
        autoencoder = DenoisingAutoencoder(
            encoder_layers=_encoder_layers,
            latent_space_dim=_latent_space_dim,
            decoder_layers=_decoder_layers,
            data=noisy_data,
            activation_function=activation_function,
            derivation_function=derivation_function,
            normalization_function=normalization_function,
            optimization=optimization,
            noise_func=noise_type
        )
        
        print(f"Training {test_config['name']}{i}...")
        autoencoder.train(test_config['training']['epochs'], test_config['training']['error_cutoff'])
        extra_noisy_data = [(noise_type(font), font) for font in FONTS_BIT_TUPLES]
        print(f"Plotting {test_config['name']}{i}...")
        fig = make_subplots(rows=8, cols=8)
        colorscale = [[0, 'white'], [1, 'black']]

        correct = 0
        for i, (noisy, data) in enumerate(extra_noisy_data):
            result = autoencoder.run_input(noisy)
            discretized_result = (round(result[0]) + 1) / 2
            add_heatmap_trace(fig, noisy, discretized_result, colorscale, i)
            correct += 1 if diff(discretized_result, data) < 5 else 0
        
        test_results = {}
        test_results['correct'] = correct
        test_results['total'] = len(extra_noisy_data)
        test_results['accuracy'] = correct / len(extra_noisy_data)
        return { 'fig': fig, 'test_results': test_results, "name": f"{test_config['name']}{i}" }
    """
    with Pool(max_workers = len(config['tests'])) as pool:
        for result in pool.map(unzip, [(test, config['evaluation'], config['output']) for test in config['tests']]):
            print(result)
    """ 
    name = test_config['name']
    epochs = test_config['training']['epochs']
    error_cutoff = test_config['training']['error_cutoff']

    #         results = [result for result in inner_pool.map(unzip2, [(i, noise_type, name, epochs, error_cutoff, _encoder_layers, _decoder_layers, activation_function, derivation_function, normalization_function, optimization) for i in range(test_eval['repetitions'])])]

    # with MPool(processes=test_eval['repetitions']) as inner_pool:
    #     results = [result for result in inner_pool.map(runnable, [i for i in range(test_eval['repetitions'])])]
    #     return results
    for i in range(test_eval['repetitions']):
        result = runnable(i)
        results.append(result)
    return results
        
def process_config(filename: str) -> dict:
    try:
        with open(filename, "r") as f:
            config = json.load(f)
            return config
    except Exception as e:
        print(f"Error while loading config file: {e}")
        exit(1)

def unzip(args):
    return run_test(*args)

if __name__ == '__main__':
    # print(diff(FONTS_BIT_TUPLES[0], FONTS_BIT_TUPLES[2]))
    config = process_config("./plots/denoising_config.json")

    with Pool(max_workers = len(config['tests'])) as pool:
        arr = []
        for exec_result in pool.map(unzip, [(test, config['evaluation'], config['output']) for test in config['tests']]):
            result = []
            for i, r in enumerate(exec_result):
                result.append(r['test_results'])
                r['fig'].write_image(f"{r['name']}{i}.png")
            arr.append({ "label": exec_result[0]['name'], "results": result })
        json.dump(arr, open(f"{config['output']}", "w"))
    # for test in config['tests']:
    #     results = run_test(test, config['evaluation'], config['output'])
    #     for result in results:
    #         print([x['test_results'] for x in result])
            # result['fig'].write_image(f"{config['output']}/{test['name']}.png")
            # plot_error(result['test_results'], f"{config['output']}/{test['name']}_error.png")


