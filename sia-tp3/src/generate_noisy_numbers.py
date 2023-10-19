from src.ej_3_test_runner_with_different_datasets import apply_noise
from data.ej3_digitos import DATA_DIGITOS


def generate_noisy_numbers(n):
    total = []
    for i in range(n):
        total.extend([(apply_noise(.30 + i/50, data), exp) for (data, exp) in DATA_DIGITOS])
    print(total)

generate_noisy_numbers(5)