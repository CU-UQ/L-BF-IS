# Langevin Bi-fidelity Importance Sampling

The paper is available at [arXiv](https://arxiv.org/abs/2503.17796).

## Referecing this code

If you use this code in any of your own work, please reference our paper:
```
@misc{cheng2025langevin,
      title={Langevin Bi-fidelity Importance Sampling for Failure Probability Estimation}, 
      author={Nuojin Cheng and Alireza Doostan},
      year={2025},
      eprint={2503.17796},
      archivePrefix={arXiv},
      primaryClass={stat.CO}
}
```

## Description of code

### Download Data

Download data from [here](https://zenodo.org/records/15104070?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6ImJhMjAyYzFhLTAzODUtNDMyZS1iOTY3LTYyODVhNGE0YzA0MSIsImRhdGEiOnt9LCJyYW5kb20iOiI5NTFkZDgzYWEwOTU4N2MxZDIzZDg1NWQyMmMyZWIwNyJ9.VCr7bANk_2FZhtg7Aak8jT3Zgc4maw0oXp8hkNzPLAoHp6Q5POPcCJKD_dORz-bOfYa5d9S3lKCK_YZnC6ZNaA) as `./data`

### Reproduce results

To reproduce results in our paper, please follow the pipeline in the Jupyter notebook `./src/reproduce.ipynb`

### Explore methodology

To explore the L-BF-IS estimator and our proposed method, please follow our provided example of Borehold problem in the Jupyter notebook `./src/Borehole.ipynb`
