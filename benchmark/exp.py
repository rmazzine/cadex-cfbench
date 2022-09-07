import sys
from collections import defaultdict

sys.path.append('../')

import time

import numpy as np
import pandas as pd
from cfbench.cfbench import BenchmarkCF, TOTAL_FACTUAL

from Cadex import Cadex
from benchmark.utils import timeout, TimeoutError

# Get initial and final index if provided
if len(sys.argv) == 3:
    initial_idx = sys.argv[1]
    final_idx = sys.argv[2]
else:
    initial_idx = 0
    final_idx = TOTAL_FACTUAL

# Create Benchmark Generator
benchmark_generator = BenchmarkCF(
    output_number=2,
    show_progress=True,
    disable_tf2=True,
    disable_gpu=True,
    initial_idx=int(initial_idx),
    final_idx=int(final_idx)).create_generator()

# The Benchmark loop
cadex_current_dataset = None
for benchmark_data in benchmark_generator:
    # Get factual array
    factual_array = benchmark_data['factual_oh']

    # Get train data
    train_data = benchmark_data['df_oh_train']

    # Get columns info
    columns = list(train_data.columns)[:-1]

    # Get factual row as pd.Series
    factual_row = pd.Series(benchmark_data['factual_oh'], index=columns)

    # Get factual class
    fc = benchmark_data['factual_class']

    # Get Keras TensorFlow model
    model = benchmark_data['model']

    # Get Evaluator
    evaluator = benchmark_data['cf_evaluator']

    if cadex_current_dataset != benchmark_data['dsname']:
        columns = list(train_data.columns)[:-1]
        columns_prefix = [col.split('_')[0] for col in columns]

        # Categorical features
        cat_feats = benchmark_data['cat_feats']

        cat_idx = defaultdict(list)
        for cat in cat_feats:
            for c_idx, c in enumerate(columns):
                if int(cat) == int(c.split('_')[0]):
                    cat_idx[cat].append(c_idx)

        cat_idx = list(cat_idx.values())

        # Remove binary features since when they raise an error when selected
        cat_idx = [c for c in cat_idx if len(c) > 1]

        explainer = Cadex(model, categorical_attributes=cat_idx if cat_feats else None)

        cadex_current_dataset = benchmark_data['dsname']


    @timeout(600)
    def generate_cf():
        try:
            # Create CF using CADEX's explainer and measure generation time
            start_generation_time = time.time()
            cf_gen = explainer.train(pd.DataFrame([np.array(factual_array)]), 1 if fc == 0 else 0, 2)
            cf_generation_time = time.time() - start_generation_time

            if cf_gen[0] is None:
                cf = factual_array
            elif type(cf_gen[0]) == pd.DataFrame:
                cf = cf_gen[0].iloc[0].to_list()
            else:
                cf = cf_gen[0][0].tolist()

        except Exception as e:
            print('Error generating CF')
            print(e)
            # In case the CF generation fails, return same as factual
            cf = factual_row.to_list()
            cf_generation_time = np.NaN

        # Evaluate CF
        evaluator(
            cf_out=cf,
            algorithm_name='cadex',
            cf_generation_time=cf_generation_time,
            save_results=True)

    try:
        generate_cf()
    except TimeoutError:
        print('Timeout generating CF')
        # If CF generation time exceeded the limit
        evaluator(
            cf_out=factual_row.to_list(),
            algorithm_name='dice',
            cf_generation_time=np.NaN,
            save_results=True)
