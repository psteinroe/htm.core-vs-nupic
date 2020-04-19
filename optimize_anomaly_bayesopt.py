import subprocess
import os
import json
from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
from bayes_opt.util import load_logs


def default_params():
    return {
        "enc": {
            "value": {
                # "resolution": 0.9, calculate by max(0.001, (maxVal - minVal) / numBuckets) where numBuckets = 130
                "size": 400,
                "activeBits": 21,
                "seed": 5,
            },
            "time": {
                "timeOfDay": (21, 9.49),
            }
        },
        "sp": {
            # inputDimensions: use width of encoding
            "columnDimensions": 2048,
            # "potentialRadius": use width of encoding
            "potentialPct": 0.8,
            "globalInhibition": True,
            "localAreaDensity": 0.10354931536188854,  # optimize this one
            "stimulusThreshold": 0,
            "synPermInactiveDec": 0.0005,
            "synPermActiveInc": 0.003,
            "synPermConnected": 0.2,
            "boostStrength": 0.0,
            "wrapAround": True,
            "minPctOverlapDutyCycle": 0.001,
            "dutyCyclePeriod": 1000,
            "seed": 5,
        },
        "tm": {
            "columnDimensions": 2048,
            "cellsPerColumn": 32,
            "activationThreshold": 20,
            "initialPermanence": 0.24,
            "connectedPermanence": 0.5,
            "minThreshold": 13,
            "maxNewSynapseCount": 31,
            "permanenceIncrement": 0.04,
            "permanenceDecrement": 0.008,
            "predictedSegmentDecrement": 0.001,
            "maxSegmentsPerCell": 128,
            "maxSynapsesPerSegment": 128,
            "seed": 5,
        },
        "anomaly": {
            "likelihood": {
                "probationaryPct": 0.1,
                "reestimationPeriod": 100
            }
        }
    }


def get_res():
    with open('NAB/results/final_results.json') as json_file:
        data = json.load(json_file)
        return data.get('htmcore').get('standard')


def target_func(localAreaDensity):
    params = default_params()

    params['sp']['localAreaDensity'] = localAreaDensity

    with open(os.path.join('NAB', 'nab', 'detectors', 'htmcore', 'params.json'), 'w') as outfile:
        json.dump(params, outfile)

    print('Starting NAB with localAreaDensity', localAreaDensity)

    try:
        subprocess.check_call(
            "cd ./NAB && python ./run.py -d htmcore --skipConfirmation --detect --optimize --score --normalize",
            shell=True)
        print('Reading Results...')
        score = get_res()
        print('localAreaDensity', ':', localAreaDensity, '-->', score)
        return score
    except subprocess.CalledProcessError as err:
        print(err)
        raise err
    except OSError as err:
        print(err)
        raise err

def optimize_local_area_density():
    # optimize localAreaDensity
    bounds = {
        'localAreaDensity': (0.01, 0.15),
    }

    optimizer = BayesianOptimization(
        f=target_func,
        pbounds=bounds,
        random_state=1,
    )

    if os.path.isfile('./local_area_density_optimization_logs_base.json'):
        print('Loading Logs...')
        load_logs(optimizer, logs=["./local_area_density_optimization_logs_base.json"]);

    logger = JSONLogger(path="./local_area_density_optimization_logs.json")
    optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)

    val = 0.02
    while val <= 0.04:
        print('Adding', val)
        optimizer.probe(
            params={
                'localAreaDensity': val,
            },
            lazy=True,
        )
        val = round(val + 0.001, 3)

    print('Starting optimization...')

    optimizer.maximize(
        init_points=20,
        n_iter=50,
    )

    print(optimizer.max)


if __name__ == "__main__":
    optimize_local_area_density()
