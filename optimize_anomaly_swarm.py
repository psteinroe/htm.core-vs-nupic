import subprocess
import sys
import os
import json

# dont use

def default_params():
    return {
        "enc": {
            "value": {
                # "resolution": 0.9, calculate by max(0.001, (maxVal - minVal) / numBuckets) where numBuckets = 130
                "size": 400,
                "activeBits": 21
            },
            "time": {
                "timeOfDay": (21, 9.49),
            }
        },
        "sp": {
            # inputDimensions: use width of encoding
            "columnDimensions": 2048,
            # "potentialRadius": 999999, use width of encoding
            "potentialPct": 0.8,
            "globalInhibition": True,
            # "localAreaDensity": 0.1,  # optimize this one
            "stimulusThreshold": 0,
            "synPermInactiveDec": 0.0005,
            "synPermActiveInc": 0.003,
            "synPermConnected": 0.2,
            "boostStrength": 0.0,
            "wrapAround": True,
            "minPctOverlapDutyCycle": 0.001,
            "dutyCyclePeriod": 1000,
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


default_parameters = {
    'localAreaDensity': 0.10354931536188854,
}


def main(parameters=default_parameters, argv=None, verbose=True):
    # get params
    params = default_params()

    params['sp']['localAreaDensity'] = default_parameters['localAreaDensity']

    with open(os.path.join('NAB', 'nab', 'detectors', 'htmcore', 'params.json'), 'w') as outfile:
        json.dump(params, outfile)

    subprocess.call("cd ./NAB && python ./run.py -d htmcore --skipConfirmation --detect --optimize --score --normalize",
                    shell=True)
    return get_res()


if __name__ == '__main__':
    sys.exit(main() < 0.95)
