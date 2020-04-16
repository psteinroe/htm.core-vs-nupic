import math

from htm.bindings.sdr import SDR, Metrics
from htm.encoders.rdse import RDSE, RDSE_Parameters
from htm.encoders.date import DateEncoder
from htm.bindings.algorithms import SpatialPooler
from htm.bindings.algorithms import TemporalMemory
from htm.algorithms.anomaly_likelihood import AnomalyLikelihood
from htm.bindings.algorithms import Predictor

parameters_numenta_comparable = {
    # there are 2 (3) encoders: "value" (RDSE) & "time" (DateTime weekend, timeOfDay)
    'enc': {
        "value":  # RDSE for value
            {
                'resolution': 0.001,
                'size': 4000,
                'sparsity': 0.10
            },
        "dateTimeTime":  # DateTime for timestamps
            {
                'season': (1, 30),  # represents months, each "season" is 30 days
                'timeOfDay': (21, 9.49),  # 40 on bits for each hour
                'dayOfWeek': 20,  # this field has most significant impact, as incorporates (day + hours)
                'weekend': 0,  # TODO try impact of weekend
            },
        "integerTime":  # RDSE for integer timestamps
            {
                'resolution': 0.001,
                'size': 4000,
                'sparsity': 0.10
            }
    },
    'predictor': {'sdrc_alpha': 0.1},
    'sp': {
        'boostStrength': 0.0,
        'columnCount': 2048,
        'localAreaDensity': 40 / 2048,
        'potentialPct': 0.4,
        'synPermActiveInc': 0.003,
        'synPermConnected': 0.2,
        'synPermInactiveDec': 0.0005},
    'tm': {
        'activationThreshold': 13,
        'cellsPerColumn': 32,
        'initialPerm': 0.21,
        'maxSegmentsPerCell': 128,
        'maxSynapsesPerSegment': 32,
        'minThreshold': 10,
        'newSynapseCount': 20,
        'permanenceDec': 0.1,
        'permanenceInc': 0.1},
    'anomaly': {
        'likelihood': {
            'probationaryPct': 0.1,
            'reestimationPeriod': 1,  # reestimate gaussian distr every iteration default value based on empiric
            # observations from numenta. the number of iterations required for the algorithm to learn the basic
            # patterns in the dataset and for the anomaly score to 'settle down'.
            # default: 100, gets reestimated every reestimationPeriod, so just leave it like that
        }
    }
}


class HTMCoreDetector(object):
    def __init__(self, inputMin, inputMax, probationaryPeriod, *args, **kwargs):
        self.inputMin = inputMin
        self.inputMax = inputMax
        self.probationaryPeriod = probationaryPeriod
        ## API for controlling settings of htm.core HTM detector:

        # Set this to False if you want to get results based on raw scores
        # without using AnomalyLikelihood. This will give worse results, but
        # useful for checking the efficacy of AnomalyLikelihood. You will need
        # to re-optimize the thresholds when running with this setting.
        self.use_likelihood = True
        self.verbose = False
        self.anomaly_likelihood = None

        # Set this to true if you want to use DateTime timestamps
        self.use_datetime_timestamps = True

        ## internal members
        # (listed here for easier understanding)
        # initialized in `initialize()`
        self.enc_timestamp = None
        self.enc_value = None
        self.sp = None
        self.tm = None
        self.an_like = None
        # optional debug info
        self.enc_info = None
        self.sp_info = None
        self.tm_info = None
        # required for spatial anomaly
        self.min_val = None
        self.max_val = None
        # internal helper variables:
        self.inputs_ = []
        self.iteration_ = 0

    def initialize(self):
        # toggle parameters here
        # parameters = default_parameters
        parameters = parameters_numenta_comparable

        # setup Enc, SP, TM, Likelihood
        # Make the Encoders.  These will convert input data into binary representations.
        if self.use_datetime_timestamps:
            self.enc_timestamp = DateEncoder(timeOfDay=parameters["enc"]["dateTimeTime"]["timeOfDay"])#,
                                             #weekend=parameters["enc"]["dateTimeTime"]["weekend"],
                                             #dayOfWeek=parameters["enc"]["dateTimeTime"]["dayOfWeek"],
                                             #season=parameters["enc"]["dateTimeTime"]["season"])
        else:
            timestamp_scalar_encoder_params = RDSE_Parameters()
            timestamp_scalar_encoder_params.size = parameters["enc"]["integerTime"]["size"]
            timestamp_scalar_encoder_params.sparsity = parameters["enc"]["integerTime"]["sparsity"]
            timestamp_scalar_encoder_params.resolution = parameters["enc"]["integerTime"]["resolution"]
            self.enc_timestamp = RDSE(timestamp_scalar_encoder_params)

        val_scalar_encoder_params = RDSE_Parameters()
        val_scalar_encoder_params.size = parameters["enc"]["value"]["size"]
        val_scalar_encoder_params.sparsity = parameters["enc"]["value"]["sparsity"]
        #val_scalar_encoder_params.resolution = max(0.001, (self.inputMax - self.inputMin) / 130)
        #val_scalar_encoder_params.resolution = 0.5
        val_scalar_encoder_params.resolution = parameters["enc"]["value"]["resolution"]
        self.enc_value = RDSE(val_scalar_encoder_params)

        encoding_width = (self.enc_timestamp.size + self.enc_value.size)
        self.enc_info = Metrics([encoding_width], 999999999)

        # Make the HTM.  SpatialPooler & TemporalMemory & associated tools.
        # SpatialPooler
        sp_params = parameters["sp"]
        self.sp = SpatialPooler(
            inputDimensions=(encoding_width,),
            columnDimensions=(sp_params["columnCount"],),
            potentialPct=sp_params["potentialPct"],
            potentialRadius=encoding_width,
            globalInhibition=True,
            localAreaDensity=sp_params["localAreaDensity"],
            synPermInactiveDec=sp_params["synPermInactiveDec"],
            synPermActiveInc=sp_params["synPermActiveInc"],
            synPermConnected=sp_params["synPermConnected"],
            boostStrength=sp_params["boostStrength"],
            wrapAround=True
        )
        self.sp_info = Metrics(self.sp.getColumnDimensions(), 999999999)

        # Temporal Memory
        tm_params = parameters["tm"]
        self.tm = TemporalMemory(
            columnDimensions=(sp_params["columnCount"],),
            cellsPerColumn=tm_params["cellsPerColumn"],
            activationThreshold=tm_params["activationThreshold"],
            initialPermanence=tm_params["initialPerm"],
            connectedPermanence=sp_params["synPermConnected"],
            minThreshold=tm_params["minThreshold"],
            maxNewSynapseCount=tm_params["newSynapseCount"],
            permanenceIncrement=tm_params["permanenceInc"],
            permanenceDecrement=tm_params["permanenceDec"],
            predictedSegmentDecrement=0.0,
            maxSegmentsPerCell=tm_params["maxSegmentsPerCell"],
            maxSynapsesPerSegment=tm_params["maxSynapsesPerSegment"]
        )
        self.tm_info = Metrics([self.tm.numberOfCells()], 999999999)

        # setup likelihood, such as in original NAB implementation
        if self.use_likelihood:
            an_params = parameters["anomaly"]["likelihood"]
            learningPeriod = int(math.floor(self.probationaryPeriod / 2.0))
            self.anomaly_likelihood = AnomalyLikelihood(
                learningPeriod=learningPeriod,
                estimationSamples=self.probationaryPeriod - learningPeriod,
                reestimationPeriod=an_params["reestimationPeriod"])

        return parameters

    def handle_record(self, ts, val):
        ## run data through our model pipeline: enc -> SP -> TM -> Anomaly
        self.inputs_.append(val)
        self.iteration_ += 1

        # 1. Encoding
        # Call the encoders to create bit representations for each value. These are SDR objects.
        date_bits = self.enc_timestamp.encode(ts)
        value_bits = self.enc_value.encode(float(val))
        # Concatenate all these encodings into one large encoding for Spatial Pooling.
        encoding = SDR(self.enc_timestamp.size + self.enc_value.size).concatenate([value_bits, date_bits])
        self.enc_info.addData(encoding)

        # 2. Spatial Pooler
        # Create an SDR to represent active columns, This will be populated by the
        # compute method below. It must have the same dimensions as the Spatial Pooler.
        active_columns = SDR(self.sp.getColumnDimensions())
        # Execute Spatial Pooling algorithm over input space.
        self.sp.compute(encoding, True, active_columns)
        self.sp_info.addData(active_columns)

        # 3. Temporal Memory
        # Execute Temporal Memory algorithm over active mini-columns.
        self.tm.compute(active_columns, learn=True)
        self.tm_info.addData(self.tm.getActiveCells().flatten())

        # 4. Anomaly
        # handle spatial, contextual (raw, likelihood) anomalies

        # -temporal (raw)
        raw = self.tm.anomaly
        temporal_anomaly = raw

        if self.use_likelihood:
            # Compute log(anomaly likelihood)
            like = self.anomaly_likelihood.anomalyProbability(val, raw, ts)
            log_score = self.anomaly_likelihood.computeLogLikelihood(like)
            temporal_anomaly = log_score  # TODO optional: TM to provide anomaly {none, raw, likelihood}, compare correctness with the py anomaly_likelihood

        anomaly_score = temporal_anomaly  # this is the "main" anomaly, compared in NAB

        # 5. print stats
        if self.verbose and self.iteration_ % 1000 == 0:
            print(self.enc_info)
            print(self.sp_info)
            print(self.tm_info)

        return anomaly_score, raw
