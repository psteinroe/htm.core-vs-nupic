import json
import requests
from datetime import timezone


class NupicDetector(object):

    def __init__(self, inputMin, inputMax, probationaryPeriod):
        self.inputMin = inputMin
        self.inputMax = inputMax
        self.probationaryPeriod = probationaryPeriod

    def initialize(self):
        r = requests.post(url="http://localhost:5000/api/init", json={
            'inputMin': self.inputMin,
            'inputMax': self.inputMax,
            'probationaryPeriod': self.probationaryPeriod
        })

    def handleRecord(self, ts, val):
        r = requests.post(url="http://localhost:5000/api/handleRecord", json={
            'timestamp': ts.replace(tzinfo=timezone.utc).timestamp(),
            'value': val
        })
        r_json = json.loads(r.text)
        return r_json['anomalyScore'], r_json['rawScore']
