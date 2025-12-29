from moabb.datasets import BNCI2014_001
from moabb.paradigms import MotorImagery

dataset = BNCI2014_001()
paradigm = MotorImagery(n_classes=4, fmin=0, fmax=38, tmin=-5, tmax=6.5)
epochs, y, metadata = paradigm.get_data(dataset=dataset, subjects=[1, 2, 3, 4, 5, 6, 7, 8, 9], return_epochs=True)

evokeds = {cond: epochs[cond].average() for cond in BNCI2014_001().event_id}

for cond, evk in evokeds.items():
    evk.plot(spatial_colors=True, show=True)
