import moabb
from moabb import benchmark
from moabb.datasets import BNCI2014_001

results = benchmark(
    pipelines="./pipeline.yml",
    evaluations=["WithinSession"],
    paradigms=["MotorImagery"],
    include_datasets=[BNCI2014_001()],
    results="./results/",
    overwrite=False,
    output="./benchmark/",
    plot=False,
)