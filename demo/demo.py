import moabb
from moabb import benchmark
from moabb.datasets import BNCI2014_001

results = benchmark(
    pipelines="/kaggle/working/ShallowConvNet/demo/pipeline.yml",
    evaluations=["WithinSession"],
    paradigms=["MotorImagery"],
    include_datasets=[BNCI2014_001()],
    results="/kaggle/working/ShallowConvNet/demo/results/",
    overwrite=False,
    output="/kaggle/working/ShallowConvNet/demo/benchmark/",
    plot=False,
)