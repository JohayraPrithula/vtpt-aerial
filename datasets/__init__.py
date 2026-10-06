# Importing the dataset modules triggers their @DATASET_REGISTRY.register()
# decorators so the trainer can build them by name.
from datasets.officehome_multitarget import OfficeHomeMultiTarget
from datasets.aerial_multitarget import AerialMultiTarget
