from torchgeo.datasets import EuroSAT, EuroSATSpatial

ds = EuroSAT(root="data", download=True, checksum=True)
spatial_ds = EuroSATSpatial(root="data", download=True, checksum=True)
