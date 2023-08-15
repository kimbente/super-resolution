# Super-resolution

Data fusions of high-resolution gridded data with low-resolutions gridded data with the aim of transferring structural features to from high to low. 

# Experiments

- High-resolution image/channel: Antarctica Ice Velocity (satellite-based surface data) [MEaSUREs InSAR-Based Antarctica Ice Velocity Map, Version 2 data](https://nsidc.org/data/nsidc-0484/versions/2)
    - check for 500m gridded

- Low-resolution image/channel: Antarctic bed elevation [MEaSUREs BedMachine Antarctica, Version 3 data](https://nsidc.org/data/nsidc-0756/versions/3)
    - contains surface, thickness, bed, firn
    - Controlled experiment: upsample and evaluate reconstruction

## Notes
- Use 2016 data since this is the latest common denominator
- Booth datasets are provided on [WGS 84 / Antarctic Polar Stereographic EPSG:3031](https://epsg.io/3031)
- create a new complimentary modality
- Limitation paper: subdivision
- images were artificially downsampled by a range of magnification factors to construct a controlled testing scenario.
- Average pooling is the same as magnifying
- Visualise lr input, hr input, output
- Visualise kernel components
- Compare to setconv/density channels
- Extend to disjoint grids
- role of geophysical data e.g. altitude

# References

[Reid, Alistair, Fabio Ramos, and Salah Sukkarieh. "Bayesian Fusion for Multi-Modal Aerial Images." Robotics: Science and Systems. 2013.](https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=89c7a83a8b9a99240c33b9d49d330755d071ae53)
