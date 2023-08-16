# Super-resolution

Data fusions of high-resolution gridded data with low-resolutions gridded data with the aim of transferring structural features to from high to low. 

# Experiments

- High-resolution image/channel: Antarctica Ice Velocity (satellite-based surface data) [MEaSUREs Annual Antarctic Ice Velocity Maps, Version 1](https://nsidc.org/data/nsidc-0720/versions/1#anchor-1)
  - 1000m resolution
  - Time: 2016-07-01 - 2017-06-30 - has a lot of missing data
  - Time: 2008-07-01 - 2009-06-30 - has less but also has missing data
  - [Documentation](https://nsidc.org/sites/default/files/nsidc-0720-v001-userguide_0.pdf)
  - bidirectional data

- Low-resolution image/channel: Antarctic bed elevation [MEaSUREs BedMachine Antarctica, Version 3 data](https://nsidc.org/data/nsidc-0756/versions/3)
    - Original 500m resolution
    - contains surface elevation, ice thickness, bed elevation (topography), firn air content
    - Surface elevations data is from REMA (2015), ice thickness was modelled and bed elevation (topography) was inferred. Firn is a correction value that was applied to ice thickness.
    - Errors are available
    - [Documentation](https://nsidc.org/sites/default/files/nsidc-0756-v002-userguide_1.pdf)
    - Controlled experiment: upsample and evaluate reconstruction

Alternative:
- High-resolution image/channel: Antarctica Ice Velocity (satellite-based surface data) [MEaSUREs InSAR-Based Antarctica Ice Velocity Map, Version 2 data](https://nsidc.org/data/nsidc-0484/versions/2)
    - On 450m grid (perfrom 10x magnification)
    - comes only in 6 GB data file
    - [Documentation](https://nsidc.org/sites/default/files/nsidc-0484-v002-userguide.pdf)

## Notes
- Use 2016 data since this is the latest common denominator
  - Change to 2015 potentially
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
- Use surface elevation to improve resolution of ice flow: measured in 2 directions
  - Physics informed kernel?

# References

[Reid, Alistair, Fabio Ramos, and Salah Sukkarieh. "Bayesian Fusion for Multi-Modal Aerial Images." Robotics: Science and Systems. 2013.](https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=89c7a83a8b9a99240c33b9d49d330755d071ae53)

# Data references

Morlighem, M. 2020. MEaSUREs BedMachine Antarctica, Version 2.[2016 data used]. Boulder,
Colorado USA. NASA National Snow and Ice Data Center Distributed Active Archive Center.
https://doi.org/10.5067/E1QL9HFQ7A8M. [accessed 16 August 2023]. 

Morlighem, M., E. Rignot, T. Binder, D. D. Blankenship, R. Drews, G. Eagles, O., et al. 2020. Deep
glacial troughs and stabilizing ridges unveiled beneath the margins of the Antarctic ice sheet, Nature
Geoscience. 13. 132-137. https://doi.org/10.1038/s41561-019-0510-8 

Rignot, E., J. Mouginot, and B. Scheuchl. 2017. MEaSUREs InSAR-Based Antarctica Ice Velocity
Map, Version 2. [Indicate subset used]. Boulder, Colorado USA. NASA National Snow and Ice Data
Center Distributed Active Archive Center. https://doi.org/10.5067/D7GK8F5J8M8R. [accessed 16 August 2023].

not used currently:  
*Mouginot, J., B. Scheuchl, and E. Rignot. 2017, updated 2017. MEaSUREs Annual Antarctic Ice
Velocity Maps 2005-2017, Version 1. [2016-2017 data used]. Boulder, Colorado USA. NASA National
Snow and Ice Data Center Distributed Active Archive Center
https://doi.org/10.5067/9T4EPQXTJYW9. [accessed 16 August 2023].*


