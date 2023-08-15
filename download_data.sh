# Download files to local server (Roger)

# Download nc file to a new folder /home/kim/data/zenodo_7760491_RACMO2-3p2 outside of repo to keep data separate
# -P parent directory

# Use prompt for password so it is save to upload this to Github
# https://nsidc.org/data/user-resources/help-center/programmatic-data-access-guide
wget --http-user=kimbente --ask-password -np antarctica_ice_velocity_450m_v2.nc -P /home/kim/data/nsidc https://n5eil01u.ecs.nsidc.org/MEASURES/NSIDC-0484.002/1996.01.01/antarctica_ice_velocity_450m_v2.nc

wget --http-user=kimbente --ask-password -np BedMachineAntarctica-v3.nc -P /home/kim/data/nsidc https://n5eil01u.ecs.nsidc.org/MEASURES/NSIDC-0756.003/1970.01.01/BedMachineAntarctica-v3.nc