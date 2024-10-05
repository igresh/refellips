import numpy as np
import os.path
from pathlib import Path
from numpy.testing import assert_allclose

from refellips.dataSE import DataSE, open_EP4file, open_FilmSenseFile

pth = Path(os.path.dirname(os.path.abspath(__file__)))


def test_multiple_areas():
    data = open_EP4file(pth / "post synthesis.dat")
    assert len(data) == 32
    assert "X pos" in data[0].metadata
    for d in data:
        np.testing.assert_allclose(len(d), 5)

    data = open_EP4file(pth / "19-1-1.dat")
    assert isinstance(data, DataSE)
    assert "Y pos" in data.metadata

    data = open_EP4file(pth / "15-1-1.dat")
    assert isinstance(data, DataSE)
    assert "Y pos" in data.metadata
    assert len(data) == 11
    np.testing.assert_allclose(data.psi[-1], 12.72666667)

def test_filmsense_loader():
    ## STANDARD DATA TEST
    std_fsdata = open_FilmSenseFile(pth / 'Filmsense_staticTest.txt')

    std_bench  = np.array([[367.20, 26.46, 158.74],
                           [449.64, 15.77, 173.19],
                           [525.63, 12.64, 174.05],
                           [593.58, 11.13, 174.54],
                           [656.32, 10.25, 174.92], 
                           [732.22,  9.51, 175.37],
                           [852.88,  8.76, 175.98],
                           [946.73,  8.35, 176.36]]).T

    assert std_fsdata.metadata['numwvls'] == 8
    assert_allclose(std_fsdata.aoi, std_fsdata.metadata['nomAOI'])
    assert_allclose(std_bench[0], std_fsdata.wavelength, rtol=1e-3)
    assert_allclose(std_bench[1], std_fsdata.psi, rtol=1e-3)
    assert_allclose(std_bench[2], std_fsdata.delta, rtol=1e-3)

    ## DYNAMIC DATA TEST
    dyn_fsdata = open_FilmSenseFile(pth / 'Filmsense_kineticTest.txt')

    bench_times = [4.95, 14.86, 24.77, 34.66, 44.57, 54.48, 64.38]

    times = list(dyn_fsdata.keys())

    assert_allclose(bench_times, times,   rtol=1e-3)

    sing_dyn_fsdata = dyn_fsdata[times[2]]

    dyn_bench  = np.array([[367.20, 27.04, 148.91],
                           [449.64, 16.09, 162.67],
                           [525.63, 12.91, 163.83],
                           [593.58, 11.36, 164.72],
                           [656.32, 10.45, 165.54], 
                           [732.22,  9.68, 166.52],
                           [852.88,  8.89, 167.94],
                           [946.73,  8.45, 168.86]]).T

    assert sing_dyn_fsdata.metadata['numwvls'] == 8
    assert_allclose(sing_dyn_fsdata.aoi, sing_dyn_fsdata.metadata['nomAOI'])
    assert_allclose(dyn_bench[0], sing_dyn_fsdata.wavelength, rtol=1e-3)
    assert_allclose(dyn_bench[1], sing_dyn_fsdata.psi, rtol=1e-3)
    assert_allclose(dyn_bench[2], sing_dyn_fsdata.delta, rtol=1e-3)