import pytest
import weekly_gauss_fit as wgf
import numpy as np






@pytest.mark.parametrize("data,a,b",
[(np.linspace(0,1,101),-100,100),
(np.linspace(-10,10,101),0,1),
(np.linspace(-1,1,1001),0,1),
(np.linspace(5,50,101),2,5)
])
def test_normalization(data,a,b):
    normed= wgf.norm(data,a,b)
    assert (normed[0] == a) and (normed[-1] == b)


@pytest.mark.parametrize("data",
[ ( wgf.standardize(wgf.Gauss(
    np.linspace(-10,10,1001),2,0.5,0)) ),
(   
    wgf.standardize(wgf.Gauss(
    np.linspace(-10,10,1001),2,0.1,0)) 
    ),
(   
    wgf.standardize(wgf.Gauss(
    np.linspace(-10,10,1001),20,0.5,0)) 
    ),
(   
    wgf.standardize(wgf.Gauss(
    np.linspace(-10,10,1001),2,1,0)) 
    )
    ])
def test_standardization(data):
    assert (round(np.mean(data),10) == 0) and (round(np.std(data),10) == 1)





@pytest.mark.parametrize("norm,xmin,xmax",
[(np.linspace(0,1,101),-100,100),
(np.linspace(-1,1,101),0,100),
(np.linspace(-1,1,1001),0,1),
(np.linspace(-10,10,101),0,10)
])
def test_denormalization(norm,xmin,xmax):
    denormed= wgf.deNorm(norm,xmin,xmax)
    assert (denormed[0] == xmin) and (denormed[-1] == xmax)

