# `RealSim-CFIS`

A version of `RealSim` tailored to the Canada-France Imaging Survey (CFIS).

To access CFIS images, you must have a `CANFAR` account and the `vos` collection. First get a  `CANFAR` account and then install the Canadian Astronomy Data Center (CADC) `vos` collection of packages/tools. I installed it in a conda environment and dumped it to `cfis.yml` if you want to use the same environment that I used (recommended, so that I can debug for you). Otherwise, `vos` can be installed via `pip`:

    pip install -U vos
    
or by following any of the instructions here: https://pypi.org/project/vos/. With `vos` you will have to use the `getCert` tool every two weeks to renew your certificate. You just need your `CANFAR` username and password.

### Compatibility
Tested with Python 3 only. Tested with `Python 3.6.3`. `vos` is currently not compatible with `Python 3.7.x`.


