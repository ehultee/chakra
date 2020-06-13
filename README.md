# chakra

This repository should hold code snippets and test cases to explore the use of 
plastic-style iceberg calving in the Open Global Glacier Model.

What does the acronym mean?  Stay tuned...

[![badge](https://img.shields.io/badge/launch-On%20MyBinder-579aca.svg?logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAYAAABzenr0AAAABmJLR0QA/wD/AP+gvaeTAAAACXBIWXMAAC4jAAAuIwF4pT92AAAAB3RJTUUH4wEeDC0U4ki9ZgAAABl0RVh0Q29tbWVudABDcmVhdGVkIHdpdGggR0lNUFeBDhcAAAQFSURBVFjD7ZddaBxVGIafc2Y22+1u0qSxyaZJbBMTxLapUseUIsWWWrRIFhYL9bfgjdirQgX1qhUUb7zwQnqhghZ7IYo4MKJoo1YUtcQxlTRpWkyWGEp+l+zmZ5vs7swcL9yENeruxhi9yXc358z5nvfM9853ZgT/X/gAyXqsx3r8W2FHI9jRyNpDll1X2dFIVbH7CoVYCdgwLexoZA/wDHA34M/lSAE9wNuGaX2/fM2qBCyDnwFOCKU2S9fxCaUAUEKgpHQ8qc0BJvC8YVqTpYjQS3kCOfgrwEnheRvnqrYwuHMv09V1eEISnE2wdahfDw9fr9SczFNKyDvsaOSYYVrDxcohiu0+Bz8CnBeeVz1y2y6utD+AEmIpgRDgIqkeH2aH/QWhZBwl5SfAE4ZpJRfzrMYDF4XnHUjUNtJ16BgqN66UIlxVjtFSzzd9Q0ylXcKjMXb+2MnG2SRKiCeB9wzTcv8utywBbgBNUnn07zmIAny6JBTwo3DZFMpyZ3MYo7Uezc0y1thKvK4JT2oAJ4CyQvlLOQ7bhPJCc5XVpCo2A3BPawOHdjdzIxHj3LdnSaQSNIcrqa0MQNYhXredrD8ASrUX81kpAiqFUr754CZQCp8uOdDWREt9JRv8ccaTE3THekllphhJ/oLrZkgHQniatmhyXyEjliIgpRBuWXoeIXXiswN0DXTj18t45N7DzMyn+CnWw+R0nK7YV8RTN/C5DsLzFterQslLeQ17lBSp8mS8SkyPcmHwQ77sf58zR0/RUL0Vv17G11d/YDw5yeDENbRAI/uSG9CzGRBiCEivqg8YpnXJjkauStdtaLjcyUJgBkcpXvroNVrDzWRdh9j4rwyMDZERkBnto2LMh8/J4sEHgPOPPbBUOyFedGGiZXyC44kgITTG5mf4bqAbD3AFpHVBrSrjyMgs4dk5FGoGOA9kCjG0EpqQnPn5cnzLQx1pJx4/uH1earendSrQUJpECMEtjmTfTT8PJwPctVCGD4GCU8Dnhml5djTCm9eul16CZa49Wb6r7bi/pubizWDwsDY9/fKOBX3/toxGSvrJCtAUBD1ByBMACwpeAM4ZpuUUOwtEkd13AK8D24TuQ7nOgyhlA3sFPAbsB27NOT2m4GPgXeDKIrxQG/5LAXnw3cBbQHtuqhe4zzCtKTsakbnvej3PRx6QBbKGaalS4H8SkAevAc4CR3NTidxu+4olXMm3wJKAfKV2NBIATvN7HRcbSQfwqWFaqpRdrSTkMuUSeDwPDvAscGEt4AAy77EL4H7g1bz5N4B3DNPKrgX8Dx6wo5Fm4DOgNTfUCTxtmNbQWsGXSpAzTSivL/QCp9caDqDnAXrsaORR4DnAMkzr0n/xP/Ebgv/Oed8KI9UAAAAASUVORK5CYII=)](https://mybinder.org/v2/gh/OGGM/binder/master?urlpath=git-pull?repo=https://github.com/ehultee/chakra%26amp%3Bbranch=master%26amp%3Burlpath=lab/tree/chakra/sandbox.ipynb%3Fautodecode)

## Installation instructions

### Idealized glaciers set-up

chakra should work in the stress-free 
"[minimal environment](https://docs.oggm.org/en/latest/installing-oggm.html#install-a-minimal-oggm-environment)" 
required by OGGM's numerical core. The required packages are:

- numpy 
- scipy 
- pandas 
- matplotlib 
- shapely
- requests 
- configobj 
- netcdf4 
- xarray
- pytest
- oggm

You can install them manually or let ``pip`` do it for you with:

    pip install git+https://github.com/OGGM/oggm
    
Once in this environment, you can test OGGM alone with:

    pytest --pyargs oggm
    
And, in the chakra folder, you can test chakra with:

    pytest .
    

### Real glaciers set-up
    
For the "full OGGM experience" you'll need a few more packages. For a stress
free experience we recommend to use a dedicated conda environment (so that 
you can't break anything in the environments that work for you).

The easiest way to install all necessary packages with conda is to
copy the content of [this environment file](https://docs.oggm.org/en/latest/installing-oggm.html#installation-troubleshooting) 
into a text file, rename the environment to what suits you (in the linked 
file the environment is simply called "oggm_env"), and run:

    conda env create -f environment.yml

This should be completely independant of all other conda environments you 
may have ([conda documentation](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file) 
for reference). 

Type `source activate MY-ENV-NAME` to enter the environment and run the tests
as described in the section above.

Have fun!
