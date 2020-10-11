Install dependencies:
```
conda env create -f environment.yml
conda activate jjs229
```

If you need to install additional or change existing packages, then change 'environment.yml' and run the following within your activated environment:

```
conda env update --file environment.yml --prune
```

Run unit tests:
```
pytest
```

Run unit tests with stdout:
```
pytest -s
```
