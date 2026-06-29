# XGA Test Suite

## Running the tests

Run prepare_data.py to generate the test data.

From the root project directory (i.e. one level up from tests/), run:

```
export XGA_CONFIG_DIR=tests/test_data/config/
python -m unittest discover -s tests -t . 
```

## Running with coverage

```
export XGA_CONFIG_DIR=tests/test_data/config/
coverage run -m unittest discover -s tests -t .
```

```
coverage report
```

```
coverage html
```