# XGA Test Suite

## Running the tests

Run prepare_data.py to generate the test data.

From the root project directory (i.e. one level up from tests/), run:

```
export XGA_CONFIG_DIR=tests/test_data/config/
python -m unittest discover -s tests -t . 
```

**Note: You can add `-v` to the end of the above command to see the test names as they are run.**

## Running specific tests

For example: 

```
export XGA_CONFIG_DIR=tests/test_data/config/
python -m unittest -v tests/test_products/test_events.py -k "TestEventListImageGeneration.check_missions_evt_init_image_gen" 
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

## Running specific tests with coverage

Note that this will produce misleading coverage results at the XGA module level, as only a subset of
tests are being run. It does help to see coverage of the parts of XGA that are being probed by the
specific test(s) being run, however.

```
export XGA_CONFIG_DIR=tests/test_data/config/
coverage run -m unittest -v tests/test_products/test_events.py -k "TestEventListImageGeneration.check_missions_evt_init_image_gen" 
```