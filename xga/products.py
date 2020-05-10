#  This code is a part of XMM: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (david.turner@sussex.ac.uk) 10/05/2020, 15:17. Copyright (c) David J Turner

import os

# TODO Use these classes in the initial load in of supplied files in the source objects


class BaseProduct:
    def __init__(self, path, stdout_str, stderr_str):
        # Hopefully uses the path setter method
        self.path = path
        # Not sure if looking at stdout will be necessary, may delete
        self._stdout = self.parse_stdout(stdout_str)
        # This one definitely will, and likely each class will have to override the base method.
        self._stderr = self.parse_stderr(stderr_str)
        self.usable = False

    @property
    def path(self):
        return self._path

    @path.setter
    def path(self, prod_path: str):
        if not os.path.exists(prod_path):
            prod_path = None
        else:
            self.usable = True
        self._path = prod_path

    def parse_stdout(self, out_str):
        raise NotImplementedError("I'm getting round to it")

    def parse_stderr(self, err_str):
        raise NotImplementedError("I'm getting round to it")

    @property
    def stdout(self):
        # Hopefully because I'm not defining a setter method, this will be read only
        return self._stdout

    @property
    def stderr(self):
        return self._stderr


class Image(BaseProduct):
    def __init__(self, path, stdout_str, stderr_str):
        super().__init__(path, stdout_str, stderr_str)


class ExpMap(BaseProduct):
    def __init__(self, path, stdout_str, stderr_str):
        super().__init__(path, stdout_str, stderr_str)


class Spec(BaseProduct):
    def __init__(self, path, stdout_str, stderr_str):
        super().__init__(path, stdout_str, stderr_str)


class AnnSpec(BaseProduct):
    def __init__(self, path, stdout_str, stderr_str):
        super().__init__(path, stdout_str, stderr_str)






