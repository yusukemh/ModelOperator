from typing import List
import pandas as pd
import numpy as np

class Variable():
    def __init__(self, name, transform, scale=None, offset=None, _fit=False):
        """Defines Variable and how it should be transformed as preprocessing.
        """
        # transform cannot be None
        if not transform in ['normalize', 'percent', 'identity', 'custom']:
            raise ValueError(f"Expected 'transform' in ['normalize', 'percent', 'identity', 'custom']. Got {transform} instead.")
            
        self.name = name
        self.transform = transform
        self._fit = _fit

        if transform == 'percent':
            self.fit(values=None)
            if (scale != 100) or (offset != 50):
                print(f"[User Warning] Argument conflict; {transform=} but {scale=} and {offset=} specified.\n" + \
                          f"'scale' and/or 'offset' will be ignored for this Variable object {self.__repr__()}.")
        elif transform == 'identity':
            self.fit(values=None)
            if (scale != 1) or (offset != 0):
                print(f"[User Warning] Argument conflict; {transform=} but {scale=} and {offset=} specified.\n" + \
                          f"'scale' and/or 'offset' will be ignored for this Variable object {self.__repr__()}.")
        elif transform == 'normalize':
            if _fit:
                self.scale = scale
                self.offset = offset
            else:
                self.scale = None
                self.offset = None
                if (scale is not None) or (offset is not None):
                    # if _fit = False and scale and offset is given, warn the user.
                    print(f"[User Warning] Argument conflict; {transform=} but {_fit=}, {scale=}, and {offset=} specified.\n" + \
                            f"'scale' and/or 'offset' will be ignored for this Variable object {self.__repr__()}.")

        elif transform == 'custom':
            if (scale is None) or (offset is None):
                raise ValueError(f"If transform == 'custom', then 'scale' and 'offset' must be specified.")
            self.scale = scale
            self.offset = offset

        """
        # if transform in ['normalize', 'percent', 'identiry'] but scale and/or offset is/are given, they will be ignored.
        if transform != 'custom':
            if (scale is not None) or (offset is not None):
                # if transform is not custom but scale and offset is given.
                if transform == 'normalize' and _fit:
                    # In this case operation is allowed.
                    self.scale = scale
                    self.offset = offset
                else:
                    # If transform='normalize' but _fit == False, OR, transform is in ['percent', 'identity']
                    print(f"[User Warning] Argument conflict; {transform=} but {scale=} and {offset=} specified.\n" + \
                          f"'scale' and/or 'offset' will be ignored for this Variable object {self.__repr__()}.")
                    # ignore input
                    self.scale = None
                    self.offset = None
            else:
                self.scale = None
                self.offset = None
        else:
            self.scale = scale
            self.offset = offset
        self._fit = _fit
        """

    @staticmethod
    def from_dict(d):
        raise NotImplementedError("[WARNING] This function is not fully tested. Use of this function should be restricted.")
        ret = Variable(name=d['name'], transform='percent') # 'transform' will be overwritten anyways.

        for attr in ['transform', 'scale', 'offset', '_fit']:
            try:
                ret.__setattr__(attr, d[attr])
            except KeyError as e:
                raise KeyError(f"from_dict() requires key '{attr}' in its argument 'd'.")

        return ret

    def __repr__(self) -> str:
        return f"<class Variable({self.name})>"

    def as_dict(self):
        return {
            "name": self.name,
            "transform": self.transform,
            "scale": self.scale,
            "offset": self.offset,
            "_fit": self._fit
        }

    def fit(self, values):
        if self.transform == 'percent':
            self.scale = 100
            self.offset = 50
        elif self.transform == 'normalize':
            self.scale = np.std(values)
            self.offset = np.mean(values)
            if np.abs(self.scale) < 1e-4:
                print(f'[Warning] Std for {self.name} is too small; {self.scale:.05f}. This could cause precision error.')
        elif self.transform == 'identity':
            self.scale = 1.
            self.offset = 0.
        # elif self.transform is None:
        #     # values must have been set at instantiation.
        #     pass

        self._fit = True
    
    def get_params(self):
        if self._fit == False:
            raise ValueError(f'Accessing params before fit(). Abort.')
        return dict(scale=self.scale, offset=self.offset)

class Scaler():
    def __init__(self, variables: List[Variable]):
        self.variables = variables

        for attr in variables:
            if not isinstance(attr, Variable):
                raise ValueError(f"Expected elements of 'variables' to be of type Variable. Got {attr} instead at position {variables.index(attr)}.")
            
    def fit(self, df: pd.DataFrame):
        for attr in self.variables:
            attr.fit(df[attr.name].values)
    
    def get_params(self):
        ret = {}
        for attr in self.variables:
            ret[attr.name] = attr.get_params()
        return ret
    
    def transform(self, df: pd.DataFrame):
        df = df.copy(deep=True)
        for attr in self.variables:
            params = attr.get_params()
            scale, offset = params['scale'], params['offset']
            df[attr.name] = (df[attr.name] - offset) / scale
        return df
    
    def fit_transform(self, df):
        self.fit(df)
        return self.transform(df)
    
    def inverse_transform(self, df):
        df = df.copy(deep=True)
        for attr in self.variables:
            if attr.transform == 'none': continue
            params = attr.get_params()
            scale, offset = params['scale'], params['offset']
            df[attr.name] = df[attr.name] * scale + offset
        return df