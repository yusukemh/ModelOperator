from typing import List
import pandas as pd
import numpy as np

AVAILABLE_OPTIONS = ['normalize', 'percent', 'identity', 'minmax', 'custom']

class Variable():
    def __init__(self, name, transform, scale=None, offset=None, _fit=False):
        """Defines Variable and how it should be transformed as preprocessing.
        """
        # transform cannot be None
        if not transform in AVAILABLE_OPTIONS:
            raise ValueError(f"Expected 'transform' in {AVAILABLE_OPTIONS}. Got {transform} instead.")
            
        self.name = name
        self.transform = transform
        self._fit = _fit

        if transform == 'percent':
            self.fit(values=None)
            if scale is None and offset is None:
                pass
            elif (scale != 100) or (offset != 50):
                print(f"[User Warning] Argument conflict; {transform=} but {scale=} and {offset=} specified.\n" + \
                          f"'scale' and/or 'offset' will be ignored for this Variable object {self.__repr__()}.")
        elif transform == 'identity':
            self.fit(values=None)
            if scale is None and offset is None:
                pass
            elif (scale != 1) or (offset != 0):
                print(f"[User Warning] Argument conflict; {transform=} but {scale=} and {offset=} specified.\n" + \
                          f"'scale' and/or 'offset' will be ignored for this Variable object {self.__repr__()}.")
        elif transform in ['normalize', 'minmax']:
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
        elif self.transform == 'identity':
            self.scale = 1.
            self.offset = 0.
        elif self.transform == 'minmax':
            self.scale = np.max(values) - np.min(values)
            self.offset = np.min(values)
        elif self.transform == 'custom':
            pass
        else:
            raise NotImplementedError()

        if np.abs(self.scale) < 1e-4:
                raise ValueError(f'[Warning] Std for {self.name} is too small; {self.scale:.05f}. This could cause precision error.')

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