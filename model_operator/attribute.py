from typing import List
import pandas as pd
import numpy as np

class Attribute():
    def __init__(self, name, transform=None, scale=None, offset=None):
        self.name = name
        if transform is not None:
            self.transform = transform
            if not transform in ['normalize', 'percent', 'none']:
                raise ValueError(f"Expected 'transform' in ['normalize', 'percent', 'none']. Got {transform} instead.")
            if (scale is not None) or (offset is not None):
                print("[User Warning] Argument(s) 'scale' and/or 'offset' will be ignored.")
        else:
            self.transform = None
            self.scale = scale
            self.offset = offset
            if (scale is None) or (offset is None):
                raise ValueError(f"If transform=None, both 'scale' and 'offset' must be specified.")
        
        self._fit = False

    def __repr__(self) -> str:
        return f"<class Attribute({self.name})>"

    def fit(self, values):
        if self.transform == 'percent':
            self.scale = 100
            self.offset = 50
        elif self.transform == 'normalize':
            self.scale = np.std(values)
            self.offset = np.mean(values)
            if np.abs(self.scale) < 1e-4:
                print(f'[Warning] Std for {self.name} is too small; {self.scale:.05f}. This could cause precision error.')
        elif self.transform == 'none':
            self.scale = 1.
            self.offset = 0.
        elif self.transform is None:
            # values must have been set at instantiation.
            pass

        self._fit = True
    
    def get_params(self):
        if self._fit == False:
            raise ValueError(f'Accessing params before fit(). Abort.')
        return dict(scale=self.scale, offset=self.offset)

class Scaler():
    def __init__(self, attributes: List[Attribute]):
        self.attributes = attributes

        for attr in attributes:
            if not isinstance(attr, Attribute):
                raise ValueError(f"Expected elements of 'attributes' to be of type Attribute. Got {attr} instead at position {attributes.index(attr)}.")
            
    def fit(self, df: pd.DataFrame):
        for attr in self.attributes:
            attr.fit(df[attr.name].values)
    
    def get_params(self):
        ret = {}
        for attr in self.attributes:
            ret[attr.name] = attr.get_params()
        return ret
    
    def transform(self, df: pd.DataFrame):
        df = df.copy(deep=True)
        for attr in self.attributes:
            if attr.transform == 'none': continue
            params = attr.get_params()
            scale, offset = params['scale'], params['offset']
            df[attr.name] = (df[attr.name] - offset) / scale
        return df
    
    def fit_transform(self, df):
        self.fit(df)
        return self.transform(df)
    
    def inverse_transform(self, df):
        df = df.copy(deep=True)
        for attr in self.attributes:
            if attr.transform == 'none': continue
            params = attr.get_params()
            scale, offset = params['scale'], params['offset']
            df[attr.name] = df[attr.name] * scale + offset
        return df