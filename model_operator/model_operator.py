from typing import List, Callable
from .attribute import Attribute, Scaler
import pandas as pd
import tensorflow as tf
import numpy as np
# from sklearn.linear_model import LinearRegression

class AttributeConflictError(Exception):
    def __init__(self, attr_a, attr_b):
        super().__init__(f"Attributes '{attr_a}' and '{attr_b}' cannot be specified simultaneously.")

def isnone(val):
    return val is None

def rrmse(df, *, target, prediction):
    return np.sqrt(np.power(df[target] - df[prediction], 2).mean()) / df[target].mean()

class BaseModelOperator():
    def __init__(
            self, *, # keyword-only.
            # meta data
            model_version: str,
            description: str,
            # inputs/outputs
            input_attributes: List[Attribute],
            output_attribute: Attribute,
            # dataset
            # load_dataset_fn: Callable,
            use_validation_set: bool,
            random_validation_set: bool=None,
            validation_size: float=None,
            # model
            model_architecture_fn: Callable,
            # model hyperparameters
            constant_learning_rate: float=None,
            batch_size: int,
            loss_fn,
            metrics: list,
            n_epochs: int,
            model_architecture_params: dict,
            cosine_decay_params: dict=None
    ):
        # meta
        self.model_version = model_version # ex. 'v0p01'
        self.description = description
        # inputs/outputs
        self.input_attributes = input_attributes
        self.output_attribute = output_attribute
        # dataset
        # self.load_dataset_fn = load_dataset_fn
        self.use_validation_set = use_validation_set
        if use_validation_set:
            # use_validation_set: True/False to indicate whether or not to set aside validation set
            # If True, requires validation_size
            # random_validation_set: True/False
            if (random_validation_set is None):
                raise KeyError(f"Requires 'random_validation_set' if use_validation_set==True.")
            if validation_size is None:
                raise KeyError(f"Requires 'validation_size' if use_validation_set==True.")
            self.random_validation_set = random_validation_set
            self.validation_size = validation_size
        # model
        self.model_architecture_fn = model_architecture_fn
        if isnone(model_architecture_params):
            self.model_architecture_params = dict()
        else:
            self.model_architecture_params = model_architecture_params
        
        # model hypterparameters
        self.constant_learning_rate = constant_learning_rate
        self.batch_size = batch_size
        self.loss_fn = loss_fn
        self.metrics = metrics
        self.n_epochs = n_epochs

        self.cosine_decay_params = cosine_decay_params
        self.scaler = Scaler(attributes=input_attributes + [output_attribute])

        if not isnone(cosine_decay_params) and not isnone(constant_learning_rate):
            raise AttributeConflictError('cosine_decay_params', 'constant_learning_rate')
    
    def get_attribute(self, target_attr):
        """Given string name for attribute, returns the Attribute() object.
        Searches both self.input_attributes and self.output_attribute.
        """
        ret = []
        for attr in self.input_attributes + [self.output_attribute]:
            if attr.name == target_attr:
                ret.append(attr)
        assert len(ret) == 1
        return ret[0]

    def split_dataframe(self, df:pd.DataFrame, fold):
        df_train, df_test = df[df['fold'] != fold].copy(deep=True), df[df['fold'] == fold].copy(deep=True)

        if self.use_validation_set:
            threshold = int(df_train.shape[0] * (1. - self.validation_size))
            if self.random_validation_set:
                df_train = df_train.sample(frac=1)
            df_train, df_valid = df_train[:threshold], df_train[threshold:]
            return df_train, df_valid, df_test
        else:
            return df_train, df_test
        

    
    def evaluate_on_all_folds(self, df, folds):
        df_valids, df_tests, histories = [], [], []
        for fold in folds:
            if self.use_validation_set:
                history, df_valid, df_test = self.evaluate_on_single_fold(df, fold)
                histories.append(history)
                df_valids.append(df_valid)
                df_tests.append(df_test)
            else:
                history, df_test = self.evaluate_on_single_fold(df, fold)
                histories.append(history)
                df_tests.append(df_test)
        if self.use_validation_set:
            return histories, pd.concat(df_valids), pd.concat(df_tests)
        else:
            return histories, pd.concat(df_tests)

    def evaluate_on_single_fold(self, df, fold):
        if self.use_validation_set:
            df_train, df_valid, df_test = self.split_dataframe(df, fold=fold)
            df_train = self.fit_transform(df_train)
            df_valid = self.transform(df_valid)
            df_test = self.transform(df_test)
            x_train, y_train, x_valid, y_valid, x_test, y_test = self.split_x_y(df_train, df_valid, df_test)
        else:
            df_train, df_test = self.split_dataframe(df, fold=fold)
            df_train = self.fit_transform(df_train)
            df_test = self.transform(df_test)
            x_train, y_train, x_test, y_test = self.split_x_y(df_train, df_test)
            x_valid, y_valid = None, None

        model, history = self._train(x_train=x_train, y_train=y_train, x_valid=x_valid, y_valid=y_valid)

        yhat_test = self._predict(model, x_test)
        for key, val in yhat_test.items():
            df_test[key] = val

        if self.use_validation_set:
            yhat_valid = self._predict(model, x_valid)
            for key, val in yhat_valid.items():
                df_valid[key] = val

            df_valid = self.inverse_transform(df_valid)
            df_test = self.inverse_transform(df_test)
            return history, df_valid, df_test
        
        df_test = self.inverse_transform(df_test)
        return history, df_test


    def train_on_folds(self, df, folds):
        """Given dataset df, train on all data where folds == df['fold'].
        Returns the trained model.
        """
        # split into df_train and df_valid
        df_filtered = df[df['fold'].isin(folds)].copy(deep=True)
        if self.use_validation_set:
            threshold = int(df_filtered.shape[0] * (1. - self.validation_size))
            if self.random_validation_set:
                df_filtered = df_filtered.sample(frac=1)
            df_train, df_valid = df_filtered[:threshold], df_filtered[threshold:]
            df_train = self.fit_transform(df_train)
            df_valid = self.transform(df_valid)
            x_train, y_train, x_valid, y_valid = self.split_x_y(df_train, df_valid)
        else:
            df_train = self.fit_transform(df_filtered)
            x_train, y_train = self.split_x_y(df_train)
            x_valid, y_valid = None, None

        model, history = self._train(x_train=x_train, y_train=y_train, x_valid=x_valid, y_valid=y_valid)
        return model, history


    def fit_transform(self, df):
        return self.scaler.fit_transform(df)
    
    def transform(self, df):
        return self.scaler.transform(df)
    
    def inverse_transform(self, df):
        return self.scaler.inverse_transform(df)
    
    def split_x_y(self, *arg):
        ret = []
        for df in arg:
            x = df[[attr.name for attr in self.input_attributes]].to_numpy()
            y = df[[self.output_attribute.name]].to_numpy()
            ret.extend([x, y])
        return (ret)

    def _train(self, x_train, y_train, x_valid=None, y_valid=None):
        """Routine function that sets up model and trains the model.
        """
        # set cosine decay
        callbacks = []
        if self.cosine_decay_params is not None:
            lr = self.cosine_decay_params['initial_learning_rate']
            lr_scheduler = tf.keras.optimizers.schedules.CosineDecay(**self.cosine_decay_params)
            callbacks.append(tf.keras.callbacks.LearningRateScheduler(lr_scheduler))
        else:
            lr = self.constant_learning_rate

        # compile
        model = self.model_architecture_fn(**self.model_architecture_params)
        model.compile(optimizer=tf.optimizers.Adam(learning_rate=lr), loss=self.loss_fn, metrics=self.metrics)

        history = model.fit(x=x_train, y=y_train,
                            batch_size=self.batch_size,
                            epochs=self.n_epochs, shuffle=True,
                            validation_data=(x_valid, y_valid) if self.use_validation_set else None,
                            callbacks=callbacks,
                            verbose=1
        )
        return model, history
    
    def _predict(self, model, x_test) -> List[dict]:
        """This function helps flexibility.
        E.G. If the output is tfp and we don't want convert_to_tensor_fn() to be in effect, we can overwrite this.
        Returns a dictionary of form {col_name: values} which will be added to df_[test, valid] in self.evaluate_on_single_fold
        """
        yhat = model.predict(x_test)
        if self.output_attribute.transform is not None:
            # inverse transform
            params = self.output_attribute.get_params()
            scale, offset = params['scale'], params['offset']
            yhat = yhat * scale + offset
        return dict(yhat=yhat)

    def save_model(self, model, filename):
        model.save(filename)

    def load_model(self, filename):
        model = self.model_architecture_fn(**self.model_architecture_params)
        model.build(input_shape=())
        model.load_weights(filename)
        return model


class LinearRegressionOperator():
    def __init__(self,
                 input_variables: List[str],
                 output_variable: str
        ):
        raise NotImplementedError('implement LinearRegression without sklearn')
        self.input_variables = input_variables
        self.output_variable = output_variable

    def evaluate_on_all_folds(self, df, folds):
        df_tests = []
        for fold in folds:
            df_test = self.evaluate_on_single_fold(df, fold)
            df_tests.append(df_test)
        return pd.concat(df_tests)

    def evaluate_on_single_fold(self, df, fold):
        df_train, df_test = self.split_dataframe(df, fold)
        x_train, y_train, x_test, y_test = self.split_x_y(df_train, df_test)
        model = LinearRegression()
        model.fit(x_train, y_train)
        # predict
        yhat = model.predict(x_test)

        df_test['yhat'] = yhat

        return df_test


    def split_dataframe(self, df:pd.DataFrame, fold):
        df_train, df_test = df[df['fold'] != fold].copy(deep=True), df[df['fold'] == fold].copy(deep=True)
        return df_train, df_test

    def split_x_y(self, *arg):
        ret = []
        for df in arg:
            x = df[[attr for attr in self.input_variables]].to_numpy()
            y = df[[self.output_variable]].to_numpy()
            ret.extend([x, y])
        return (ret)
