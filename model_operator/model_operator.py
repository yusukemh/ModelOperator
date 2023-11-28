from typing import List, Callable
from .attribute import Attribute, Scaler
import pandas as pd
import tensorflow as tf

class AttributeConflictError(Exception):
    def __init__(self, attr_a, attr_b):
        super().__init__(f"Attributes '{attr_a}' and '{attr_b}' cannot be specified simultaneously.")

def isnone(val):
    return val is None

class BaseModelOperator():
    def __init__(
            self, *, # keyword-only.
            # meta data
            model_version: str,
            description: str,
            # inputs/outputs
            input_attributes: List[Attribute],
            output_attributes: List[Attribute],
            # dataset
            # load_dataset_fn: Callable,
            use_validation_set: bool,
            random_validation_set: bool=None,
            validation_size: float=None,
            # model
            model_architecture_fn: Callable,
            # model hyperparameters
            constant_learning_rate: float,
            batch_size: int,
            loss_fn,
            metrics: list,
            n_epochs: int,
            extra_architecture_parameters: dict,
            cosine_decay_params: dict
    ):
        # meta
        self.model_version = model_version # ex. 'v0p01'
        self.description = description
        # inputs/outputs
        self.input_attributes = input_attributes
        self.output_attributes = output_attributes
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
        self.extra_architecture_parameters = extra_architecture_parameters
        # model hypterparameters
        self.constant_learning_rate = constant_learning_rate
        self.batch_size = batch_size
        self.loss_fn = loss_fn
        self.metrics = metrics
        self.n_epochs = n_epochs

        self.cosine_decay_params = cosine_decay_params
        self.scaler = Scaler(attributes=input_attributes + output_attributes)

        if not isnone(cosine_decay_params) and isnone(constant_learning_rate):
            raise AttributeConflictError(cosine_decay_params, constant_learning_rate)

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

        # set cosine decay
        callbacks = []
        if self.cosine_decay_params is not None:
            lr = self.cosine_decay_params['initial_learning_rate']
            lr_scheduler = tf.keras.optimizers.schedules.CosineDecay(**self.cosine_decay_params)
            callbacks.append(tf.keras.callbacks.LearningRateScheduler(lr_scheduler))
        else:
            lr = self.constant_learning_rate

        # compile
        model = self.model_architecture_fn(n_inputs=x_train.shape[1], **self.extra_architecture_parameters)
        model.compile(optimizer=tf.optimizers.Adam(learning_rate=lr), loss=self.loss_fn, metrics=self.metrics)

        history = model.fit(x=x_train, y=y_train,
                            batch_size=self.batch_size,
                            epochs=self.n_epochs, shuffle=True,
                            validation_data=(x_valid, y_valid) if self.use_validation_set else None,
                            callbacks=callbacks,
                            verbose=1
        )

        yhat_test = self._predict(model, x_test)
        for key, val in yhat_test.items():
            df_test[key] = val

        if self.use_validation_set:
            yhat_valid = self._predict(model, x_valid)
            for key, val in yhat_valid.items():
                df_valid[key] = val
            return history, df_valid, df_test
        
        return history, df_test


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
            y = df[[attr.name for attr in self.output_attributes]].to_numpy()
            ret.extend([x, y])
        return (ret)
    
    def _predict(self, model, x_test) -> List[dict]:
        """This function helps flexibility.
        E.G. If the output is tfp and we don't want convert_to_tensor_fn() to be in effect, we can overwrite this.
        Returns a dictionary of form {col_name: values} which will be added to df_[test, valid] in self.evaluate_on_single_fold
        """
        return dict(yhat=model.predict(x_test))
