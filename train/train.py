import argparse
import os
import numpy as np
import joblib
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from azureml.core.run import Run
from azureml.core import Dataset, Workspace

def clean(ds):

    df = ds.to_pandas_dataframe()

    df = df.join(pd.get_dummies(df.type, prefix="type")).drop(columns=['type'])
    df = df.join(pd.get_dummies(df.homeTeam, prefix="home")).drop(columns=['homeTeam'])
    df = df.join(pd.get_dummies(df.awayTeam, prefix="away")).drop(columns=['awayTeam'])

    return df

def train(ds, fit, normalize, split):

    # Extract dummies for training
    df = clean(ds)

    # Separate out the label column
    x = df.drop(columns=['penaltyMinutes'])
    y = df['penaltyMinutes']

    # Split data into train and test sets.
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=split)

    model = LinearRegression(fit_intercept=bool(fit), normalize=bool(normalize), copy_X=False, n_jobs=-1).fit(x_train, y_train)

    rsquared = model.score(x_test, y_test)

    return model, rsquared

def main():
    # Add arguments to script
    parser = argparse.ArgumentParser()

    parser.add_argument('--normalize', type=int, default=0)
    parser.add_argument('--fit', type=int, default=0)
    parser.add_argument('--split', type=float, default=0.3)

    args = parser.parse_args()

    run = Run.get_context()

    run.log("Normalize:", np.bool(args.normalize))
    run.log("Fit:", np.bool(args.fit))
    run.log("Split %:", np.int(args.split))

    # Grab the registered dataset from the workspace. This still started as a public / external dataset. See dataprep.ipynb for more details
    ws = run.experiment.workspace
    ds = Dataset.get_by_name(workspace=ws, name='NHL-Penalties-2020')

    model, rsquared = train(ds, args.fit, args.normalize, args.split)    

    run.log("R Squared", np.float(rsquared))

    # Register the Model
    os.makedirs("outputs", exist_ok=True)
    filename = "outputs/model.pkl"
    joblib.dump(value=model, filename=filename)
    run.upload_file(name='ModelFile', path_or_stream=filename)

    # Complete the dang run!
    run.complete()

if __name__ == '__main__':
    main()