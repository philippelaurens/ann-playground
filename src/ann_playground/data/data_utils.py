from pathlib import Path
import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo 


def fetch_and_save_dataset(
    id: int,
    output_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series | pd.DataFrame]:

    # Fetch dataset
    dataset = fetch_ucirepo(id=id) 
    
    X = dataset.data.features 
    y = dataset.data.targets 

    df = pd.concat([X, y], axis=1)
    # print(df.metadata) 
    # print(df.variables) 

    df.to_csv(output_path, index=False)
    return df, X, y