from src.forecasting.utils.libraries_data_handling import pd, np
from src.forecasting.utils.libraries_modelling import MinMaxScaler

def sqrt_transform(
    df   : pd.DataFrame,
    cols : np.array
) -> pd.DataFrame:
    """
    Function to pre-normalization using Square Root Transform.

    Args:
        df (pd.DataFrame) : DataFrame input.
        cols (np.array)   : Columns to transform.

    Returns:
        pd.DataFrame : This function returns dataframe format. 
    
    """
    
    # Iterate through selected columns to transform  
    for col in cols:

        # Pre-normalization with Square Root Transform
        df[col] = np.sqrt(df[col])
    return df

def sqrt_transform_inverse(
    df   : pd.DataFrame,
    cols : np.array,
) -> pd.DataFrame:
    """
    Function to inverse pre-normalization Square Root Transform.

    Args:
        df (pd.DataFrame) : DataFrame input.
        cols (np.array)   : Columns to transform.

    Returns:
        pd.DataFrame : This function returns dataframe format. 
    
    """
    
    # Iterate through selected columns to transform
    for col in cols:

        # Inverse pre-normalization with Square Root Transform
        df[col] = (df[col] ** 2)
    return df

def log1p_transform(
    df   : pd.DataFrame,
    cols : np.array
) -> pd.DataFrame:
    """
    Function to pre-normalization using Log Transform.

    Args:
        df (pd.DataFrame): DataFrame input.
        cols (np.array)  : Columns to transform.
    
    Returns:
        pd.DataFrame: This function returns dataframe format.
    """

    # Iterate through selected columns to transform
    for col in cols:
        df[col] = np.log1p(df[col])
    return df

def log1p_transform_inverse(
    df   : pd.DataFrame,
    cols : np.array
) -> pd.DataFrame:
    """
    Function to inverse pre-normalization of Log Transform.

    Args:
        df (pd.DataFrame): DataFrame input.
        cols (np.array)  : Columns to inverse transform.

    Returns:
        pd.DataFrame: This function returns dataframe format.
    """

    # Iterate through selected columns to inverse transform
    for col in cols:
        df[col] = np.log1p(df[col])
    return df

def minmax_transform(
    df   : pd.DataFrame,
    cols : np.array,
) -> pd.DataFrame:
    """
    Function to pre-normalization using MinMaxScaler.

    Args:
        df (pd.DataFrame) : DataFrame input.
        cols (np.array)   : Columns to transform.

    Returns:
        pd.DataFrame : This function returns dataframe format.
    """

    # Iterate through selected columns to transform
    for col in cols:
        scaler = MinMaxScaler()
        df[[col]] = scaler.fit_transform(df[[col]])
    return df

def sincos_transform(
    df     : pd.DataFrame,
    cols   : np.array,
    period : int
) -> pd.DataFrame:
    """
    Function to pre-normalization using Sin-Cos encoding.

    Args: 
        df (pd.DataFrame) : DataFrame input.
        cols (np.array)   : Columns to transform.

    Returns:
        pd.DataFrame : This function returns dataframe input.
    """
    
    # Iterate through selected columns to transform
    for col in cols:

        # Initialize radiant
        radians = 2 * np.pi * df[col] / period
        df[col + "_sin"] = np.sin(radians)
        df[col + "_cos"] = np.cos(radians)
    return df

def onehot_transform(
    df        : pd.DataFrame,
    cols      : np.array,
    threshold : float
) -> pd.DataFrame:
    """
    Function to pre-normalization using One Hot Encoding.

    Args:
        df (pd.DataFrame) : DataFrame input.
        cols (np.array)   : Columns to transform.
        threshold (float) : Cutoff to define zero vs non zero.

    Returns:
        pd.DataFrame : This function returns dataframe input.
    """

    # Iterate through selected columns to transform
    for col in cols:
        df[col + '_zero'] = (df[col] <= threshold).astype(int)
        df[col + '_nonzero'] = (df[col] > threshold).astype(int)

    return df

def divide_transform(
    df : pd.DataFrame
) -> pd.DataFrame:
    """
    Function to pre-normalization using Divided Transform.

    Args:
        df (pd.DataFrame) : DataFrame input.

    Returns:
        pd.DataFrame : This function returns dataframe format. 
    """

    # Iterate through each columns
    for col in df.columns:
        if col in ['y3', 'x5']:
            df[col] = df[col] / 10
        if col == 'x8':
            df[col] = df[col] / 7
    
    return df

def divide_transform_inverse(
    df : pd.DataFrame
) -> pd.DataFrame:
    """
    Function to inverse pre-normalization using Divided Transform.

    Args:
        df (pd.DataFrame) : DataFrame input.

    Returns:
        pd.DataFrame : This function returns dataframe format. 
    """

    # Iterate through each columns
    for col in df.columns:
        if col in ['y3', 'x5']:
            df[col] = df[col] * 10
        if col == 'x8':
            df[col] = df[col] * 7
    return df


def plus_transform(
    df : pd.DataFrame,
) -> pd.DataFrame:
    """
    Function to pre-normalization using Plus Transform.

    Args:
        df (pd.DataFrame) : DataFrame input.

    Returns:
        pd.DataFrame : This function returns dataframe format. 
    """

    # Iterate through each columns
    for col in df.columns:

        # Pre-normalization with Square Root Transform
        df[col] = df[col] + 20

    return df

def plus_transform_inverse(
    df : pd.DataFrame,
) -> pd.DataFrame:
    """
    Function to inverse pre-normalization using Plus Transform.

    Args:
        df (pd.DataFrame) : DataFrame input.

    Returns:
        pd.DataFrame : This function returns dataframe format. 
    """

    # Iterate through each columns
    for col in df.columns:

        # Pre-normalization with Square Root Transform
        df[col] = df[col] - 20
        
    return df