from src.forecasting.utils.libraries_data_handling import pd, np

def log_transform(
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

    # Iterate through each columns to transform
    for col in cols:
        df[col] = np.log1p(df[col])
    return df

def log_transform_inverse(
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

    # Iterate through each columns to inverse transform
    for col in cols:
        df[col] = np.log1p(df[col])
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

def sqrt_transform(
    df      : pd.DataFrame,
) -> pd.DataFrame:
    """
    Function to pre-normalization using Square Root Transform.

    Args:
        df (pd.DataFrame) : DataFrame input.

    Returns:
        pd.DataFrame : This function returns dataframe format. 
    
    """

    # Initialize epsilon value
    epsilon = 1e-3
    
    # Iterate through each columns  
    for col in df.columns:

        # Pre-normalization with Square Root Transform
        df[col] = np.sqrt(df[col] + epsilon)

    return df

def sqrt_transform_inverse(
    df      : pd.DataFrame,
) -> pd.DataFrame:
    """
    Function to inverse pre-normalization Square Root Transform.

    Args:
        df (pd.DataFrame) : DataFrame input.

    Returns:
        pd.DataFrame : This function returns dataframe format. 
    
    """

    # Initialize epsilon value
    epsilon = 1e-3
    
    # Iterate through each columns  
    for col in df.columns:

        # Inverse pre-normalization with Square Root Transform
        df[col] = (df[col] ** 2) - epsilon

    return df