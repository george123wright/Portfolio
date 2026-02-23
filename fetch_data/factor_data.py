import pandas_datareader.data as web
import pandas as pd
import config


start_date = "2000-01-01"

end_date = config.TODAY


def load_factor_data():
    """
        Load Fama-French 5 factors data from the Fama-French database.
        
        Returns:
            DataFrame: Fama-French 5 factors data with columns renamed and formatted.
        
    """
    
    ff5_monthly = (
        web.DataReader("F-F_Research_Data_5_Factors_2x3", "famafrench",
                       start = start_date, end = end_date)[0]
          .rename(columns = {
              "Mkt-RF": "mkt_excess",
              "SMB": "smb",
              "HML": "hml",
              "RMW": "rmw",
              "CMA": "cma",
              "RF": "rf"
          })
          .divide(100)
          .rename_axis("date")
          .reset_index()
          .assign(date=lambda df: pd.to_datetime(df["date"].astype(str)))
          .set_index("date")
    )
    
    ff5 = (1 + ff5_monthly).resample('QE').prod().subtract(1)
    
    ff3_monthly = (
        web.DataReader("F-F_Research_Data_Factors", "famafrench",
                       start = start_date, end = end_date)[0]
          .rename(columns={
              "Mkt-RF": "mkt_excess",
              "SMB": "smb",
              "HML": "hml",
              "RF": "rf"
          })
          .divide(100)
          .rename_axis("date")
          .reset_index()
          .assign(date = lambda df: pd.to_datetime(df["date"].astype(str)))
          .set_index("date")
    )
    
    ff3 = (1 + ff3_monthly).resample('QE').prod().subtract(1)
    
    return ff5_monthly, ff3_monthly, ff5, ff3
