from pathlib import Path
import pandas as pd
from IPython.core.display_functions import display

project_root = Path(__file__).parent.parent.parent
sp_500_list_df = pd.read_csv((project_root / "data" / "stock_lists" / "sp_500_list.csv"))

display(sp_500_list_df)