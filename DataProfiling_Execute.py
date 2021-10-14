import time
import pyspark
from pyspark.sql import functions as F
import uuid
import pandas as pd
import numpy as np
from pyspark.sql.types import StructType
from statistics import mode
from collections import Counter

### Execute Profiling

profile_output = master_profile(table = data_pd
                                       ,tableName = table_name)