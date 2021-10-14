import time
import pyspark
from pyspark.sql import functions as F
import uuid
import pandas as pd
import numpy as np
from pyspark.sql.types import StructType
from statistics import mode
from collections import Counter
import re
import Levenshtein as lev
from pyspark.sql.types import StructType
from statistics import mode
from collections import Counter

### Sampling Function
def sampling_test(x):
  if x.count() > 10000000 and x.count() < 100000000:
    x = x.withColumn("Sample", F.lit("Sampled"))
    return x.sample(.1)
  if x.count() > 100000000:
    x = x.withColumn("Sample", F.lit("Sampled"))
    return x.sample(.01)
  else:
    x = x.withColumn("Sample", F.lit("Not Sampled"))
    return x


### Create Functions, Step 1

# First, create individual functions for generating values like counts, averages, etc.

def row_count(x):
  # Counts number of overall records, including Nulls, in a column
  list = x.tolist()
  output = len(list)
  return output


def count_missing(x):
  # Counts the number of missing or Null records in a column
  list = x.tolist()
  missing = []
  for x in list:
    if x is None:
      missing.append("1")
  count = len(missing)
  return count


def completeness(x):
  # Calculates the percentage of a column that is populated with non-Null values
  list = x.tolist()
  missing = []
  complete = []
  for x in list:
    if x is None:
      missing.append("1")
    else:
      complete.append("1")
  completeness = len(complete) / (len(complete) + len(missing))
  return completeness


def uniqueness(x):
  # Calculates the number of unique values in a column
  list = x.tolist()
  uni = set(list)
  unique = len(uni)
  return unique


def max_length(x):
  # Calculates the length of the longest individual element within a column
  list = x.tolist()
  length = -1
  res = 0
  for i in list:
    if i is None:
      continue
    if len(i) > length:
      length = len(i)
      res = i
  if res == 0:
    return "Column is Empty"
  else:
    return len(res)


def min_length(x):
  # Calculates the length of the shortest individual element within a column
  list = x.tolist()
  length = 9999999999
  res = 0
  for i in list:
    if i is None:
      continue
    #     if pd.core.dtypes.common.is_datetime_or_timedelta_dtype(x) is True:
    #       continue
    if len(i) < length:
      length = len(i)
      res = i
  if res == 0:
    return "Column is Empty"
  else:
    return len(res)


def max_value(x):
  # Calculates the maximum value of a column
  list = x.tolist()
  new_list = []
  for x in list:
    if x is None:
      continue
    if x == "NULL":
      continue
    else:
      try:
        float_ = float(x)
        int_ = int(float_)
        new_list.append(int_)
      except:
        pass
  if len(new_list) == 0:
    return "No Numeric Values"
  else:
    max_val = max(new_list)
    return max_val


def min_value(x):
  # Calculates the minimum value of a column
  list = x.tolist()
  new_list = []
  for x in list:
    if x is None:
      continue
    if x == "NULL":
      continue
    else:
      try:
        float_ = float(x)
        int_ = int(float_)
        new_list.append(int_)
      except:
        pass
  if len(new_list) == 0:
    return "No Numeric Values"
  else:
    min_val = min(new_list)
    return min_val


def get_average(x):
  # calculates the average of a numeric column
  list = x.tolist()
  sum_num = 0
  new = []
  for x in list:
    if x is None:
      continue
    if x == "NULL":
      continue
    else:
      try:
        float_ = float(x)
        int_ = int(float_)
        sum_num = sum_num + int_
        new.append(int_)
      except:
        pass
  if len(new) == 0:
    return "No Numeric Values"
  else:
    test = sum_num / len(new)
    return test


def get_stdev(x):
  list = x.tolist()
  new = []
  for x in list:
    if x is None:
      continue
    if x == "NULL":
      continue
    else:
      try:
        float_ = float(x)
        int_ = int(float_)
        new.append(int_)
      except:
        pass
  if len(new) == 0:
    return "No Numeric Values"
  else:
    output = float(np.std(new))
    return output


def get_median(x):
  # Calculates the median of a numeric column
  list = x.tolist()
  new = []
  for x in list:
    if x is None:
      continue
    if x == "NULL":
      continue
    else:
      try:
        float_ = float(x)
        int_ = int(float_)
        new.append(int_)
      except:
        pass
  if len(new) == 0:
    return "No Numeric Values"
  else:
    output = float(np.median(new))
    return output


def most_frequent(x):
  # Identifies the most frequently occurring value in a column
  list = x.tolist()
  new = []
  for x in list:
    if x is None:
      continue
    if x == "NULL":
      continue
    else:
      new.append(x)
  if len(new) == 0:
    return "Column is Empty"
  else:
    most_common = mode(new)
  return most_common


def get_first_mode(a):
  # Second attempt at finding most frequent value
  list = a.tolist()
  new = []
  for x in list:
    if x is None:
      continue
    if x == "NULL":
      continue
    else:
      new.append(x)
  if len(new) == 0:
    return "Column is Empty"
  else:
    c = Counter(new)
    mode_count = max(c.values())
    mode = {key for key, count in c.items() if count == mode_count}
    first_mode = next(x for x in a if x in mode)
    return first_mode

def get_sample(x):
  # Used to limit the size of a profiled dataset in order to save on compute. Optional use.
  #   list = x.tolist()
  output = len(x.index)
  if output > 1000000:
    new_output = x.sample(n=1000000, replace=False)
    return new_output
  else:
    return x

### Create Functions, Step 2

# Create master function that executes individual functions against every column in a dataset

def master_profile(table, tableName):
  table_name = []
  column = []
  dtype= []
  rows = []
  missing = []
  complete = []
  unique = []
  minLength = []
  maxLength =[]
  minValue = []
  maxValue = []
  avgValue = []
  medValue = []
  stdValue = []
  freqValue = []
  for col in table:
    table_name.append(str(tableName))
    column.append(str(col))
    dtype.append(str(type(table[col].iloc[1])))
    rows.append(str(row_count(table[col])))
    missing.append(str(count_missing(table[col])))
    complete.append(str(completeness(table[col])))
    unique.append(str(uniqueness(table[col])))
    minValue.append(str(min_value(table[col])))
    maxValue.append(str(max_value(table[col])))
    avgValue.append(str(get_average(table[col])))
    medValue.append(str(get_median(table[col])))
    stdValue.append(str(get_stdev(table[col])))
    freqValue.append(str(get_first_mode(table[col])))
  df = pd.DataFrame([table_name
                     ,column
                     ,dtype
                     ,rows
                     ,missing
                     ,complete
                     ,unique
                     ,minValue
                     ,maxValue
                     ,avgValue
                     ,medValue
                     ,stdValue
                     ,freqValue
                    ])
  df = df.transpose()
  df.columns = ["TableName"
                ,"ColumnName"
                ,"DataType"
                ,"#Records"
                ,"#Missing"
                ,"%Complete"
                ,"#UniqueValues"
                ,"MinimumValue"
                ,"MaximumValue"
                ,"Average"
                ,"Median"
                ,"StandardDeviation"
                ,"MostFrequentValue"
               ]
  return df


# Round 1
### Initialize Matching Functions

def get_exact_field_match(a, b):
  ## This function checks if there are any extact string matches between field names in two tables.
  # initalize empty lists
  match_type = []
  table_name_a = []
  table_name_b = []
  # conduct matching
  s = set(a.FieldName).intersection(b.FieldName)
  new_list = list(s)
  # build lists for output dataframe
  for i in new_list:
    match_type.append("suggestedMatch_exactString")
    table_name_a.append(a.TableName[0])
    table_name_b.append(b.TableName[0])
  # build output dataframe
  df = pd.DataFrame([table_name_a
                      , new_list
                      , table_name_b
                      , new_list
                      , match_type])
  df = df.transpose()
  df.columns = ["TableNameA"
    , "FieldNameA"
    , "TableNameB"
    , "FieldNameB"
    , "MatchType"]
  # return output dataframe
  return df


def get_modified_field_match(a, b):
  ## This function checks if there are any string matches between field names in two tables after cleaning the field name strings.
  # initalize empty lists
  match_type = []
  a_clean = []
  b_clean = []
  # rename columns
  a = a.rename(columns={"TableName": "TableNameA", "FieldName": "FieldNameA"})
  b = b.rename(columns={"TableName": "TableNameB", "FieldName": "FieldNameB"})
  # remove non-letter characters, convert to lower case
  for i in a.FieldNameA:
    regex = re.compile('[^a-zA-Z]')
    a_clean.append(regex.sub('', i).lower())
    a['FieldNameCleaned'] = pd.Series(a_clean)
  for i in b.FieldNameB:
    regex = re.compile('[^a-zA-Z]')
    b_clean.append(regex.sub('', i).lower())
    b['FieldNameCleaned'] = pd.Series(b_clean)
  # conduct matching
  s = set(a.FieldNameCleaned).intersection(b.FieldNameCleaned)
  new_list = list(s)
  # join to distill table to only matching value(s)
  df = pd.concat([a, b], axis=1, join="inner")
  df = a.merge(b, on='FieldNameCleaned', how='inner')
  for i in new_list:
    match_type.append("suggestedMatch_modifiedString")
    df['MatchType'] = pd.Series(match_type)
  df = df.drop(["FieldNameCleaned"], axis=1)
  # return output dataframe
  return df


def get_clean_match(a, b):
  ## This function checks two cleaned strings against each other, to be used modularly
  a_clean = []
  b_clean = []
  # remove non-letter characters, convert to lower case
  regex = re.compile('[^a-zA-Z]')
  a = regex.sub('', a).lower()
  regex = re.compile('[^a-zA-Z]')
  b = regex.sub('', b).lower()
  if a == b:
    return "suggestedMatch_modifiedString"
  else:
    return "no match"


get_clean_match(a="PatientID", b="patient_id")


def get_levFuzzy_field_match(a, b):
  ## This function checks the Levenshtein Ratio between field names in two tables and returns those above a set threshold.
  test = []
  test2 = []
  ratio = []
  matchType = []
  for i in a.FieldName:
    for j in b.FieldName:
      test.append(i)
      test2.append(j)
      ratio.append(lev.ratio(i.lower(), j.lower()))
  df = pd.DataFrame([test, test2, ratio])
  df = df.transpose()
  df.columns = ["FieldNameA"
    , "FieldNameB"
    , "LevenshteinRatio"]
  # Remove all below threshold, set currently to 90%
  df = df[df["LevenshteinRatio"] > .9]
  # Remove all exact matches
  df = df[df["LevenshteinRatio"] != 1]
  # Join on Table A
  df = pd.concat([df, a], axis=1, join="inner")
  df = df.rename(columns={"TableName": "TableNameA"})
  # Join on Table B
  df = pd.concat([df, b], axis=1, join="inner")
  df = df.rename(columns={"TableName": "TableNameB"})
  # Add Match Type
  for i in df.LevenshteinRatio:
    matchType.append("suggestedMatch_fuzzyString")
  df["MatchType"] = pd.Series(matchType)
  # Select and Reorder
  df = df[['TableNameA', 'FieldNameA', 'TableNameB', 'FieldNameB', 'MatchType']]
  return df


def get_levo_ratio(a, b):
  ratio = lev.ratio(a.lower(), b.lower())
  if ratio > .9:
    return "suggestedMatch_fuzzyString"
  else:
    return "no_match"


def master_field_match(a, b):
  ## This function runs the previous functions together
  # run matching checks
  df_exact = get_exact_field_match(a, b)
  df_modified = get_modified_field_match(a, b)
  df_fuzzy = get_levFuzzy_field_match(a, b)
  # create master table
  frames = [df_exact, df_modified]
  result = pd.concat(frames)
  # remove duplicates, giving priority to exact match
  result["dupe_check"] = result["TableNameA"] + result["FieldNameA"] + result["TableNameB"] + result["FieldNameB"]
  result = result.drop_duplicates(subset="dupe_check", keep="first")
  result = result.drop(["dupe_check"], axis=1)
  # return result
  return result

