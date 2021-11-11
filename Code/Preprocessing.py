import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import re

# combine file first, save it as merge_file.csv
os.chdir(r'D:\GWU\Aihan\DATS 6103 Data Mining\Final Project\Code')
df1 = pd.read_csv(r'lt-vehicle-loan-default-prediction/train.csv')
df2 = pd.read_csv(r'lt-vehicle-loan-default-prediction/test.csv')
merge_df = pd.concat([df1, df2])
# merge_df.to_csv(r'lt-vehicle-loan-default-prediction/merge_file.csv')

# checking the format of each variable
print(merge_df.dtype)

# checking the null value first
print(merge_df.columns.isnull())
# There is no null value in this data.

# convert the format of 'AVERAGE.ACCT.AGE' and 'CREDIT.HISTORY.LENGTH' from 'xyrs xmon' to numbers that represent month.
def find_number(text):
    num = re.findall(r'[0-9]+',text)
    return int(num[0])*12 + int(num[1])

AVERAGE_ACCT_AGE_MON = merge_df['AVERAGE.ACCT.AGE'].apply(lambda x: find_number(x))
CREDIT_HISTORY_LENGTH_MON = merge_df['CREDIT.HISTORY.LENGTH'].apply(lambda x: find_number(x))

merge_df['AVERAGE.ACCT.AGE'] = AVERAGE_ACCT_AGE_MON
merge_df['CREDIT.HISTORY.LENGTH'] = CREDIT_HISTORY_LENGTH_MON

# convert categorical string to numbers
merge_df['Employment.Type'] = merge_df['Employment.Type'].astype('category')
merge_df['PERFORM_CNS.SCORE.DESCRIPTION'] = merge_df['PERFORM_CNS.SCORE.DESCRIPTION'].astype('category')
cat_columns = merge_df.select_dtypes(['category']).columns
merge_df[cat_columns] = merge_df[cat_columns].apply(lambda x: x.cat.codes)

# format the date variable
merge_df['Date.of.Birth'] = pd.to_datetime(merge_df['Date.of.Birth']).dt.strftime('%d/%m/%Y')

# covert Date of birth to age
from datetime import datetime
from dateutil.relativedelta import relativedelta

def age(born):
    born_date = datetime.strptime(born, "%d/%m/%Y").date()
    today = datetime.now()
    return relativedelta(today, born_date).years

merge_df['Age'] = merge_df['Date.of.Birth'].apply(age)


'''
FINISH FOR NOW
'''

# continuous_df = merge_df.iloc[[2,3,4,20,21,25,26,27,31,32,33,34,35,38]]


fig, axes = plt.subplots(7, 2, figsize=(18, 10))

fig.suptitle('Pokemon Stats by Generation')

sns.boxplot(ax=axes[0, 0], data=merge_df, x='disbursed_amount')
sns.boxplot(ax=axes[0, 1], data=merge_df, x='asset_cost')
sns.boxplot(ax=axes[1, 0], data=merge_df, x='ltv')
sns.boxplot(ax=axes[1, 1], data=merge_df, x='PERFORM_CNS.SCORE')
sns.boxplot(ax=axes[2, 1], data=merge_df, x='PRI.CURRENT.BALANCE')
sns.boxplot(ax=axes[3, 0], data=merge_df, x='PRI.SANCTIONED.AMOUNT')
sns.boxplot(ax=axes[3, 1], data=merge_df, x='PRI.DISBURSED.AMOUNT')
sns.boxplot(ax=axes[4, 0], data=merge_df, x='SEC.CURRENT.BALANCE')
sns.boxplot(ax=axes[4, 1], data=merge_df, x='SEC.SANCTIONED.AMOUNT')
sns.boxplot(ax=axes[5, 0], data=merge_df, x='SEC.DISBURSED.AMOUNT')
sns.boxplot(ax=axes[5, 1], data=merge_df, x='PRIMARY.INSTAL.AMT')
sns.boxplot(ax=axes[6, 0], data=merge_df, x='SEC.INSTAL.AMT')
sns.boxplot(ax=axes[6, 1], data=merge_df, x='AVERAGE.ACCT.AGE')
plt.show()
