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



#########
11/15/2021

#Generating a dataset for only categorical
df_categorical = df.select_dtypes(exclude=['number'])
df_categorical=df_categorical.drop(['Date.of.Birth','DisbursalDate'],axis=1)
df_categorical.head()

#Building a Dataset for numerical (continous)
df_continuous = df.select_dtypes(include=['number'])
df_continuous=df_continuous.drop(['UniqueID'],axis=1)
df_continuous.head()

#Univariate Analysis
import matplotlib.pyplot as plt # Charge matplotlib
import seaborn as sns   # Charge seaborn

#To obtain the basic statistics
df_continuous.describe()

#Get the List of all Column Names
continuous_list = list(df_continuous)

# Plot for all continous
#1
sns.displot(df['disbursed_amount'][df['disbursed_amount'] < df['disbursed_amount'].quantile(.99)],kind='hist',kde=True)
plt.show()

#2
sns.displot(df['asset_cost'][df['asset_cost'] < df['asset_cost'].quantile(.99)],kind='hist',kde=True)
plt.show()

#3
sns.displot(df['ltv'][df['ltv'] < df['ltv'].quantile(.99)],kind='hist',kde=True)
plt.show()

#4
sns.displot(df['PERFORM_CNS.SCORE'][df['PERFORM_CNS.SCORE'] < df['PERFORM_CNS.SCORE'].quantile(.99)],kind='hist',kde=True)
plt.show()

#5
sns.displot(df['PRI.NO.OF.ACCTS'][df['PRI.NO.OF.ACCTS'] < df['PRI.NO.OF.ACCTS'].quantile(.99)],kind='hist',kde=True)
plt.show()

#6
sns.displot(df['PRI.ACTIVE.ACCTS'][df['PRI.ACTIVE.ACCTS'] < df['PRI.ACTIVE.ACCTS'].quantile(.99)],kind='hist',kde=True)
plt.show()

#6
sns.displot(df['PRI.OVERDUE.ACCTS'][df['PRI.OVERDUE.ACCTS'] < df['PRI.OVERDUE.ACCTS'].quantile(.99)],kind='hist',kde=True)
plt.show()

#7
sns.displot(df['PRI.CURRENT.BALANCE'][df['PRI.CURRENT.BALANCE'] < df['PRI.CURRENT.BALANCE'].quantile(.99)],kind='hist',kde=True)
plt.show()

#8
sns.displot(df['PRI.SANCTIONED.AMOUNT'][df['PRI.SANCTIONED.AMOUNT'] < df['PRI.SANCTIONED.AMOUNT'].quantile(.99)],kind='hist',kde=True)
plt.show()

#9
sns.displot(df['PRI.DISBURSED.AMOUNT'][df['PRI.DISBURSED.AMOUNT'] < df['PRI.DISBURSED.AMOUNT'].quantile(.99)],kind='hist',kde=True)
plt.show()

#10
sns.displot(df['SEC.NO.OF.ACCTS'][df['SEC.NO.OF.ACCTS'] < df['SEC.NO.OF.ACCTS'].quantile(.99)],kind='hist',kde=True)
plt.show()

#11
sns.displot(df['SEC.ACTIVE.ACCTS'][df['SEC.ACTIVE.ACCTS'] < df['SEC.ACTIVE.ACCTS'].quantile(.99)],kind='hist',kde=True)
plt.show()

#12
sns.displot(df['SEC.OVERDUE.ACCTS'][df['SEC.OVERDUE.ACCTS'] < df['SEC.OVERDUE.ACCTS'].quantile(.99)],kind='hist',kde=True)
plt.show()

#13
sns.displot(df['SEC.CURRENT.BALANCE'][df['SEC.CURRENT.BALANCE'] < df['SEC.CURRENT.BALANCE'].quantile(.99)],kind='hist',kde=True)
plt.show()

#14
sns.displot(df['SEC.SANCTIONED.AMOUNT'][df['SEC.SANCTIONED.AMOUNT'] < df['SEC.SANCTIONED.AMOUNT'].quantile(.99)],kind='hist',kde=True)
plt.show()

#15
sns.displot(df['SEC.DISBURSED.AMOUNT'][df['SEC.DISBURSED.AMOUNT'] < df['SEC.DISBURSED.AMOUNT'].quantile(.99)],kind='hist',kde=True)
plt.show()

#16
sns.displot(df['PRIMARY.INSTAL.AMT'][df['PRIMARY.INSTAL.AMT'] < df['PRIMARY.INSTAL.AMT'].quantile(.99)],kind='hist',kde=True)
plt.show()

#17
sns.displot(df['NEW.ACCTS.IN.LAST.SIX.MONTHS'][df['NEW.ACCTS.IN.LAST.SIX.MONTHS'] < df['NEW.ACCTS.IN.LAST.SIX.MONTHS'].quantile(.99)],kind='hist',kde=True)
plt.show()

#18
sns.displot(df['DELINQUENT.ACCTS.IN.LAST.SIX.MONTHS'][df['DELINQUENT.ACCTS.IN.LAST.SIX.MONTHS'] < df['DELINQUENT.ACCTS.IN.LAST.SIX.MONTHS'].quantile(.99)],kind='hist',kde=True)
plt.show()

#19
sns.displot(df['AVERAGE.ACCT.AGE'][df['AVERAGE.ACCT.AGE'] < df['AVERAGE.ACCT.AGE'].quantile(.99)],kind='hist',kde=True)
plt.show()

#20
sns.displot(df['CREDIT.HISTORY.LENGTH'][df['CREDIT.HISTORY.LENGTH'] < df['CREDIT.HISTORY.LENGTH'].quantile(.99)],kind='hist',kde=True)
plt.show()

#21
sns.displot(df['CREDIT.HISTORY.LENGTH'][df['CREDIT.HISTORY.LENGTH'] < df['CREDIT.HISTORY.LENGTH'].quantile(.99)],kind='hist',kde=True)
plt.show()

#22
sns.displot(df['NO.OF_INQUIRIES'][df['NO.OF_INQUIRIES'] < df['NO.OF_INQUIRIES'].quantile(.99)],kind='hist',kde=True)
plt.show()

#23
sns.displot(df['Age'][df['Age'] < df['Age'].quantile(.99)],kind='hist',kde=True)
plt.show()

#23
sns.displot(df['Disbursal_months'][df['Disbursal_months'] < df['Disbursal_months'].quantile(.99)],kind='hist',kde=True)
plt.show()

########Multivariate Analysis

plt.rcParams["figure.figsize"] = (10,7)
sns.heatmap(df_continuous.corr())
plt.show()

#Heat map
sns.heatmap(df_continuous.corr(), cmap="YlGnBu", annot=False,mask=np.triu(df_continuous.corr()))
plt.show()

#Heat map that highligts if the correlation is greater than 0.6
sns.heatmap(df_continuous.corr().abs()>0.6, cmap="YlGnBu", annot=False,mask=np.triu(df_continuous.corr()))
plt.show() # black are with the highest correlation

def get_redundant_pairs(df):
    '''Get diagonal and lower triangular pairs of correlation matrix'''
    pairs_to_drop = set()
    cols = df.columns
    for i in range(0, df.shape[1]):
        for j in range(0, i+1):
            pairs_to_drop.add((cols[i], cols[j]))
    return pairs_to_drop

def get_top_abs_correlations(df, n=5):
    au_corr = df.corr().abs().unstack()
    labels_to_drop = get_redundant_pairs(df)
    au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)
    return au_corr[0:n]

list1=get_top_abs_correlations(df_continuous,n=9)
print(list1)

