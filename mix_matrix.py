import pandas as pd
import numpy as np

import streamlit as st

df1 = pd.DataFrame()
df2 = pd.DataFrame()
df3 = pd.DataFrame()

df1['id'] = [0,1,2,3,4,5,6,7,8,9]
df1['set1'] = [4 ,7 ,1 ,3 ,4 ,np.nan, np.nan, 10, 11, 111]
df1['id1'] = [1,1,2,2,3,3,4,4,5,5]


df2['id'] = [0,1,2,3,4,5,6,7,8,9]
df2['set1'] = [5 ,8 ,2 ,np.nan ,5 ,6, 4, 10, 11, 222]
df2['id1'] = [1,1,2,2,3,3,4,4,5,5]


df3['id'] = [1,2,3,4,5]
df3['id1'] = [1,2,3,4,5]
df3['set3'] = [21,22,23,24,25]


st.write('df1', df1)
st.write('df2', df2)
st.write('df3', df3)

st.title('concat')

st.title('merge')

df_merge = df1.merge(df3, on = ['id1'])
st.write(df_merge)

df_merge = df1.merge(df3, on = ['id'])
st.write(df_merge)

st.title('join')

df_join = df1.join(df3, on = 'id1', lsuffix = '_madre', rsuffix='_figlio', how = 'outer')
st.write(df_join)

df_join = df1.join(df3, on = 'id', lsuffix = '_madre', rsuffix='_figlio', how = 'outer')
st.write(df_join)
