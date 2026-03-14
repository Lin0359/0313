import pandas as pd
fp='Airplane_Crashes_and_Fatalities_Since_1908.csv'
df=pd.read_csv(fp)
df['Date']=pd.to_datetime(df['Date'], errors='coerce')
df['Year']=df['Date'].dt.year
mask_h=df['Summary'].str.contains('Hudson', case=False, na=False)
print(df[mask_h][['Date','Operator','Flight #','Location','Summary']].head())
mask_m=df['Summary'].str.contains('Malaysia', case=False, na=False)
print('Malaysia rows',df[mask_m][['Date','Operator','Flight #','Summary']].head())
mask_737=df['Summary'].str.contains('737', case=False, na=False)
print('737 rows',df[mask_737][['Date','Operator','Flight #','Summary']].head())
