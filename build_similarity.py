import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json

fp='Airplane_Crashes_and_Fatalities_Since_1908.csv'
df=pd.read_csv(fp)
df['Date']=pd.to_datetime(df['Date'], errors='coerce')
df['Year']=df['Date'].dt.year

def build_text(row):
    parts=[row.get('Summary',''), row.get('Operator',''), row.get('AC Type',''), row.get('Route','')]
    return ' '.join([str(p) for p in parts if isinstance(p,str)])

df['text']=df.apply(build_text, axis=1)
mask=df['text'].str.strip().astype(bool)
filtered=df[mask].copy()
vectorizer=TfidfVectorizer(stop_words='english', ngram_range=(1,2), min_df=2)
X=vectorizer.fit_transform(filtered['text'])
index_map=filtered.index.to_list()

queries={
    'hudson_ditching': {'label':'US Airways 1549 Hudson River ditching','condition': (df['Date']==pd.Timestamp('2009-01-15')) & df['Operator'].str.contains('US Airways', na=False)},
    'mh17': {'label':'Malaysia Airlines Flight 17 shootdown','condition': (df['Date']==pd.Timestamp('2014-07-17')) & df['Operator'].str.contains('Malaysia', na=False)},
    'lion_air_610': {'label':'Lion Air Flight 610 JT610 (737 MAX)','condition': (df['Date']==pd.Timestamp('2018-10-29')) & df['Operator'].str.contains('Lion', na=False)}
}

results={}
for key,meta in queries.items():
    ids=df[meta['condition']].index.intersection(filtered.index)
    if not len(ids):
        continue
    idx=ids[0]
    pos=index_map.index(idx)
    sims=cosine_similarity(X[pos], X).flatten()
    ranked=sorted([(s,i) for i,s in enumerate(sims) if index_map[i]!=idx], reverse=True)
    top=ranked[:10]
    entries=[]
    for score,pos_idx in top:
        rid=index_map[pos_idx]
        row=df.loc[rid]
        entries.append({
            'rank': len(entries)+1,
            'score': float(score),
            'date': row['Date'].strftime('%Y-%m-%d') if pd.notna(row['Date']) else None,
            'operator': row['Operator'],
            'location': row['Location'],
            'flight': row['Flight #'],
            'ac_type': row['AC Type'],
            'route': row['Route'],
            'fatalities': row['Fatalities'],
            'aboard': row['Aboard'],
            'ground': row['Ground'],
            'summary': row['Summary']
        })
    results[key]={
        'query_label': meta['label'],
        'query_index': int(idx),
        'top_matches': entries
    }

with open('similarity_results.json','w',encoding='utf-8') as f:
    json.dump(results,f,ensure_ascii=False,indent=2)

print('saved similarity_results.json with',len(results),'queries')
