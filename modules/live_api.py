import requests
import pandas as pd
import streamlit as st

@st.cache_data(ttl=3600)
def fetch_jsearch_jobs(rapidapi_key, query="Data Scientist", location="US"):
    url = "https://jsearch.p.rapidapi.com/search"
    
    # num_pages hata diya hai aur query ko simple rakha hai
    querystring = {
        "query": query, 
        "page": "1",
        "country": "us" if location.upper() == "US" else location.lower()
    }
    
    headers = {
        "X-RapidAPI-Key": rapidapi_key,
        "X-RapidAPI-Host": "jsearch.p.rapidapi.com"
    }
    
    try:
        response = requests.get(url, headers=headers, params=querystring)
        response.raise_for_status()
        data = response.json()
        
        if data.get("status") == "OK" and data.get("data"):
            return pd.DataFrame(data["data"])
        else:
            return None
    except Exception as e:
        st.error(f"API Error: {e}")
        return None

def map_live_to_local_format(live_df):
    """JSearch data ko humare local dataset (job_title, company, salary) ke format mein laata hai."""
    if live_df is None or live_df.empty:
        return pd.DataFrame()

    mapped_df = pd.DataFrame()
    mapped_df['job_title'] = live_df['job_title']
    mapped_df['company'] = live_df['employer_name']
    
    if 'job_min_salary' in live_df.columns and 'job_max_salary' in live_df.columns:
        mapped_df['salary'] = live_df[['job_min_salary', 'job_max_salary']].mean(axis=1)
    else:
        mapped_df['salary'] = np.nan
        
    mapped_df['description'] = live_df.get('job_description', "No description")
    mapped_df['job_level'] = 'Mid' 
    
    mapped_df = mapped_df.dropna(subset=['salary'])
    
    return mapped_df