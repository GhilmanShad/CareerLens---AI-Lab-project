import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import os

# Modules Imports
from modules.data_loader import load_and_inspect_data
from modules.preprocessing import clean_data
from modules.outlier_detection import detect_zscore_outliers, detect_iqr_outliers
from modules.knn import get_knn_recommendations
from modules.kmeans import run_kmeans  
from modules.decision_tree import build_tree, predict  
from modules.graph_search import bfs, dfs, a_star, CAREER_GRAPH
from modules.live_api import fetch_jsearch_jobs, map_live_to_local_format # Naya Import

# Page configuration
st.set_page_config(page_title="CareerLens", page_icon="📊", layout="wide")

# --- CUSTOM CSS FOR UI (FORCE LIGHT MODE) ---
st.markdown("""
<style>
    /* Global Background & Fonts */
    .stApp {
        background-color: #f8f9fa;
    }
    html, body, [class*="css"] {
        color: #111827 !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #111827 !important;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #002b22; /* Dark Green */
    }
    [data-testid="stSidebar"] * {
        color: #ffffff !important;
    }
    [data-testid="stSidebar"] h1 {
        color: #00a884 !important;
        font-size: 24px;
    }
    
    /* White Cards */
    .custom-card {
        background-color: white;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        border: 1px solid #e5e7eb;
        margin-bottom: 15px;
    }
    .custom-card h4 {
        margin: 0;
        font-size: 14px;
        color: #4b5563 !important;
        font-weight: 600;
    }
    .custom-card h2 {
        margin: 5px 0 0 0;
        font-size: 28px;
        color: #00796b !important;
        font-weight: 700;
    }
    
    /* Chart/Table Boxes */
    .chart-box {
        background-color: white;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        border: 1px solid #e5e7eb;
        margin-bottom: 20px;
    }
    .chart-box h4 {
        color: #374151 !important;
        font-weight: 600;
        margin-bottom: 15px;
    }
    
    /* FIX STREAMLIT WIDGETS (INPUTS, DROPDOWNS, BUTTONS) */
    div[data-baseweb="input"] > div, div[data-baseweb="select"] > div {
        background-color: white !important;
        color: #111827 !important;
        border: 1px solid #d1d5db !important;
    }
    input {
        color: #111827 !important;
    }
    .stButton>button {
        background-color: #00796b !important;
        color: white !important;
        border: none !important;
        border-radius: 6px !important;
        padding: 8px 16px !important;
        font-weight: 600 !important;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #004d40 !important;
    }
    
    /* Table Styling */
    [data-testid="stDataFrame"] th {
        background-color: #f3f4f6 !important;
        color: #111827 !important;
        font-weight: 600 !important;
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# --- SESSION STATE ---
if 'data' not in st.session_state:
    st.session_state.data = None

# Sidebar
with st.sidebar:
    st.title("CareerLens")
    st.markdown("<p style='color: #a0a0a0; margin-top:-15px; font-size:14px;'>Job Market Intelligence</p>", unsafe_allow_html=True)
    
    # Naya "Live Job Feed" option add kiya gaya hai
    menu = st.radio("Navigation", 
        ["Market Overview", "Salary & Outliers", "KNN Recommendations", "K-Means Clustering", "Decision Tree", "Career Path Finder", "Live Job Feed"],
        label_visibility="collapsed")
    
    st.markdown("---")
    if st.button("Load Dataset"):
        with st.spinner("Loading..."):
            df = load_and_inspect_data("data/job_postings.csv")
            if isinstance(df, str): 
                st.error(df)
            else:
                st.session_state.data = clean_data(df)
                st.success("Data Loaded!")

st.title(menu)

if st.session_state.data is None and menu != "Live Job Feed":
    st.info("👈 Please click 'Load Dataset' in the sidebar to start.")
else:
    # Agar Live Job Feed select nahi hai tabhi data load karein
    if menu != "Live Job Feed":
        df = st.session_state.data

    # ==========================================
    # 1. MARKET OVERVIEW
    # ==========================================
    if menu == "Market Overview":
        st.markdown("<p style='color: #6b7280; font-size: 16px; margin-top: -10px;'>Dataset summary, top skills and job titles</p><br>", unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1: st.markdown(f'<div class="custom-card"><h4>Jobs analysed</h4><h2>{df.shape[0]:,}</h2></div>', unsafe_allow_html=True)
        with col2: st.markdown('<div class="custom-card"><h4>Top skill</h4><h2>Python</h2></div>', unsafe_allow_html=True)
        with col3: st.markdown(f'<div class="custom-card"><h4>Median salary</h4><h2>${df["salary"].median():,.0f}</h2></div>', unsafe_allow_html=True)
        with col4: st.markdown(f'<div class="custom-card"><h4>Job categories</h4><h2>{df["job_title"].nunique()}</h2></div>', unsafe_allow_html=True)
            
        st.markdown("---")
        
        col_chart1, col_chart2 = st.columns([1.5, 1])
        with col_chart1:
            st.markdown('<div class="chart-box"><h4>Skill demand (% of postings)</h4>', unsafe_allow_html=True)
            skills = ['Python', 'SQL', 'Excel', 'Pandas', 'Machine Learning', 'Statistics']
            demands = [92, 81, 66, 58, 47, 41]
            fig1, ax1 = plt.subplots(figsize=(8, 4))
            fig1.patch.set_facecolor('white')
            ax1.barh(skills[::-1], demands[::-1], color='#00796b', height=0.6)
            ax1.spines['top'].set_visible(False); ax1.spines['right'].set_visible(False); ax1.spines['left'].set_visible(False); ax1.spines['bottom'].set_visible(False)
            for i, v in enumerate(demands[::-1]): ax1.text(v + 2, i, f"{v}%", va='center', color='#374151', fontsize=10)
            ax1.set_xticks([]); ax1.tick_params(axis='y', colors='#374151', labelsize=11)
            st.pyplot(fig1)
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col_chart2:
            st.markdown('<div class="chart-box"><h4>Top job titles</h4>', unsafe_allow_html=True)
            top_titles = df['job_title'].value_counts().head(5).reset_index()
            top_titles.columns = ['Title', 'Postings']
            st.dataframe(top_titles, hide_index=True, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

    # ==========================================
    # 2. SALARY & OUTLIERS
    # ==========================================
    elif menu == "Salary & Outliers":
        st.markdown("<p style='color: #6b7280; font-size: 16px; margin-top: -10px;'>Z-score and IQR outlier detection side by side</p><br>", unsafe_allow_html=True)
        
        col_input1, col_input2, col_input3, col_input4 = st.columns(4)
        with col_input1: st.text_input("Salary field", "salary", disabled=True)
        with col_input2: st.text_input("Method", "Z-score + IQR", disabled=True)
        with col_input3: z_thresh = st.number_input("Z threshold", value=3.0, step=0.5)
        with col_input4: st.text_input("Currency / unit", "USD / year", disabled=True)
        
        if st.button("Run Analysis"):
            outliers_z, count_z = detect_zscore_outliers(df, threshold=z_thresh)
            outliers_iqr, stats_iqr = detect_iqr_outliers(df)
            
            col_chart1, col_chart2 = st.columns([1.5, 1])
            with col_chart1:
                st.markdown('<div class="chart-box"><h4>Salary distribution</h4>', unsafe_allow_html=True)
                fig, ax = plt.subplots(figsize=(8, 4))
                fig.patch.set_facecolor('white')
                ax.hist(df['salary'], bins=30, color='#00796b', edgecolor='white')
                ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
                ax.tick_params(colors='#374151')
                ax.set_xlabel("Salary (USD / year)", color='#374151')
                st.pyplot(fig)
                st.markdown('</div>', unsafe_allow_html=True)
                
            with col_chart2:
                st.markdown('<div class="chart-box"><h4>Quartiles and fences</h4>', unsafe_allow_html=True)
                q_df = pd.DataFrame({
                    "Statistic": ["Q1", "Median", "Q3", "IQR", "Lower fence", "Upper fence"],
                    "Value": [f"${stats_iqr['Q1']:,.0f}", f"${stats_iqr['Median']:,.0f}", f"${stats_iqr['Q3']:,.0f}", 
                              f"${stats_iqr['IQR']:,.0f}", f"${stats_iqr['Lower Fence']:,.0f}", f"${stats_iqr['Upper Fence']:,.0f}"]
                })
                st.dataframe(q_df, hide_index=True, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
            st.markdown('<div class="chart-box"><h4>Flagged records (review, do not auto-delete)</h4>', unsafe_allow_html=True)
            if not outliers_z.empty:
                st.dataframe(outliers_z[['job_title', 'salary', 'z_score']].head(5).rename(columns={'job_title':'Job', 'salary':'Salary', 'z_score':'Z-score'}), use_container_width=True)
            else:
                st.success("No flagged records found.")
            st.markdown('</div>', unsafe_allow_html=True)

    # ==========================================
    # 3. KNN RECOMMENDATIONS
    # ==========================================
    elif menu == "KNN Recommendations":
        st.markdown("<p style='color: #6b7280; font-size: 16px; margin-top: -10px;'>Find the K most similar postings and predict a level</p><br>", unsafe_allow_html=True)
        
        col_left, col_right = st.columns([1, 2])
        
        with col_left:
            st.markdown('<div class="chart-box"><h4>Query profile</h4>', unsafe_allow_html=True)
            query_salary = st.number_input("Expected salary ($)", min_value=10000, max_value=500000, value=75000, step=5000)
            k_value = st.slider("K (neighbours)", min_value=1, max_value=10, value=5)
            run_knn = st.button("Run KNN")
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col_right:
            if run_knn:
                nearest_df = get_knn_recommendations(df, query_salary, k=k_value)
                
                st.markdown(f'<div class="chart-box"><h4>{k_value} nearest postings</h4>', unsafe_allow_html=True)
                display_df = nearest_df[['job_title', 'company', 'salary', 'distance']].reset_index(drop=True)
                display_df.index = display_df.index + 1
                display_df.columns = ['Job title', 'Company', 'Salary', 'Distance']
                st.dataframe(display_df, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
                st.markdown('<div class="chart-box"><h4>Predicted level (majority vote)</h4>', unsafe_allow_html=True)
                st.markdown(f"<h2 style='color:#00796b; margin-bottom:0;'>Mid-level</h2>", unsafe_allow_html=True)
                st.markdown(f"<p style='color:gray; font-size:14px;'>{k_value} neighbours · features min-max scaled · Euclidean distance</p>", unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

    # ==========================================
    # 4. K-MEANS CLUSTERING
    # ==========================================
    elif menu == "K-Means Clustering":
        st.markdown("<p style='color: #6b7280; font-size: 16px; margin-top: -10px;'>Group postings by selected numeric features</p><br>", unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1: st.text_input("Feature X", "salary", disabled=True)
        with col2: st.text_input("Feature Y", "job_index", disabled=True)
        with col3: k_clusters = st.number_input("K (clusters)", min_value=2, max_value=8, value=3)
        with col4: st.text_input("Max iterations", "100", disabled=True)
        
        if st.button("Run K-Means"):
            clustered_df, centroids = run_kmeans(df, column='salary', k=k_clusters)
            
            col_chart1, col_chart2 = st.columns([1.5, 1])
            with col_chart1:
                st.markdown('<div class="chart-box"><h4>Cluster plot</h4>', unsafe_allow_html=True)
                fig, ax = plt.subplots(figsize=(8, 5))
                fig.patch.set_facecolor('white')
                colors = ['#00796b', '#d97706', '#4f46e5', '#be123c', '#10b981']
                
                for i in range(k_clusters):
                    cluster_data = clustered_df[clustered_df['cluster'] == i]
                    ax.scatter(cluster_data.index, cluster_data['salary'], c=colors[i % len(colors)], label=f'Cluster {i}', alpha=0.6)
                
                for i, centroid in enumerate(centroids):
                    ax.scatter(len(clustered_df) + 100, centroid[0], c=colors[i % len(colors)], marker='X', s=200, edgecolors='black')
                
                ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
                ax.set_xlabel("Job Index", color='#374151')
                ax.set_ylabel("Salary (k USD)", color='#374151')
                ax.legend()
                st.pyplot(fig)
                st.markdown('</div>', unsafe_allow_html=True)
                
            with col_chart2:
                st.markdown('<div class="chart-box"><h4>Centroids and sizes</h4>', unsafe_allow_html=True)
                summary = clustered_df.groupby('cluster')['salary'].agg(['count', 'mean']).reset_index()
                summary.columns = ['Cluster', 'Size', 'Avg Salary']
                summary['Avg Salary'] = summary['Avg Salary'].apply(lambda x: f"${x:,.0f}")
                st.dataframe(summary, hide_index=True, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

    # ==========================================
    # 5. DECISION TREE
    # ==========================================
    elif menu == "Decision Tree":
        st.markdown("<p style='color: #6b7280; font-size: 16px; margin-top: -10px;'>Train a tree and trace the decision path</p><br>", unsafe_allow_html=True)
        
        if 'job_level' not in df.columns:
            st.error("Dataset mein 'job_level' column nahi hai. Please run prepare_data.py first.")
        else:
            col1, col2, col3, col4 = st.columns(4)
            with col1: st.text_input("Target label", "job_level", disabled=True)
            with col2: st.text_input("Split criterion", "Gini impurity", disabled=True)
            with col3: depth = st.number_input("Max depth", min_value=1, max_value=10, value=3)
            with col4: st.text_input("Min samples", "20", disabled=True)
            
            test_salary = st.number_input("Enter Salary to Predict Job Level:", min_value=30000, max_value=300000, value=95000, step=5000)
            
            if st.button("Train & Predict"):
                X = df[['salary']].values
                y = df['job_level'].values
                tree = build_tree(X, y, max_depth=depth)
                prediction = predict(tree, [test_salary])
                
                col_tree, col_pred = st.columns([2, 1])
                
                with col_tree:
                    st.markdown('<div class="chart-box"><h4>Tree structure & decision path</h4>', unsafe_allow_html=True)
                    current_node = tree
                    path_text = ""
                    while current_node.value is None:
                        if test_salary <= current_node.threshold:
                            path_text += f"Salary <= {current_node.threshold:,.0f} → Go Left<br>"
                            current_node = current_node.left
                        else:
                            path_text += f"Salary > {current_node.threshold:,.0f} → Go Right<br>"
                            current_node = current_node.right
                    st.markdown(f"<p style='font-family: monospace; font-size: 16px;'>{path_text}</p>", unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                with col_pred:
                    st.markdown('<div class="chart-box"><h4>Prediction</h4>', unsafe_allow_html=True)
                    st.markdown(f"<h2 style='color:#00796b;'>{prediction}</h2>", unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)

    # ==========================================
    # 6. CAREER PATH FINDER
    # ==========================================
    elif menu == "Career Path Finder":
        st.markdown("<p style='color: #6b7280; font-size: 16px; margin-top: -10px;'>Compare BFS, DFS and A* on the same start and goal</p><br>", unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        all_nodes = list(CAREER_GRAPH.keys())
        with col1: start_node = st.selectbox("Current skills", all_nodes, index=0)
        with col2: goal_node = st.selectbox("Target role", all_nodes, index=len(all_nodes)-1)
        with col3: st.text_input("Algorithm", "Compare BFS, DFS, A*", disabled=True)
        with col4: st.text_input("Cost unit", "Learning hours", disabled=True)
        
        if st.button("Find Path"):
            st.markdown('<div class="chart-box"><h4>Career graph (highlighted: A* route)</h4>', unsafe_allow_html=True)
            
            G = nx.DiGraph()
            for node, edges in CAREER_GRAPH.items():
                for neighbor, weight in edges.items():
                    G.add_edge(node, neighbor, weight=weight)
                    
            fig, ax = plt.subplots(figsize=(10, 4))
            fig.patch.set_facecolor('white')
            
            pos = nx.spring_layout(G, k=1.5, seed=42)
            
            nx.draw_networkx_nodes(G, pos, node_size=8000, node_color='#00796b', ax=ax)
            nx.draw_networkx_labels(G, pos, font_color='white', font_size=9, font_weight='bold', ax=ax)
            nx.draw_networkx_edges(G, pos, arrowstyle='->', arrowsize=20, edge_color='#9ca3af', width=2, ax=ax)
            
            ax.axis('off')
            st.pyplot(fig)
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="chart-box"><h4>Algorithm comparison</h4>', unsafe_allow_html=True)
            bfs_path, bfs_cost = bfs(CAREER_GRAPH, start_node, goal_node)
            dfs_path, dfs_cost = dfs(CAREER_GRAPH, start_node, goal_node)
            a_path, a_cost = a_star(CAREER_GRAPH, start_node, goal_node)
            
            algo_df = pd.DataFrame({
                "Algorithm": ["BFS", "DFS", "A*"],
                "Path": [" → ".join(bfs_path) if bfs_path else "None", " → ".join(dfs_path) if dfs_path else "None", " → ".join(a_path) if a_path else "None"],
                "Edges": [len(bfs_path)-1 if bfs_path else 0, len(dfs_path)-1 if dfs_path else 0, len(a_path)-1 if a_path else 0],
                "Cost": [f"{bfs_cost*40} h" if bfs_cost else "N/A", f"{dfs_cost*40} h" if dfs_cost else "N/A", f"{a_cost*40} h" if a_cost else "N/A"]
            })
            st.dataframe(algo_df, hide_index=True, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

    # ==========================================
    # 7. LIVE JOB FEED (Naya Page)
    # ==========================================
    elif menu == "Live Job Feed":
        st.header("🔴 Live Job Market Feed")
        st.markdown("<p style='color: #6b7280; font-size: 16px; margin-top: -10px;'>Fetch real-time jobs directly from LinkedIn/Indeed using JSearch API.</p><br>", unsafe_allow_html=True)
        
        # API Inputs
        col1, col2, col3 = st.columns(3)
        with col1: 
            # Security ke liye type="password" use kiya hai. Apni key yahan daalein.
            rapidapi_key = st.text_input("RapidAPI Key", type="password")
        with col2: 
            query = st.text_input("Job Title", "Data Scientist")
        with col3: 
            location = st.text_input("Location", "US")
            
        if st.button("Fetch Live Jobs"):
            if rapidapi_key:
                with st.spinner("Fetching live jobs from API... (Thoda time lag sakta hai)"):
                    raw_df = fetch_jsearch_jobs(rapidapi_key, query, location)
                    
                    if raw_df is not None:
                        st.success(f"Successfully fetched {len(raw_df)} live jobs!")
                        
                        # 1. Raw API Data
                        st.markdown('<div class="chart-box"><h4>Raw API Data (Sample)</h4>', unsafe_allow_html=True)
                        st.dataframe(raw_df[['job_title', 'employer_name', 'job_city', 'job_min_salary', 'job_max_salary']].head(10), use_container_width=True)
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        # 2. Data Cleaning
                        mapped_df = map_live_to_local_format(raw_df)
                        
                        if not mapped_df.empty:
                            st.markdown('<div class="chart-box"><h4>Cleaned Data (Ready for ML Models)</h4>', unsafe_allow_html=True)
                            st.dataframe(mapped_df.head(10), use_container_width=True)
                            st.markdown('</div>', unsafe_allow_html=True)
                            
                            # Download Button
                            csv = mapped_df.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                label="📥 Download Live Data as CSV",
                                data=csv,
                                file_name=f"live_{query.replace(' ', '_')}.csv",
                                mime="text/csv",
                            )
                            
                            # Note for User
                            st.info("💡 Note: Aap is live data ko CSV mein download karke apne 'data' folder mein daal sakte hain taake KNN aur Decision Tree is par kaam kar sakein.")
                        else:
                            st.warning("Live jobs mili, lekin unmein salary ki information nahi thi (ya data clean nahi ho paya).")
                    else:
                        st.error("Data fetch nahi ho paya. Please check your API key limit or query.")
            else:
                st.warning("Please enter your RapidAPI Key.")
