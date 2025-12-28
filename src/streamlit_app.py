import streamlit as st

# Configure page
st.set_page_config(
    page_title="Drug Safety & Side Effect Analysis",
    page_icon="💊",
    layout="wide",
     initial_sidebar_state="expanded"
)

st.title("Drug Safety & Side Effect Analysis")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from wordcloud import WordCloud
import pickle
from pathlib import Path
import re

# Custom CSS
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
}
.sub-header {
    font-size: 1.5rem;
    color: #2c3e50;
    margin: 1rem 0;
}
.metric-card {
    background-color: #f8f9fa;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 4px solid #1f77b4;
}
.alert-info {
    background-color: #d1ecf1;
    border: 1px solid #bee5eb;
    border-radius: 0.25rem;
    padding: 0.75rem;
    margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load the processed dataset"""
    try:
        df = pd.read_csv("data/drugs_processed.csv")
        return df
    except FileNotFoundError:
        st.error("Please run the data processing notebooks first!")
        return None

@st.cache_data
def load_metadata():
    """Load feature engineering metadata"""
    try:
        with open("data/feature_engineering_metadata.pkl", "rb") as f:
            metadata = pickle.load(f)
        return metadata
    except FileNotFoundError:
        st.warning("Metadata file not found. Some features may not work properly.")
        return {}

def main():
    # Header
    st.markdown('<h1 class="main-header">💊 Drug Safety & Efficacy Analysis Dashboard</h1>', unsafe_allow_html=True)
    
    # Load data
    df = load_data()
    if df is None:
        st.stop()
    
    metadata = load_metadata()
    
    # Sidebar
    st.sidebar.title("🔍 Navigation")
    page = st.sidebar.selectbox(
        "Choose Analysis Type",
        ["🏠 Overview", "🔍 Drug Search", "📊 Safety Analysis", "⭐ Efficacy Analysis", 
         "😊 Sentiment Analysis", "⚠️ Risk Assessment", "📈 Comparative Analysis"]
    )
    
    # Main content based on page selection
    if page == "🏠 Overview":
        show_overview(df, metadata)
    elif page == "🔍 Drug Search":
        show_drug_search(df)
    elif page == "📊 Safety Analysis":
        show_safety_analysis(df)
    elif page == "⭐ Efficacy Analysis":
        show_efficacy_analysis(df)
    elif page == "😊 Sentiment Analysis":
        show_sentiment_analysis(df)
    elif page == "⚠️ Risk Assessment":
        show_risk_assessment(df)
    elif page == "📈 Comparative Analysis":
        show_comparative_analysis(df)

def show_overview(df, metadata):
    """Display overview dashboard"""
    st.markdown('<h2 class="sub-header">📊 Dataset Overview</h2>', unsafe_allow_html=True)
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Drugs", f"{df['drug_name'].nunique():,}")
    with col2:
        st.metric("Medical Conditions", f"{df['medical_condition'].nunique():,}")
    with col3:
        st.metric("Average Rating", f"{df['rating'].mean():.2f}/10")
    with col4:
        st.metric("Total Reviews", f"{df['no_of_reviews'].sum():,.0f}")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Rating Distribution")
        fig = px.histogram(df, x='rating', nbins=20, title="Drug Rating Distribution")
        fig.add_vline(x=df['rating'].mean(), line_dash="dash", line_color="red", 
                     annotation_text=f"Mean: {df['rating'].mean():.2f}")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🏥 Top Medical Conditions")
        top_conditions = df['medical_condition'].value_counts().head(10)
        fig = px.bar(x=top_conditions.values, y=top_conditions.index, orientation='h',
                    title="Top 10 Medical Conditions by Drug Count")
        fig.update_yaxes(title="Medical Condition")
        fig.update_xaxes(title="Number of Drugs")
        st.plotly_chart(fig, use_container_width=True)
    
    # Safety and efficacy overview
    if 'safety_score' in df.columns and 'efficacy_score' in df.columns:
        st.subheader("🛡️ Safety vs Efficacy Analysis")
        fig = px.scatter(df, x='safety_score', y='efficacy_score', 
                        color='severity_category' if 'severity_category' in df.columns else None,
                        title="Drug Safety vs Efficacy Scores",
                        hover_data=['drug_name', 'medical_condition'])
        fig.update_layout(xaxis_title="Safety Score (Higher = Safer)")
        fig.update_layout(yaxis_title="Efficacy Score (Higher = More Effective)")
        st.plotly_chart(fig, use_container_width=True)

def show_drug_search(df):
    """Drug search and details page"""
    st.markdown('<h2 class="sub-header">🔍 Drug Search & Analysis</h2>', unsafe_allow_html=True)
    
    # Search filters
    col1, col2 = st.columns(2)
    
    with col1:
        condition = st.selectbox(
            "Select Medical Condition",
            ["All"] + sorted(df['medical_condition'].unique())
        )
    
    with col2:
        min_rating = st.slider("Minimum Rating", 0.0, 10.0, 0.0, 0.5)
    
    # Filter data
    filtered_df = df.copy()
    if condition != "All":
        filtered_df = filtered_df[filtered_df['medical_condition'] == condition]
    filtered_df = filtered_df[filtered_df['rating'] >= min_rating]
    
    st.write(f"Found {len(filtered_df)} drugs matching your criteria")
    
    if len(filtered_df) > 0:
        # Drug selection
        drug_names = sorted(filtered_df['drug_name'].unique())
        selected_drug = st.selectbox("Select a drug for detailed analysis", drug_names)
        
        if selected_drug:
            drug_data = filtered_df[filtered_df['drug_name'] == selected_drug].iloc[0]
            
            # Drug details
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Rating", f"{drug_data['rating']:.1f}/10")
                st.metric("Number of Reviews", f"{drug_data['no_of_reviews']:,.0f}")
            
            with col2:
                if 'safety_score' in drug_data:
                    st.metric("Safety Score", f"{drug_data['safety_score']:.3f}")
                if 'efficacy_score' in drug_data:
                    st.metric("Efficacy Score", f"{drug_data['efficacy_score']:.3f}")
            
            with col3:
                st.metric("Prescription Type", drug_data['rx_otc'])
                if 'pregnancy_category' in drug_data:
                    st.metric("Pregnancy Category", drug_data['pregnancy_category'])
            
            # Side effects analysis
            if 'side_effects_cleaned' in drug_data:
                st.subheader("🔍 Side Effects Analysis")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Word cloud
                    if drug_data['side_effects_cleaned']:
                        wordcloud = WordCloud(width=400, height=300, background_color='white').generate(drug_data['side_effects_cleaned'])
                        fig, ax = plt.subplots(figsize=(8, 6))
                        ax.imshow(wordcloud, interpolation='bilinear')
                        ax.axis('off')
                        st.pyplot(fig)
                
                with col2:
                    # Severity metrics
                    if 'severity_score_avg' in drug_data:
                        st.metric("Average Severity", f"{drug_data['severity_score_avg']:.2f}/5")
                    if 'severe_effects_count' in drug_data:
                        st.metric("Severe Side Effects", int(drug_data['severe_effects_count']))
                    if 'sentiment_polarity' in drug_data:
                        sentiment = drug_data['sentiment_polarity']
                        sentiment_label = "Positive" if sentiment > 0.1 else "Negative" if sentiment < -0.1 else "Neutral"
                        st.metric("Sentiment", f"{sentiment:.3f} ({sentiment_label})")

def show_safety_analysis(df):
    """Safety analysis dashboard"""
    st.markdown('<h2 class="sub-header">🛡️ Drug Safety Analysis</h2>', unsafe_allow_html=True)
    
    # Safety overview
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if 'has_severe_effects' in df.columns:
            severe_count = df['has_severe_effects'].sum()
            st.metric("Drugs with Severe Effects", f"{severe_count} ({severe_count/len(df)*100:.1f}%)")
    
    with col2:
        if 'alcohol_warning' in df.columns:
            alcohol_count = df['alcohol_warning'].sum()
            st.metric("Alcohol Interaction Warnings", f"{alcohol_count} ({alcohol_count/len(df)*100:.1f}%)")
    
    with col3:
        if 'pregnancy_high_risk' in df.columns:
            pregnancy_count = df['pregnancy_high_risk'].sum()
            st.metric("High Pregnancy Risk", f"{pregnancy_count} ({pregnancy_count/len(df)*100:.1f}%)")
    
    # Safety score distribution
    if 'safety_score' in df.columns:
        st.subheader("📊 Safety Score Distribution")
        fig = px.histogram(df, x='safety_score', nbins=30, title="Distribution of Safety Scores")
        fig.add_vline(x=df['safety_score'].mean(), line_dash="dash", line_color="red",
                     annotation_text=f"Mean: {df['safety_score'].mean():.3f}")
        st.plotly_chart(fig, use_container_width=True)
    
    # Severity analysis
    if 'severity_category' in df.columns:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("⚡ Severity Category Distribution")
            severity_counts = df['severity_category'].value_counts()
            fig = px.pie(values=severity_counts.values, names=severity_counts.index,
                        title="Side Effect Severity Distribution")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("🏥 Severity by Medical Condition")
            severity_condition = pd.crosstab(df['medical_condition'], df['severity_category'])
            top_conditions = df['medical_condition'].value_counts().head(10).index
            severity_condition_top = severity_condition.loc[top_conditions]
            
            fig = px.bar(severity_condition_top, title="Severity Distribution by Top Conditions")
            st.plotly_chart(fig, use_container_width=True)

def show_efficacy_analysis(df):
    """Efficacy analysis dashboard"""
    st.markdown('<h2 class="sub-header">⭐ Drug Efficacy Analysis</h2>', unsafe_allow_html=True)
    
    # Efficacy overview
    col1, col2, col3 = st.columns(3)
    
    with col1:
        high_rated = (df['rating'] >= 8.0).sum()
        st.metric("High-Rated Drugs (≥8.0)", f"{high_rated} ({high_rated/len(df)*100:.1f}%)")
    
    with col2:
        if 'efficacy_score' in df.columns:
            st.metric("Average Efficacy Score", f"{df['efficacy_score'].mean():.3f}")
    
    with col3:
        avg_reviews = df['no_of_reviews'].mean()
        st.metric("Average Reviews per Drug", f"{avg_reviews:.0f}")
    
    # Rating analysis by condition
    st.subheader("📊 Rating Analysis by Medical Condition")
    
    # Select conditions to analyze
    all_conditions = sorted(df['medical_condition'].unique())
    selected_conditions = st.multiselect(
        "Select conditions to compare (max 10)",
        all_conditions,
        default=all_conditions[:5] if len(all_conditions) >= 5 else all_conditions
    )
    
    if selected_conditions:
        condition_ratings = df[df['medical_condition'].isin(selected_conditions)]
        
        fig = px.box(condition_ratings, x='medical_condition', y='rating',
                    title="Rating Distribution by Medical Condition")
        fig.update_layout(xaxis_tickangle=45)
        st.plotly_chart(fig, use_container_width=True)
        
        # Top drugs by condition
        st.subheader("🔝 Top Rated Drugs by Condition")
        
        selected_condition = st.selectbox("Select condition for top drugs", selected_conditions)
        
        condition_data = df[df['medical_condition'] == selected_condition]
        top_drugs = condition_data.nlargest(10, 'rating')[['drug_name', 'rating', 'no_of_reviews']]
        
        fig = px.bar(top_drugs, x='rating', y='drug_name', orientation='h',
                    title=f"Top 10 Drugs for {selected_condition}",
                    hover_data=['no_of_reviews'])
        st.plotly_chart(fig, use_container_width=True)

def show_sentiment_analysis(df):
    """Sentiment analysis dashboard"""
    st.markdown('<h2 class="sub-header">😊 Sentiment Analysis</h2>', unsafe_allow_html=True)
    
    if 'sentiment_category' in df.columns:
        # Sentiment overview
        col1, col2, col3 = st.columns(3)
        
        sentiment_counts = df['sentiment_category'].value_counts()
        
        with col1:
            positive_pct = (sentiment_counts.get('Positive', 0) / len(df)) * 100
            st.metric("Positive Sentiment", f"{positive_pct:.1f}%")
        
        with col2:
            neutral_pct = (sentiment_counts.get('Neutral', 0) / len(df)) * 100
            st.metric("Neutral Sentiment", f"{neutral_pct:.1f}%")
        
        with col3:
            negative_pct = (sentiment_counts.get('Negative', 0) / len(df)) * 100
            st.metric("Negative Sentiment", f"{negative_pct:.1f}%")
        
        # Sentiment distribution
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Overall Sentiment Distribution")
            fig = px.pie(values=sentiment_counts.values, names=sentiment_counts.index,
                        title="Sentiment Distribution",
                        color_discrete_map={
                            'Positive': 'green',
                            'Neutral': 'gray',
                            'Negative': 'red'
                        })
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("📈 Sentiment vs Rating")
            if 'sentiment_polarity' in df.columns:
                fig = px.scatter(df, x='sentiment_polarity', y='rating',
                               color='sentiment_category',
                               title="Sentiment Polarity vs Drug Rating",
                               hover_data=['drug_name'])
                st.plotly_chart(fig, use_container_width=True)
        
        # Sentiment by condition
        st.subheader("🏥 Sentiment Analysis by Medical Condition")
        
        condition_sentiment = pd.crosstab(df['medical_condition'], df['sentiment_category'], normalize='index') * 100
        top_conditions = df['medical_condition'].value_counts().head(10).index
        condition_sentiment_top = condition_sentiment.loc[top_conditions]
        
        fig = px.bar(condition_sentiment_top, title="Sentiment Distribution by Medical Condition (%)")
        st.plotly_chart(fig, use_container_width=True)

def show_risk_assessment(df):
    """Risk assessment dashboard"""
    st.markdown('<h2 class="sub-header">⚠️ Risk Assessment Dashboard</h2>', unsafe_allow_html=True)
    
    # Risk overview
    st.subheader("🚨 High-Risk Drug Categories")
    
    risk_metrics = {}
    
    if 'alcohol_warning' in df.columns:
        risk_metrics['Alcohol Interaction'] = df['alcohol_warning'].sum()
    
    if 'pregnancy_high_risk' in df.columns:
        risk_metrics['Pregnancy Risk (D/X)'] = df['pregnancy_high_risk'].sum()
    
    if 'is_controlled_substance' in df.columns:
        risk_metrics['Controlled Substances'] = df['is_controlled_substance'].sum()
    
    if 'has_severe_effects' in df.columns:
        risk_metrics['Severe Side Effects'] = df['has_severe_effects'].sum()
    
    # Display risk metrics
    cols = st.columns(len(risk_metrics))
    for i, (risk_type, count) in enumerate(risk_metrics.items()):
        with cols[i]:
            percentage = (count / len(df)) * 100
            st.metric(risk_type, f"{count} ({percentage:.1f}%)")
    
    # Risk heatmap
    if len(risk_metrics) > 1:
        st.subheader("🔥 Risk Correlation Matrix")
        
        risk_columns = []
        if 'alcohol_warning' in df.columns:
            risk_columns.append('alcohol_warning')
        if 'pregnancy_high_risk' in df.columns:
            risk_columns.append('pregnancy_high_risk')
        if 'is_controlled_substance' in df.columns:
            risk_columns.append('is_controlled_substance')
        if 'has_severe_effects' in df.columns:
            risk_columns.append('has_severe_effects')
        
        if len(risk_columns) > 1:
            risk_corr = df[risk_columns].corr()
            fig = px.imshow(risk_corr, title="Risk Factor Correlation Matrix",
                           color_continuous_scale='RdYlBu_r')
            st.plotly_chart(fig, use_container_width=True)
    
    # High-risk drugs list
    st.subheader("📋 High-Risk Drugs Identification")
    
    # Create composite risk score
    risk_score = 0
    if 'alcohol_warning' in df.columns:
        risk_score += df['alcohol_warning']
    if 'pregnancy_high_risk' in df.columns:
        risk_score += df['pregnancy_high_risk']
    if 'is_controlled_substance' in df.columns:
        risk_score += df['is_controlled_substance']
    if 'has_severe_effects' in df.columns:
        risk_score += df['has_severe_effects']
    
    df_risk = df.copy()
    df_risk['risk_score'] = risk_score
    
    high_risk_drugs = df_risk[df_risk['risk_score'] >= 2].nlargest(20, 'risk_score')
    
    if len(high_risk_drugs) > 0:
        st.write(f"Showing top {len(high_risk_drugs)} highest-risk drugs:")
        
        display_cols = ['drug_name', 'medical_condition', 'rating', 'risk_score']
        if 'alcohol_warning' in df.columns:
            display_cols.append('alcohol_warning')
        if 'pregnancy_high_risk' in df.columns:
            display_cols.append('pregnancy_high_risk')
        
        st.dataframe(high_risk_drugs[display_cols])

def show_comparative_analysis(df):
    """Comparative analysis dashboard"""
    st.markdown('<h2 class="sub-header">📈 Comparative Drug Analysis</h2>', unsafe_allow_html=True)
    
    # Drug comparison tool
    st.subheader("🔍 Compare Multiple Drugs")
    
    # Select drugs to compare
    all_drugs = sorted(df['drug_name'].unique())
    selected_drugs = st.multiselect(
        "Select drugs to compare (2-5 drugs)",
        all_drugs,
        default=all_drugs[:3] if len(all_drugs) >= 3 else all_drugs
    )
    
    if len(selected_drugs) >= 2:
        comparison_data = df[df['drug_name'].isin(selected_drugs)]
        
        # Comparison metrics
        st.subheader("📊 Key Metrics Comparison")
        
        metrics_to_compare = ['rating', 'no_of_reviews']
        if 'safety_score' in df.columns:
            metrics_to_compare.append('safety_score')
        if 'efficacy_score' in df.columns:
            metrics_to_compare.append('efficacy_score')
        if 'severity_score_avg' in df.columns:
            metrics_to_compare.append('severity_score_avg')
        
        # Create comparison chart
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Rating Comparison', 'Reviews Comparison', 
                           'Safety vs Efficacy', 'Risk Profile'],
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Rating comparison
        for drug in selected_drugs:
            drug_data = comparison_data[comparison_data['drug_name'] == drug]
            if len(drug_data) > 0:
                fig.add_trace(
                    go.Bar(name=drug, x=[drug], y=[drug_data['rating'].iloc[0]]),
                    row=1, col=1
                )
        
        # Reviews comparison
        for drug in selected_drugs:
            drug_data = comparison_data[comparison_data['drug_name'] == drug]
            if len(drug_data) > 0:
                fig.add_trace(
                    go.Bar(name=drug, x=[drug], y=[drug_data['no_of_reviews'].iloc[0]], showlegend=False),
                    row=1, col=2
                )
        
        # Safety vs Efficacy
        if 'safety_score' in df.columns and 'efficacy_score' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=comparison_data['safety_score'],
                    y=comparison_data['efficacy_score'],
                    mode='markers+text',
                    text=comparison_data['drug_name'],
                    textposition="top center",
                    showlegend=False
                ),
                row=2, col=1
            )
        
        fig.update_layout(height=800, title_text="Drug Comparison Dashboard")
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed comparison table
        st.subheader("📋 Detailed Comparison Table")
        
        comparison_summary = []
        for drug in selected_drugs:
            drug_data = comparison_data[comparison_data['drug_name'] == drug].iloc[0]
            
            summary = {
                'Drug Name': drug,
                'Medical Condition': drug_data['medical_condition'],
                'Rating': f"{drug_data['rating']:.1f}/10",
                'Reviews': f"{drug_data['no_of_reviews']:,.0f}",
                'Prescription Type': drug_data['rx_otc']
            }
            
            if 'safety_score' in drug_data:
                summary['Safety Score'] = f"{drug_data['safety_score']:.3f}"
            if 'efficacy_score' in drug_data:
                summary['Efficacy Score'] = f"{drug_data['efficacy_score']:.3f}"
            if 'pregnancy_category' in drug_data:
                summary['Pregnancy Category'] = drug_data['pregnancy_category']
            
            comparison_summary.append(summary)
        
        comparison_df = pd.DataFrame(comparison_summary)
        st.dataframe(comparison_df, use_container_width=True)

if __name__ == "__main__":
    main() 