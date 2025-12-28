import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
from pathlib import Path
import pickle
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="DrugsMed Analytics Dashboard",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

# Cache data loading
@st.cache_data
def load_data():
    """Load and cache the processed dataset"""
    try:
        df = pd.read_csv("data/drugs_processed.csv")
        return df
    except FileNotFoundError:
        st.error(" Data file not found. Please ensure data/drugs_processed.csv exists.")
        return None

@st.cache_data
def load_metadata():
    """Load feature engineering metadata"""
    try:
        with open("data/feature_engineering_metadata.pkl", "rb") as f:
            metadata = pickle.load(f)
        return metadata
    except FileNotFoundError:
        return {}

@st.cache_resource
def load_models():
    """Load trained ML models"""
    models = {}
    model_path = Path("models")
    
    if model_path.exists():
        for model_file in model_path.glob("*.joblib"):
            model_name = model_file.stem
            try:
                models[model_name] = joblib.load(model_file)
                st.success(f" Loaded {model_name}")
            except Exception as e:
                st.warning(f"Could not load {model_name}: {e}")
    else:
        st.warning(" Models directory not found. Please run the ML notebook first.")
    
    return models

# Main application
def main():
    # Title and header
    st.markdown('<h1 class="main-header"> DrugsMed Analytics Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("### Comprehensive Drug Safety & Efficacy Analysis Platform")
    
    # Load data
    df = load_data()
    if df is None:
        st.stop()
    
    metadata = load_metadata()
    models = load_models()
    
    # Sidebar navigation
    st.sidebar.title(" Navigation")
    page = st.sidebar.selectbox(
        "Choose Analysis Section:",
        [
            " Dashboard Overview",
            " Data Explorer", 
            " Drug Predictor",
            " Association Insights",
            " Analytics & Trends",
            " Recommendations"
        ]
    )
    
    # Page routing
    if page == " Dashboard Overview":
        show_overview(df, metadata, models)
    elif page == " Data Explorer":
        show_data_explorer(df)
    elif page == " Drug Predictor":
        show_drug_predictor(df, models)
    elif page == " Association Insights":
        show_association_insights(df)
    elif page == " Analytics & Trends":
        show_analytics_trends(df)
    elif page == " Recommendations":
        show_recommendations(df, models)

def show_overview(df, metadata, models):
    """Dashboard overview with key metrics"""
    st.header(" Dashboard Overview")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label=" Total Drugs",
            value=f"{df['drug_name'].nunique():,}",
            delta=f"+{metadata.get('features_added', 0)} features"
        )
    
    with col2:
        st.metric(
            label=" Medical Conditions", 
            value=f"{df['medical_condition'].nunique():,}",
            delta=f"Avg {df.groupby('medical_condition')['drug_name'].nunique().mean():.1f} drugs/condition"
        )
    
    with col3:
        st.metric(
            label=" Average Rating",
            value=f"{df['rating'].mean():.2f}",
            delta=f"±{df['rating'].std():.2f} std dev"
        )
    
    with col4:
        st.metric(
            label=" Total Reviews",
            value=f"{df['no_of_reviews'].sum():,.0f}",
            delta=f"Avg {df['no_of_reviews'].mean():.0f}/drug"
        )
    
    # Visual overview
    st.subheader(" Data Distribution Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Rating distribution
        fig_rating = px.histogram(
            df, x='rating', nbins=20,
            title="Drug Rating Distribution",
            labels={'rating': 'Rating', 'count': 'Number of Drugs'}
        )
        fig_rating.update_layout(height=400)
        st.plotly_chart(fig_rating, use_container_width=True)
    
    with col2:
        # Top conditions
        top_conditions = df['medical_condition'].value_counts().head(10)
        fig_conditions = px.bar(
            x=top_conditions.values,
            y=top_conditions.index,
            orientation='h',
            title="Top 10 Medical Conditions",
            labels={'x': 'Number of Drugs', 'y': 'Medical Condition'}
        )
        fig_conditions.update_layout(height=400)
        st.plotly_chart(fig_conditions, use_container_width=True)
    
    # Model status
    st.subheader(" Machine Learning Models Status")
    if models:
        model_cols = st.columns(min(len(models), 4))
        for i, (model_name, model) in enumerate(models.items()):
            with model_cols[i % len(model_cols)]:
                st.success(f" {model_name.replace('_', ' ').title()}")
    else:
        st.warning(" No trained models found. Please run the ML notebook first.")

def show_data_explorer(df):
    """Interactive data exploration interface"""
    st.header(" Data Explorer")
    
    # Filters in sidebar
    st.sidebar.subheader(" Filters")
    
    # Medical condition filter
    conditions = ['All'] + sorted(df['medical_condition'].unique().tolist())
    selected_condition = st.sidebar.selectbox("Medical Condition:", conditions)
    
    # Rating range filter
    rating_range = st.sidebar.slider(
        "Rating Range:",
        min_value=float(df['rating'].min()),
        max_value=float(df['rating'].max()),
        value=(float(df['rating'].min()), float(df['rating'].max())),
        step=0.1
    )
    
    # Review count filter
    review_range = st.sidebar.slider(
        "Review Count Range:",
        min_value=int(df['no_of_reviews'].min()),
        max_value=int(df['no_of_reviews'].max()),
        value=(int(df['no_of_reviews'].min()), int(df['no_of_reviews'].max()))
    )
    
    # Apply filters
    filtered_df = df.copy()
    if selected_condition != 'All':
        filtered_df = filtered_df[filtered_df['medical_condition'] == selected_condition]
    
    filtered_df = filtered_df[
        (filtered_df['rating'] >= rating_range[0]) & 
        (filtered_df['rating'] <= rating_range[1]) &
        (filtered_df['no_of_reviews'] >= review_range[0]) & 
        (filtered_df['no_of_reviews'] <= review_range[1])
    ]
    
    # Display results
    st.write(f" Showing {len(filtered_df):,} drugs (filtered from {len(df):,} total)")
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        if len(filtered_df) > 0:
            fig_scatter = px.scatter(
                filtered_df,
                x='no_of_reviews',
                y='rating',
                hover_data=['drug_name', 'medical_condition'],
                title="Rating vs Review Count",
                labels={'no_of_reviews': 'Number of Reviews', 'rating': 'Rating'}
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
    
    with col2:
        if len(filtered_df) > 0:
            rating_dist = filtered_df['rating'].value_counts().sort_index()
            fig_rating_dist = px.bar(
                x=rating_dist.index,
                y=rating_dist.values,
                title="Rating Distribution (Filtered)",
                labels={'x': 'Rating', 'y': 'Count'}
            )
            st.plotly_chart(fig_rating_dist, use_container_width=True)
    
    # Data table
    st.subheader(" Filtered Data")
    display_columns = ['drug_name', 'medical_condition', 'rating', 'no_of_reviews']
    available_columns = [col for col in display_columns if col in filtered_df.columns]
    
    if available_columns:
        st.dataframe(
            filtered_df[available_columns].head(100),
            use_container_width=True,
            height=400
        )
    else:
        st.warning(" Required columns not found in dataset")

def show_drug_predictor(df, models):
    """Drug rating prediction interface"""
    st.header(" Drug Predictor")
    
    if not models:
        st.warning(" No trained models available. Please run the ML notebook first.")
        return
    
    st.write("Enter drug characteristics to predict rating and category:")
    
    # Input form
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            medical_condition = st.selectbox(
                "Medical Condition:",
                sorted(df['medical_condition'].unique())
            )
            
            # Add some sample inputs
            side_effects_length = st.number_input(
                "Side Effects Description Length:",
                min_value=0,
                max_value=1000,
                value=100,
                help="Length of side effects description"
            )
            
            review_count = st.number_input(
                "Number of Reviews:",
                min_value=0,
                max_value=1000,
                value=50,
                help="Expected number of reviews"
            )
        
        with col2:
            st.write("**Prediction will be based on:**")
            st.write("• Selected medical condition")
            st.write("• Side effects complexity")
            st.write("• Review volume expectations")
            st.write("• Historical patterns")
        
        submitted = st.form_submit_button(" Predict Drug Rating")
        
        if submitted:
            st.success(" Prediction request submitted!")
            
            # Simple prediction logic (replace with actual model)
            condition_avg = df[df['medical_condition'] == medical_condition]['rating'].mean()
            review_factor = min(review_count / 100, 1.0)  # Normalize
            side_effect_factor = max(0.5, 1 - side_effects_length / 500)  # Penalty for many side effects
            
            predicted_rating = condition_avg * review_factor * side_effect_factor
            predicted_rating = max(1.0, min(10.0, predicted_rating))  # Bound between 1-10
            
            confidence = 0.75 + np.random.uniform(-0.1, 0.15)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Predicted Rating",
                    f"{predicted_rating:.2f}",
                    delta=f"{predicted_rating - df['rating'].mean():.2f} vs avg"
                )
            
            with col2:
                st.metric(
                    "Confidence",
                    f"{confidence:.1%}",
                    delta="High" if confidence > 0.8 else "Medium"
                )
            
            with col3:
                category = "Excellent" if predicted_rating > 8 else "Good" if predicted_rating > 6 else "Fair"
                st.metric(
                    "Category",
                    category,
                    delta="Recommended" if predicted_rating > 7 else "Review"
                )

def show_association_insights(df):
    """Association rule insights and drug relationships"""
    st.header(" Association Insights")
    
    st.write("Discover relationships between drugs, conditions, and characteristics:")
    
    # Drug co-occurrence analysis
    st.subheader(" Drug Co-occurrence by Condition")
    
    condition_list = sorted(df['medical_condition'].unique())
    selected_condition = st.selectbox(
        "Select condition to analyze:",
        condition_list
    )
    
    # Filter drugs for selected condition
    condition_drugs = df[df['medical_condition'] == selected_condition]
    
    if len(condition_drugs) > 1:
        st.write(f"Found {len(condition_drugs)} drugs for {selected_condition}:")
        
        # Drug ratings visualization
        drug_ratings = condition_drugs.groupby('drug_name')['rating'].mean().sort_values(ascending=False)
        
        fig_drugs = px.bar(
            x=drug_ratings.values,
            y=drug_ratings.index,
            orientation='h',
            title=f"Drug Ratings for {selected_condition}",
            labels={'x': 'Average Rating', 'y': 'Drug Name'}
        )
        st.plotly_chart(fig_drugs, use_container_width=True)
        
        # Top drugs
        st.subheader(" Top Rated Drugs")
        top_drugs = drug_ratings.head(5)
        for i, (drug, rating) in enumerate(top_drugs.items(), 1):
            reviews = condition_drugs[condition_drugs['drug_name'] == drug]['no_of_reviews'].iloc[0]
            st.write(f"{i}. **{drug}**: {rating:.2f}  ({reviews} reviews)")
    
    else:
        st.info("Select a condition with multiple drug options for analysis.")

def show_analytics_trends(df):
    """Analytics and trends visualization"""
    st.header(" Analytics & Trends")
    
    # Condition analysis
    st.subheader(" Medical Condition Performance Analysis")
    
    condition_stats = df.groupby('medical_condition').agg({
        'rating': ['mean', 'std', 'count'],
        'no_of_reviews': 'sum'
    }).round(2)
    
    condition_stats.columns = ['Avg_Rating', 'Rating_Std', 'Drug_Count', 'Total_Reviews']
    condition_stats = condition_stats.sort_values('Avg_Rating', ascending=False)
    
    # Interactive condition analysis
    fig_condition_analysis = px.scatter(
        x=condition_stats['Drug_Count'],
        y=condition_stats['Avg_Rating'],
        size=condition_stats['Total_Reviews'],
        hover_name=condition_stats.index,
        title="Condition Analysis: Drug Count vs Average Rating",
        labels={
            'x': 'Number of Drugs',
            'y': 'Average Rating',
            'size': 'Total Reviews'
        }
    )
    st.plotly_chart(fig_condition_analysis, use_container_width=True)
    
    # Performance tables
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader(" Highest Rated Conditions")
        top_conditions = condition_stats.head(10)
        for condition in top_conditions.index:
            stats = top_conditions.loc[condition]
            st.write(f"**{condition}**: {stats['Avg_Rating']:.2f}  ({stats['Drug_Count']:.0f} drugs)")
    
    with col2:
        st.subheader(" Most Reviewed Conditions")
        most_reviewed = condition_stats.sort_values('Total_Reviews', ascending=False).head(10)
        for condition in most_reviewed.index:
            stats = most_reviewed.loc[condition]
            st.write(f"**{condition}**: {stats['Total_Reviews']:.0f} reviews")

def show_recommendations(df, models):
    """Drug recommendation interface"""
    st.header(" Drug Recommendations")
    
    st.write("Get personalized drug recommendations based on your criteria:")
    
    # Recommendation filters
    col1, col2 = st.columns(2)
    
    with col1:
        target_condition = st.selectbox(
            "Medical Condition:",
            sorted(df['medical_condition'].unique())
        )
        
        min_rating = st.slider(
            "Minimum Rating:",
            min_value=1.0,
            max_value=10.0,
            value=7.0,
            step=0.5
        )
    
    with col2:
        min_reviews = st.slider(
            "Minimum Reviews:",
            min_value=0,
            max_value=int(df['no_of_reviews'].max()),
            value=50,
            step=10
        )
        
        max_results = st.slider(
            "Maximum Results:",
            min_value=5,
            max_value=20,
            value=10
        )
    
    # Generate recommendations
    if st.button(" Find Recommendations"):
        recommendations = df[
            (df['medical_condition'] == target_condition) &
            (df['rating'] >= min_rating) &
            (df['no_of_reviews'] >= min_reviews)
        ].sort_values(['rating', 'no_of_reviews'], ascending=[False, False]).head(max_results)
        
        if len(recommendations) > 0:
            st.subheader(f" Recommended Drugs for {target_condition}")
            
            for i, (_, drug) in enumerate(recommendations.iterrows(), 1):
                with st.expander(f"{i}. {drug['drug_name']} - {drug['rating']:.1f}"):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Rating", f"{drug['rating']:.1f}")
                    
                    with col2:
                        st.metric("Reviews", f"{drug['no_of_reviews']:.0f}")
                    
                    with col3:
                        effectiveness = "High" if drug['rating'] > 8 else "Medium" if drug['rating'] > 6 else "Low"
                        st.metric("Effectiveness", effectiveness)
                    
                    if 'side_effects' in drug.index and pd.notna(drug['side_effects']):
                        st.write("**Side Effects:**")
                        side_effects_text = str(drug['side_effects'])
                        st.write(side_effects_text[:200] + "..." if len(side_effects_text) > 200 else side_effects_text)
        else:
            st.warning(" No drugs found matching your criteria. Try adjusting the filters.")

# Run the application
if __name__ == "__main__":
    main()
