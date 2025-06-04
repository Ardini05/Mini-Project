import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
import seaborn as sns
import plotly.figure_factory as ff
import Process 

# read csv
df = pd.read_csv("Supplement_Sales.csv")
df['Category'].unique()
Supplement_Sales = pd.read_csv('Supplement_Sales.csv')

# Creating time features
df = Process.fetch_time_features(df)

#st.image(r'C:\Users\welcome\Desktop\BSMS1306\streamlit\Header.png')
st.image('Header.jpg')

#st.date_input("Select a date")
st.title("""Supplement Sales Dashboard""")

#Data view
#st.subheader("Detailed Data view")
#st.write(df)

# the layout Variables
hero = st.container()
topRow = st.container()
midRow = st.container()
chartRow = st.container()
footer = st.container()

# Sidebar
with st.sidebar:
    st.markdown(f'''
        <style>
        section[data-testid="stSidebar"] {{
                width: 500px;
                background-color: #000b1a;
                }}
        section[data-testid="stSidebar"] h1 {{
                color: #e3eefc;
                }}
        section[data-testid="stSidebar"] p {{
                color: #ddd;
                text-align: left;
                }}
        section[data-testid="stSidebar"] svg {{
                fill: #ddd;
                }}
        </style>
    ''',unsafe_allow_html=True)
    st.title(":anchor: About the dataset")
    st.markdown("In the highly competitive market for health and well-being, it is crucial to understand how the consumer makes their purchases in a bid to optimize business performance. With a growing demand for health and wellness products, understanding market dynamics is crucial. In this dashboard we'll give it a try and turn everything into a readable visualizations.")



# side bar for filters
st.sidebar.title('Filters')

# Filters
selected_platform = Process.multiselect('Select Platform', df['Platform'].unique())
selected_category = Process.multiselect('Select Category', df['Category'].unique()) 
selected_year = Process.multiselect('Select Year', df['Sale_Year'].unique())
selected_month = Process.multiselect('Select  Month', df['Sale_Month'].unique()) 

filtered_df = df[(df['Platform'].isin(selected_platform)) &
                (df['Category'].isin(selected_category)) &
                (df['Sale_Year'].isin(selected_year)) &
                (df['Sale_Month'].isin(selected_month))]



# KPI - Key Performance Indicator
# Create columns for displaying KPIs
col1,col2,col3, = st.columns(3)

# Total Sales
with col1:
    st.metric(label='Total Sales', value = f"$ {float(filtered_df['Revenue'].sum())}")

# Total Units Sold
with col2:
      st.metric(label='Total Units Sold', value = f"{int(filtered_df['Units Sold'].sum())}")

# Total Returned
with col2:
     st.metric(label='Total Units Returned', value = f"{int(filtered_df['Units Returned'].sum())}")

# Visualization to analyze month-on-month sales trend
Yearly_sales = (filtered_df[['Sale_Year', 'Sale_Month', 'Revenue']]
               .groupby(['Sale_Year', 'Sale_Month'])
               .sum()
               .reset_index()
               .pivot(index = 'Sale_Month', columns = 'Sale_Year', values = 'Revenue'))

st.line_chart(Yearly_sales, x_label = "Sale_Month", y_label = 'Total Sales')


# Create columns for displaying wise Sales
col4, col5 = st.columns(2)
 
# Product-wise Sales
with col4:
        st.subheader("Product-wise Sales")
        fig = px.bar(
            filtered_df.groupby(by=["Product Name"], as_index=False)["Revenue"].sum(),
            x="Product Name",
            y="Revenue",
            template="seaborn",
        )
        st.plotly_chart(fig, use_container_width=True)

#Location-wise Sales
with col5:
        st.subheader("Location-wise Sales")
        fig = px.pie(
            filtered_df, 
            values="Revenue", 
            names="Location", 
            hole=0.5
        )
        st.plotly_chart(fig, use_container_width=True)


# Create columns for specific Data view
col6, col7 = st.columns(2)

#Product Data View
with col6:
    with st.expander("Product Data View"):
        grouped_df = filtered_df.groupby(by=["Product Name"], as_index=False)["Revenue"].sum()
        st.write(grouped_df.style.background_gradient(cmap="Blues"))

#Location Data View
with col7:
    with st.expander("Location Data View"):
        grouped_df = filtered_df.groupby(by=["Location"], as_index=False)["Revenue"].sum()
        st.write(grouped_df.style.background_gradient(cmap="Oranges"))



# Create a treemap based on Location, Category and Procut Name
st.subheader("Hierarchical View of Sales Using Treemap")
# Ensure required columns exist to avoid runtime errors
required_cols = ['Location', 'Category', 'Product Name','Revenue']
if all(col in filtered_df.columns for col in required_cols):
    fig= px.treemap(
        filtered_df,
        path=["Location", "Category", "Product Name"],
        values="Revenue",
        hover_data=["Revenue"],
        color="Product Name"
    )
    fig.update_layout(width=800, height=650)
    st.plotly_chart(fig, use_container_width=True, key="region_treemap")
else:
    st.warning("Treemap can't be displayed: Required columns not found in the filtered dataset.")



# Create columns for specific Units Sold and Returned
col8, col9 = st.columns(2)

# Units Sold
with col8:
        st.subheader("Total Units Sold ")
        fig = px.bar(
            filtered_df.groupby(by=["Product Name"], as_index=False)["Units Sold"].sum(),
            x="Product Name",
            y="Units Sold",
            template="seaborn",
            color_discrete_sequence=['Green']
        )
        st.plotly_chart(fig, use_container_width=True)

# Units Returned
with col9:
        st.subheader("Total Units Returned ")
        fig = px.bar(
            filtered_df.groupby(by=["Product Name"], as_index=False)["Units Returned"].sum(),
            x="Product Name",
            y="Units Returned",
            template="seaborn",
            color_discrete_sequence=['Red']
        )
        st.plotly_chart(fig, use_container_width=True)

# Create columns for specific Data view
col10, col11 = st.columns(2)

#Units Sold Data View
with col10:
    with st.expander("Total Units Sold Data View"):
        grouped_df = filtered_df.groupby(by=["Product Name"], as_index=False)["Units Sold"].sum()
        st.write(grouped_df.style.background_gradient(cmap="Greens"))

#Units Returned Data View
with col11:
    with st.expander("Total Units Returned Data View"):
        grouped_df = filtered_df.groupby(by=["Product Name"], as_index=False)["Units Returned"].sum()
        st.write(grouped_df.style.background_gradient(cmap="Reds"))


