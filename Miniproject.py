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
    st.markdown("Urban populations continue to rise, the expansion of supermarkets in densely populated cities has become increasingly competitive. With a growing demand for health and wellness products, understanding market dynamics is crucial. In this dashboard we'll give it a try and turn everything into a readable visualizations.")

# The Selectbox
    #Category = Supplement_Sales['Category'].unique()
    #line = st.selectbox('',['Choose the Category'] + list(Category))
    #if line == 'Choose the Category':
        #chosen_line = Supplement_Sales
    #else:
        #chosen_line = Supplement_Sales[Supplement_Sales['Category'] == line]

    # Customizing the select box
    #st.markdown(f'''
    #<style>
        #.stSelectbox div div {{
                #background-color: #fafafa;
               # color: #333;
        #}}
        #.stSelectbox div div:hover {{
                #cursor: pointer
        #}}
        #.stSelectbox div div .option {{
                #background-color: red;
               # color: #111;
        #}}
       # .stSelectbox div div svg {{
               # fill: black;
       # }}
    #</style>
    #''', unsafe_allow_html=True)
    

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

#Product Data View
with col10:
    with st.expander("Total Units Sold Data View"):
        grouped_df = filtered_df.groupby(by=["Product Name"], as_index=False)["Units Sold"].sum()
        st.write(grouped_df.style.background_gradient(cmap="Greens"))

#Location Data View
with col11:
    with st.expander("Total Units Returned Data View"):
        grouped_df = filtered_df.groupby(by=["Product Name"], as_index=False)["Units Returned"].sum()
        st.write(grouped_df.style.background_gradient(cmap="Reds"))


#Create dashboard section
#with chartRow:
    # Filter for the month
    #Supplement_Sales['Date'] = pd.to_datetime(Supplement_Sales['Date'])
    #mar_data = (Supplement_Sales['Date'].dt.month == 3)
    #lineQuantity = chosen_line[(mar_data)]

    # Quantity for each day
    #Units_Sold_per_year = lineQuantity.groupby('Date')['Units Sold'].sum().reset_index()

    # some space
    #st.markdown('<div></div>', unsafe_allow_html=True)

    # Create a line chart for Quantity over the last month using Plotly
    #fig_Units_Sold = px.line(
        #Units_Sold_per_year, 
        #x='Date', 
       # y='Units Sold', 
        #title='Unit Sold over the Year'
    #)
    #fig_Units_Sold.update_layout(
        #margin_r=100,
    #)
    #st.plotly_chart(fig_Units_Sold)



    
# creating a single-element container.
#placeholder = st.empty()
#dataframe filter 




       
        

#upload data
#upload_file = st.file_uploader("Please upload here:", type = 'csv')


#df = pd.read_csv(r"C:\Users\welcome\Desktop\BSMS1306\streamlit\Tips.csv")
#df = pd.read_csv("Supplement_Sales.csv")
#df = pd.read_csv(upload_file)

#show data nk tunjuk pun boleh x nk pun boleh
#st.subheader("Raw Data")
#st.write(df)

#histogram
#st.subheader("Histogram")
#column = st.selectbox("Choose a column",df.columns)
#fig, ax = plt.subplots(figsize = (10,6))
#df[column].plot(kind = 'hist', ax =ax)
#st.pyplot(fig)
#yg kaler purple
#fig = px.histogram(df, x=column)
#fig.update_traces( marker = {"color":"purple", "line":{"color":"black","width":2}})
#st.plotly_chart(fig)

#Scatter chart
#st.subheader("Scatter Chart")
#x_column = st.selectbox("Choose x-axis column",df.columns)
#y_column = st.selectbox("Choose y-axis column",df.columns)
#fig, ax = plt.subplots(figsize = (10,6))
#df.plot(kind = 'scatter', x=x_column, y=y_column, ax =ax)
#st.pyplot(fig)

#fig = px.scatter(df, x=x_column, y = y_column,color ='sex' , color_discrete_sequence= ['yellow', 'red'])
#st.plotly_chart(fig)

