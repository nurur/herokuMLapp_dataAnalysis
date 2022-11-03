import pandas as pd
import numpy as np
import os, sys
import re
import sklearn
import plotly.express as px
import plotly.graph_objects as go

import streamlit as st



import streamlit as st
#st.set_page_config(layout = "wide")
pd.set_option("display.precision", 2)
pd.set_option("display.float_format", lambda x: "%.2f" % x)
pd.set_option('display.max_colwidth', None)



@st.cache(allow_output_mutation=False)
def get_data() :
    from sklearn.datasets import load_iris
    df = load_iris(as_frame=True)
    X  = df.data
    y  = df.target

    oldColNames = df.feature_names 
    newColNames = [x.split('(cm)')[0].strip().title() for x in oldColNames]
    nameDict = {oldColNames[i] : newColNames[i] for i in range(len(oldColNames))}
    X = X.rename( columns=nameDict, inplace=False)
    Y = pd.DataFrame( y.rename('Labels') )

    data = pd.concat([X,Y], axis=1)

    unique_labels  = data.Labels.unique().tolist() 
    unique_species = df.target_names
    mapper_species = { unique_labels[x] : unique_species[x] for x in range(len(unique_labels)) }
    data['Species'] = data['Labels'].apply( lambda  row : mapper_species[row] )
    
    return data



def show_scatterPlot(data):
    """
    input:
    iris dataset

    output:
    a scatter plot between two features
    """

    st.subheader("Scatter Plot")
    feature_x = st.selectbox("Select Feature along X-axis", data.columns[0:4],0)
    feature_y = st.selectbox("Select Feature along Y-axis", data.columns[0:4],1)


    color_discrete_map = {
    'setosa': '#440154', 
    'versicolor': '#1f9e89',
    'virginica': '#fde725'
    }

    fig = px.scatter(
        data,
        x=data[feature_x], y=data[feature_y], 
        color="Species",
        color_discrete_map = color_discrete_map
    )

    fig.update_traces(marker_size=8)

    fig.update_xaxes(
        title_text = feature_x, 
        title_font = dict(size=16), 
        tickfont_size=14, 
        range = None, 
        showticklabels= True,
        showgrid=True
        )  
    fig.update_yaxes(
        title_text=feature_y, 
        title_font = dict(size=16),
        tickfont_size=14, 
        range = None,
        showticklabels= True,
        showgrid=True
        )  

    fig.update_layout(
        showlegend=True,
        legend=dict(title=None, 
            orientation="h",
            yanchor="top", y=0.98, 
            xanchor="left", x=0.02
            ),
        width=900, height=400,
        margin=dict(l=180,r=150,t=0,b=0))

    st.plotly_chart(fig)



def show_boxPlot(data):
    """
    input:
    iris dataset

    output:
    a box plot of given feature 
    """

    st.subheader("Box Plot")
    feature = st.selectbox("Select a Feature for Box Plot", data.columns[0:4])
    
    st.write(" ")

    color_discrete_map = {
    'setosa': '#440154', 
    'versicolor': '#1f9e89',
    'virginica': '#fde725'
    }

    fig = px.box(
        data, 
        x="Species", 
        y= feature, 
        points="all", 
        color="Species", 
        notched=True,
        color_discrete_map = color_discrete_map
        )

    fig.update_traces(marker_size=5)

    fig.update_layout(
        showlegend=True,
        legend=dict(
            title=None,
            orientation="h", 
            yanchor="top", y=0.98, 
            xanchor="left", x= 0.025,
            bgcolor=None
            ),
        width=900, height=400,
        margin=dict(l=180,r=150,t=0,b=0),
        xaxis=dict(title="Species", 
            title_font=dict(size=16), tickfont=dict(size=14)), 
        yaxis=dict(title=feature, 
            title_font=dict(size=16), tickfont=dict(size=14)) 
        )

    st.plotly_chart(fig)



def show_histogramPlot(data):
    """
    input:
    iris dataset

    output:
    a histogram plot of given feature for each class
    """

    st.subheader("Histogram Plot")
    feature = st.selectbox("Select a Feature", data.columns[0:4])

    select_func = st.selectbox("Select Bin Type", 
        ["Count", "Sum", "Average"], 0)

    select_norm = st.selectbox("Select Histogram Normalization Type", 
        ["Number", "Probability", "Percent", "Density", "Probability Density"], 0)

    st.markdown(
        """
        1. If 'number', a given bin is not normalized. 
        2. If 'probability', a given bin is divided by the sum of all bins. 
        3. If 'percent', the output of a given bin is divided by the sum of all bins and multiplied by 100. 
        4. If 'density', the output of a given bin is divided by the size of the bin. 
        5. If 'probability density', the output a given bin is normalized such that it corresponds to the probability that a random event whose distribution is described by the bin type will fall into that bin.
        """
        )

    st.write(" ")

    color_discrete_map = {
    'setosa': '#440154', 
    'versicolor': '#1f9e89',
    'virginica': '#fde725'
    }

    if select_func == "Count":
        if select_norm == "Number":
            fig = px.histogram(
                data, x=feature, 
                histfunc= "count",
                histnorm = None, 
                opacity=0.50, 
                color="Species", 
                color_discrete_map = color_discrete_map,
                marginal='violin')
        elif select_norm == "Probability":
            fig = px.histogram(
                data, x=feature, 
                histfunc= "count",
                histnorm = "probability", 
                opacity=0.50, 
                color="Species", 
                color_discrete_map = color_discrete_map,
                marginal='violin')

        elif select_norm == "Percent":
            fig = px.histogram(
                data, x=feature, 
                histfunc= "count",
                histnorm = "percent",
                opacity=0.50, 
                color="Species", 
                color_discrete_map = color_discrete_map,
                marginal='violin')

        elif select_norm == "Density":
            fig = px.histogram(
                data, x=feature, 
                histfunc= "count",
                histnorm = "density",
                opacity=0.50, 
                color="Species", 
                color_discrete_map = color_discrete_map,
                marginal='violin')

        else:
            fig = px.histogram(
                data, x=feature, 
                histfunc= "count",
                histnorm = "probability density",
                opacity=0.50, 
                color="Species", 
                color_discrete_map = color_discrete_map,
                marginal='violin')

    elif select_func == "Sum":
        if select_norm == "Number":
            fig = px.histogram(
                data, x=feature, 
                histfunc= "sum",
                histnorm = None, 
                opacity=0.50, 
                color="Species", 
                color_discrete_map = color_discrete_map,
                marginal='violin')

        elif select_norm == "Probability":
            fig = px.histogram(
                data, x=feature, 
                histfunc= "sum",
                histnorm = "probability", 
                opacity=0.50, 
                color="Species", 
                color_discrete_map = color_discrete_map,  
                marginal='violin')

        elif select_norm == "Percent":
            fig = px.histogram(
                data, x=feature, 
                histfunc= "sum",
                histnorm = "percent",
                opacity=0.50, 
                color="Species", 
                color_discrete_map = color_discrete_map,
                marginal='violin')

        elif select_norm == "Density":
            fig = px.histogram(
                data, x=feature, 
                histfunc= "sum",
                histnorm = "density",
                opacity=0.50, 
                color="Species", 
                color_discrete_map = color_discrete_map,  
                marginal='violin')

        else:
            fig = px.histogram(
                data, x=feature, 
                histfunc= "sum",
                histnorm = "probability density",
                opacity=0.50, 
                color="Species", 
                color_discrete_map = color_discrete_map,
                marginal='violin')
    else:
        if select_norm == "Number":
            fig = px.histogram(
                data, x=feature, 
                histfunc= "avg",
                histnorm = None, 
                opacity=0.50, 
                color="Species", 
                color_discrete_map = color_discrete_map,
                marginal='violin')
        if select_norm == "Probability":
            fig = px.histogram(
                data, x=feature, 
                histfunc= "avg",
                histnorm = "probability", 
                opacity=0.50, 
                color="Species", 
                color_discrete_map = color_discrete_map,
                marginal='violin')

        elif select_norm == "Percent":
            fig = px.histogram(
                data, x=feature, 
                histfunc= "avg",
                histnorm = "percent",
                opacity=0.50, 
                color="Species", 
                color_discrete_map = color_discrete_map,
                marginal='violin')

        elif select_norm == "Density":
            fig = px.histogram(
                data, x=feature, 
                histfunc= "avg",
                histnorm = "density",
                opacity=0.50, 
                color="Species", 
                color_discrete_map = color_discrete_map,
                marginal='violin')

        else:
            fig = px.histogram(
                data, x=feature, 
                histfunc= "avg",
                histnorm = "probability density",
                opacity=0.50, 
                color="Species", 
                color_discrete_map = color_discrete_map,
                marginal='violin')

    fig.update_layout(
        showlegend=True,
        legend=dict(
            title=None,
            orientation="h", 
            yanchor="top", y=0.70, 
            xanchor="right", x=0.98
            ),
        width=900, height=400,
        margin=dict(l=180,r=150,t=0,b=0),
        xaxis=dict(title=feature, 
            title_font=dict(size=16), tickfont=dict(size=14)), 
        yaxis=dict(title= select_func + "  [" + select_norm + "]  ", 
            title_font=dict(size=16), tickfont=dict(size=14)) )

    st.plotly_chart(fig)



def show_matrixPlot(data):
    """
    input:
    iris dataset

    output:
    the correlation scatter plots of all features
    """

    st.subheader("Coorelation Matrix Plot")
    
    st.write(" ")

    color_discrete_map = {
    'setosa': '#440154', 
    'versicolor': '#1f9e89',
    'virginica': '#fde725'
    }

    fig = px.scatter_matrix(
        data, 
        dimensions = data.columns[0:4],
        color="Species", 
        symbol= "Species",
        color_discrete_map = color_discrete_map

        )

    fig.update_traces(
        marker_size=5,
        diagonal_visible=False
        )

    fig.update_layout(
    showlegend=False,
    legend=dict(
        title=None, 
        orientation="v", 
        yanchor="top", y=0.70, 
        xanchor="right", x=0.98
        ),
    width=900, height=480,
    margin=dict(l=120,r=120,t=0,b=0)
    )

    st.plotly_chart(fig)




def main():
    text = """
    <h4 style='text-align: center; color: royalblue;'> 
    Streamlit is a cool software for rapid prototyping Data Science tasks
    <h6 style='text-align: center; color: cornflowerblue;'> 
    I have been working on a series to explore various functionalities of Streamlit 
    <h6 style='text-align: center; color: lightsteelblue;'> 
    This app is the first of the series
    </h6>
    """
    st.markdown(text, unsafe_allow_html=True)


    text = """
    ## Exploratory Data Analysis 
    ###### Iris Dataset
    <font color='red'> Created by Nurur Rahman </font>
    """
    st.markdown(text, unsafe_allow_html=True)
   
    st.write(" ")
    st.write(" ")
    st.info("Show Plots")
 


    df = get_data()
    show_scatterPlot(df)
    show_boxPlot(df)
    show_histogramPlot(df)
    show_matrixPlot(df)


    st.write(" ")
    st.write(" ")
    st.write(" ")
    st.write(" ")

    text = """
    <h6 style='text-align: center; color: royalblue;'> 
    Streamlit is Awesome
    </h6>
    """
    st.markdown(text, unsafe_allow_html=True)




# Run the App
main()

