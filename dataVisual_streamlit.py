import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Streamlit UI
st.title("📊 Simple Data Visualization App")

# File uploader
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file:
    # Load the data
    df = pd.read_csv(uploaded_file)
    
    # Show data preview
    st.write("### Data Preview")
    st.write(df.head())

    # Show basic stats
    if st.checkbox("Show Summary Statistics"):
        st.write(df.describe())

    # Sidebar options for graphs
    st.sidebar.header("📈 Graph Options")
    graph_type = st.sidebar.selectbox("Select a plot type", ["Histogram", "Bar Chart", "Line Graph", "Scatter Plot"])
    all_columns = df.columns.tolist()
    selected_columns = st.sidebar.multiselect("Select columns", all_columns, default=all_columns[:1])

    # Generate plot
    if st.sidebar.button("Generate Plot"):
        if len(selected_columns) < 1:
            st.warning("Please select at least one column.")
        else:
            fig, ax = plt.subplots(figsize=(8, 6))

            if graph_type == "Histogram":
                df[selected_columns].hist(ax=ax, bins=10, color="skyblue")
                plt.title(f"Histogram of {', '.join(selected_columns)}")

            elif graph_type == "Bar Chart":
                df[selected_columns[0]].value_counts().plot(kind="bar", ax=ax, color='orange')
                plt.title(f"Bar Chart of {selected_columns[0]}")

            elif graph_type == "Line Graph":
                df[selected_columns].plot(ax=ax, title="Line Graph")

            elif graph_type == "Scatter Plot":
                if len(selected_columns) < 2:
                    st.warning("Scatter plot needs at least 2 columns.")
                else:
                    sns.scatterplot(x=df[selected_columns[0]], y=df[selected_columns[1]], ax=ax)
                    plt.title(f"Scatter Plot: {selected_columns[0]} vs {selected_columns[1]}")

            st.pyplot(fig)
