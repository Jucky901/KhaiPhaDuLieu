import sys
import os
from model.DecisionTree import DecisionTree
import pandas as pd
import streamlit as st
import numpy as np
import graphviz

data = pd.read_csv('DataMining/weather_data.csv')

X = data.drop(columns=['Id', 'play']).values
y = data['play'].values
feature_names = data.drop(columns=['Id', 'play']).columns.tolist()


for i in range(1,20):
    model = DecisionTree(max_depth=i, feature_names=feature_names)
    model.fit(X, y)
    predict = model.predict(X)
    if np.array_equal(y, predict):
        print(f"Max depth is {i}" )
        break


st.write("Decision Tree Structure:")
st.write(model.print_tree())
print(model.print_tree())

def create_graphviz_tree(node, graph, parent_name=""):
    node_name = str(id(node))

    # Leaf Node
    if "leaf" in node:
        label = f"Leaf: {node['leaf']}"
        graph.node(node_name, label, shape="box")

    # Decision Node
    else:
        label = f"[Feature {node['feature']} == {node['threshold']}]"
        graph.node(node_name, label, shape="ellipse")

        # True Branch
        true_child = node["true"]
        true_child_name = create_graphviz_tree(true_child, graph, node_name)
        graph.edge(node_name, true_child_name, label="True")

        # False Branch
        false_child = node["false"]
        false_child_name = create_graphviz_tree(false_child, graph, node_name)
        graph.edge(node_name, false_child_name, label="False")

    return node_name


def visualize_tree(tree_structure):
    graph = graphviz.Digraph(format="png")
    create_graphviz_tree(tree_structure, graph)
    return graph

st.title("Decision Tree Visualizer")
    
st.subheader("Tree Structure")
st.code(model.get_tree())

st.subheader("Visualized Tree")
tree_graph = visualize_tree(model.get_tree())
st.graphviz_chart(tree_graph.source)



