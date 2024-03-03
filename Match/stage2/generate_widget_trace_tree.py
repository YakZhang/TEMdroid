# coding=utf-8

"""
input: original widget_graph,  a list of same_state_group
output: widget_trace_tree
"""


import networkx as nx
import matplotlib.pyplot as plt

def generate_tree(graph_path, same_state_dict, state_id_dict):
    """
    each same level, we only use the first to represent all same nodes
    so, if there are three same nodes in the parent level, we only use one to represent them
    as for child level, we use all the different children for all parents.
    We use a set to save the different children, only different level children can be saved.
    :param graph_path:
    :param same_state_dict:
    :param state_id_dict:
    :return:
    """
    widget_graph = nx.DiGraph()   # initial a graph to save the tree
    ori_graph = nx.read_adjlist(graph_path, create_using = nx.DiGraph)
    widget_corresponding_graph = nx.DiGraph()   #corresponding to the widget_graph but the nodes are screen_id
    id_state_dict = {v: k for k, v in state_id_dict.items()}    # traverse the state_id_dict. The new key is the state_id, the new value is screen_id
    nodes = list(ori_graph.nodes)  # a node list
    first_node = nodes[0]
    parent_nodes = [first_node]    # the first is the parent_node
    while len(nodes) > 0:
        childs_nodes = set() # save all the childs of all the parents
        for parent_node in parent_nodes:
            child_nodes = set()  # save the child nodes in the same level for one parent
            if parent_node in nodes:
                nodes.remove(parent_node)
            same_states = same_state_dict[parent_node]
            for same_state in same_states:
                child_node_iterator = ori_graph.neighbors(same_state)   # get child
                for child_node in child_node_iterator:
                    child_nodes.add(same_state_dict[child_node][0]) # use the first same state to represent the list of same state, the first is the earliest in time
                    childs_nodes.add(same_state_dict[child_node][0])
                    if child_node in nodes:
                        nodes.remove(child_node)
            if  widget_graph.has_node(parent_node)==False:
                widget_graph.add_node(parent_node) # add parent node
                parent_screen_node = id_state_dict[parent_node]
                widget_corresponding_graph.add_node(parent_screen_node)
                for child_node in child_nodes:
                    widget_graph.add_node(child_node) # add child node
                    widget_graph.add_edge(parent_node, child_node) # add edge
                    child_screen_node = id_state_dict[child_node]
                    widget_corresponding_graph.add_node(child_screen_node)
                    widget_corresponding_graph.add_edge(parent_screen_node, child_screen_node)
            else:
                parent_screen_node = id_state_dict[parent_node]
                for child_node in child_nodes:
                    widget_graph.add_node(child_node) # add child node
                    widget_graph.add_edge(parent_node, child_node) # add edge
                    child_screen_node = id_state_dict[child_node]
                    widget_corresponding_graph.add_node(child_screen_node)
                    widget_corresponding_graph.add_edge(parent_screen_node, child_screen_node)
        parent_nodes = childs_nodes
    return widget_graph, widget_corresponding_graph

def draw_tree(G2):
    subax1 = plt.subplot(121)
    nx.draw(G2, with_labels=True, font_weight='bold' )
    plt.show()

def dumpG(G,prefix,suffix):
    path = prefix+suffix+".adjlist"
    fh = open(path,'wb')
    nx.write_adjlist(G, fh)


