

import networkx as nx
import json
from exploration.Migration.dump_utg import query_node


def create_graph(utg_data):
    DG =nx.MultiDiGraph()
    for node in utg_data['nodes']:
        node_name = node['name']
        DG.add_node(node_name)
    for edge in utg_data['edges']:
        widget_name = edge['widget_name']
        from_node_name = edge['from_state_name']
        to_node_name = edge['to_state_name']
        DG.add_edge(from_node_name,to_node_name)
        DG.add_edge(from_node_name,to_node_name,widget_name=widget_name)
    return DG

def query_shortest_edge(DG,from_state_name, end_node_list):
    end_node = None
    shortest_path = None
    path_len = 100
    time_num = 1000000000
    for end_node_current in end_node_list:
        if nx.has_path(DG,from_state_name,end_node_current):
            shortest_path_by_node = nx.shortest_path(DG,source=from_state_name,target=end_node_current)
            current_time_num = int(end_node_current.split("_")[-1].replace(".xml",""))
            if path_len > len(shortest_path_by_node):
                shortest_path = shortest_path_by_node
                end_node = end_node_current
                path_len = len(shortest_path_by_node)
                time_num = current_time_num
            elif path_len == len(shortest_path_by_node) and time_num > current_time_num:
                shortest_path = shortest_path_by_node
                end_node = end_node_current
                path_len = len(shortest_path_by_node)
                time_num = current_time_num
    return end_node, shortest_path

def get_nodes_according_edge_name(DG,widget_name):
    from_node_list = []
    for (from_node, to_node, wn) in DG.edges.data('widget_name'):
        widget_name = widget_name.replace("Sample.To.do","Sample\bTo\bdo")
        if wn == widget_name:
            from_node_list.append(from_node)
    return from_node_list


def get_multi_jump_node(utg_data, from_state_name, widget_name):
    # given json_line, output utg
    # create a graph
    # given from_node and widget_name
    # output end_node (shortest_path)

    DG = create_graph(utg_data)
    end_node_list = get_nodes_according_edge_name(DG, widget_name)
    # the end_node is the last node before the widget_name
    end_node,_ = query_shortest_edge(DG, from_state_name, end_node_list)
    # given the end_node (also the from node for widget_name) and widget_name output the new_state
    to_node_name = None
    to_node_name_list = []
    for (from_node_name, to_node_name, wn) in DG.edges.data('widget_name'):
        if from_node_name == end_node and wn == widget_name:
            to_node_name = to_node_name
            to_node_name_list.append(to_node_name)
    if len(to_node_name_list) == 0:
        return None
    else:
        to_node_name, _ = query_shortest_edge(DG, from_state_name, to_node_name_list)
        to_node = query_node(utg_data, to_node_name)
        return to_node






