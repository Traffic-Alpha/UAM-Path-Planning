import os
import pandas as pd
import networkx as nx
import math
import pickle as pkl
# from math import radians, degrees, sin, cos, asin, acos, sqrt
import numpy as np

def gen_road_graph(road_dict, node_hop=50):
    G = nx.DiGraph()
    for id, road in road_dict.items():
        length = road["length"]
        road_source = road["source"]
        road_target = road['target']
        node_num = int(length//node_hop) # 向下取整
        if node_num > 0:
            for i in range(node_num):
                node_id = id+"_"+str(i)
                node_len = node_hop
                uncertainty = 0
                edge_list = []
                if i == node_num-1 : # 最后一个node
                    node_id = id+"_00"
                    node_len = length - node_hop * (node_num - 1)
                    target = []
                    source = [id+"_"+str(i-1)]
                    # if len(road_target) == 2:
                    #     print("p")
                    for rt in road_target:
                        if rt!="d": # 非终点的最后一个node
                            target.append(rt+"_0")
                        else: # 终点的最后一个node
                            target.append(rt)
                            break
                    edge_list.append((source[0], node_id))

                elif i == 0: # 第一个node
                    target = [id + "_" + str(i + 1)]
                    source = []
                    # edge_list.append((node_id, target))
                    for s in road_source:
                        if s =="s": # 起始路段的第一个node source = ["s"]
                            source.append(s)
                            break
                        else: # 非起始路段的第一个node
                            source.append(s + "_00")
                    for ns in source:
                        if ns == "s": break # 起始路段的第一个node source = ["s"]
                        edge_list.append((ns, node_id))
                else:
                    target = [id+"_"+str(i+1)]
                    source = [id+"_"+str(i-1)]
                    edge_list.append((source[0], node_id))

                G.add_nodes_from([(node_id, {"id":node_id, "length":node_len, "source":source, "target":target,
                                             "uncertainty": uncertainty})])
                G.add_edges_from(edge_list)
        # print("Road {} has been initiated".format(id))
    # print(list(G.nodes(data=True)))
    G_out = nx.convert_node_labels_to_integers(G)
    return G_out



if __name__ =="__main__":
    # start point: s, destination: d
    road_dict = {
        "E8":{
            "length": 400.98,
            "source": ["s"],
            "target": ["E9"],
        },
        "E9":{
            "length": 198.74,
            "source": ["E8"],
            "target": ["d"]
        },
        "E12": {
            "length": 300.77,
            "source": ["s"],
            "target": ["E15","E16"]
        },
        "E15":{
            "length": 201.99,
            "source": ["E12"],
            "target": ["d"],
        },
        "E16":{
            "length": 204.66,
            "source": ["E12"],
            "target": ["E17"],
        },
        "E17": {
            "length": 102.11,
            "source": ["E16"],
            "target": ["d"],
        },
    }
    map_graph = gen_road_graph(road_dict)





