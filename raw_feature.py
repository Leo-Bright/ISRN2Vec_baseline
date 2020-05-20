import numpy as np


def get_network_nodes(network_file_path):

    _nodes = set()

    with open(network_file_path) as f:
        for line in f:
            ids = line.strip().split(' ')
            _nodes.add(ids[0])
            _nodes.add(ids[1])

    return list(_nodes)


def get_node_feature(feature_file_path):

    _feature_dic = {}

    with open(feature_file_path) as f:
        for line in f:
            id_feat = line.strip().split(' ')
            _feature_dic[id_feat[0]] = int(id_feat[1])

    return _feature_dic


def gen_raw_feature_vec(network_nodes, node_tag_dic, node_type_dic):

    _emb = {}

    for node in network_nodes:
        _tag = node_tag_dic[node]
        _type = node_type_dic[node]

        _tag_vec = np.zeros(9)
        _tag_vec[_tag] = 1
        _tag = _tag_vec.tolist()

        _type_vec = np.zeros(8)
        _type_vec[_type - 1] = 1
        _type = _type_vec.tolist()

        node_feat_vec = _tag + _type
        _emb[node] = node_feat_vec

    return _emb


def print_raw_feat_emb(embedding_dic, output_file_path):
    with open(output_file_path, 'w+') as f:
        for key in embedding_dic:
            _emb = embedding_dic[key]
            f.write(key + ' ')
            f.write(' '.join(map(str, map(int, _emb))))
            f.write('\n')


if __name__ == '__main__':

    network_path = 'sanfrancisco/sanfrancisco.network'

    tag_path = 'sanfrancisco/sanfrancisco_nodes_with_crossing_increament_stop.tag'

    type_path = 'sanfrancisco/node_type.sanfrancisco'

    nodes = get_network_nodes(network_path)

    nodes_tag = get_node_feature(tag_path)

    nodes_type = get_node_feature(type_path)

    raw_feature_emb = gen_raw_feature_vec(nodes, nodes_tag, nodes_type)

    emb_file_path = 'sanfrancisco/sanfrancisco_raw_feature_crossing.embeddings'
    print_raw_feat_emb(raw_feature_emb, emb_file_path)
