import numpy as np


def get_segments(file_path):

    _seg = set()

    with open(file_path) as f:
        for line in f:
            ids_cls = line.strip().split(' ')
            _seg.add(ids_cls[0])

    return list(_seg)


def get_features(file_path):

    _feat = set()

    with open(file_path) as f:
        for line in f:
            ids_cls = line.strip().split(' ')
            _feat.add(ids_cls[1])

    return list(_feat)


def get_segment_feature(file_path):

    _feature_dic = {}

    with open(file_path) as f:
        for line in f:
            id_feat = line.strip().split(' ')
            _feature_dic[id_feat[0]] = int(id_feat[1])

    return _feature_dic


def gen_raw_feature_vec(segments, features, seg_feat_dic):

    _emb = {}

    feat_size = len(features)

    for seg in segments:
        _cls = seg_feat_dic[seg]
        cls_loc = features.index(str(_cls))

        _cls_vec = np.zeros(feat_size)
        _cls_vec[cls_loc] = 1
        _cls_vec = _cls_vec.tolist()

        _emb[seg] = _cls_vec

    return _emb


def print_raw_feat_emb(embedding_dic, output_file_path):

    with open(output_file_path, 'w+') as f:
        for key in embedding_dic:
            _emb = embedding_dic[key]
            f.write(key + ' ')
            f.write(' '.join(map(str, map(int, _emb))))
            f.write('\n')


if __name__ == '__main__':

    segment_type_path = 'sanfrancisco/segment/segment_types.sanfrancisco'

    segments = get_segments(segment_type_path)

    features = get_features(segment_type_path)

    seg_feat_dic = get_segment_feature(segment_type_path)

    raw_feature_emb = gen_raw_feature_vec(segments, features, seg_feat_dic)

    emb_file_path = 'sanfrancisco/segment/sanfrancisco_raw_feature_segment.embeddings'
    print_raw_feat_emb(raw_feature_emb, emb_file_path)
