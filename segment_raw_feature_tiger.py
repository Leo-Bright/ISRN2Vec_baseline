import numpy as np
import json


def get_segments(file_path):

    _seg = set()

    with open(file_path) as f:
        for line in f:
            ids_cls = line.strip().split(' ')
            _seg.add(ids_cls[0])

    return list(_seg)


def get_features(tiger_json_path, without=None):

    _feat = set()

    if without is not None:
        without = set(without)

    for key in tiger_json_path:
        _cls = tiger_json_path[key]
        if without is not None:
            if _cls in without :
                continue
        _feat.add(_cls)

    return list(_feat)


def gen_raw_feature_vec(segments, features, seg_feat_dic):

    _emb = {}

    feat_size = len(features)

    feat_set = set(features)

    for seg in segments:

        _cls_vec = np.zeros(feat_size)

        if seg in seg_feat_dic and seg_feat_dic[seg] in feat_set:

            _cls = seg_feat_dic[seg]
            cls_loc = features.index(str(_cls))
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
    tiger_dict = json.load(open('sanfrancisco/segment/sf_segments_tiger_nametype.json'))

    segments = get_segments(segment_type_path)

    print(len(segments))

    features = get_features(tiger_dict, without=('Ave', ))

    raw_feature_emb = gen_raw_feature_vec(segments, features, tiger_dict)

    emb_file_path = 'sanfrancisco/segment/sf_raw_feature_segment_tiger_without_avenue.embeddings'
    print_raw_feat_emb(raw_feature_emb, emb_file_path)
