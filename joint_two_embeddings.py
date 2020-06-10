
def get_emb_dic(emb_file_path):

    emb_dic = {}

    with open(emb_file_path, 'r') as f:
        for line in f:
            line = line.strip()
            _id_vector = line.split(' ')
            _id, _vector = _id_vector[0], _id_vector[1:]
            if len(_vector) < 2:
                continue
            emb_dic[_id] = _vector

    return emb_dic


def get_emb_ids(emb_file_path):

    _ids = set()

    with open(emb_file_path, 'r') as f:
        for line in f:
            line = line.strip()
            _id_vector = line.split(' ')
            _id, _vector = _id_vector[0], _id_vector[1:]
            if len(_vector) < 2:
                continue
            _ids.add(_id)

    return list(_ids)


def joint_embs(src_emb_file_path, targ_emb_file_path, output_emb_path, method):

    src_embeddings = get_emb_dic(src_emb_file_path)
    tar_embeddings = get_emb_dic(targ_emb_file_path)
    ids = get_emb_ids(src_emb_file_path)

    error_counts = 0
    with open(output_emb_path, 'w+') as f:
        for _id in ids:
            try:
                src_embedding = src_embeddings[_id]
                tar_embedding = tar_embeddings[_id]
                result_embedding = src_embedding + tar_embedding
                f.write(_id + ' %s\n' % ' '.join(map(str, result_embedding)))
            except:
                print("this osm_id encounter error:", _id)
                error_counts += 1

        print("How many osm_id can not find embedding:", error_counts)


if __name__ == '__main__':

    joint_embs(src_emb_file_path='sanfrancisco/segment/sf_gcn_raw_feature_segment_blvd_16d.embedding',
               targ_emb_file_path='sanfrancisco/segment/sf_raw_feature_segment_tiger_without_avenue.embeddings',
               output_emb_path='sanfrancisco/segment/sf_gcn_segment_target_is_blvd_16d_concat_raw_feature_without_avenue.embedding',
               method='&')
