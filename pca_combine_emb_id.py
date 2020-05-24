

def combine_id_emb(raw_emb, kpca_emb, output):

    osmids = []
    embeddings = []
    combined_embeddings = []

    with open(raw_emb) as f:
        for line in f:
            id_ = line.strip().split(' ')
            osmids.append(id_[0])

    with open(kpca_emb) as f:
        for line in f:
            emb = line.strip().split(' ')
            embeddings.append(emb)

    assert len(osmids) == len(embeddings)

    for i in range(len(osmids)):
        _id = []
        _id.append(osmids[i])
        _combined_emb = _id + embeddings[i]
        combined_embeddings.append(_combined_emb)

    with open(output, 'w+') as f:
        for emb in combined_embeddings:
            f.write(' '.join(emb))
            f.write('\n')


if __name__ == '__main__':

    dimension = 6
    raw_feature_path = 'sanfrancisco/sanfrancisco_raw_feature_crossing.embeddings'

    target_emb_path = 'sanfrancisco/sanfrancisco_pca_crossing_' + str(dimension) + 'd.embeddings'
    combined_emb_path = 'sanfrancisco/sanfrancisco_combined_pca_crossing_' + str(dimension) + 'd.embeddings'

    combine_id_emb(raw_feature_path, target_emb_path, combined_emb_path)