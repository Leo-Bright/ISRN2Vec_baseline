import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# load dataset into Pandas DataFrame
dimension = 2
emb_file_path = 'sanfrancisco/segment/sanfrancisco_raw_feature_segment.embeddings'
path_array = emb_file_path.rsplit('.', 1)
pca_emb_file_path = path_array[0] + '_pca_' + str(dimension) + 'd.embeddings'


df = pd.read_csv(emb_file_path, header=None, sep=' ', index_col=0)
rows_size, cols_size = df.shape
x = df

# Standardizing the features
x = StandardScaler().fit_transform(x)


def save_embeddings(embeddings, raw_emb_file_path, output_file_path):

    combined_emb = combine_id_emb(raw_emb_file_path, embeddings)

    with open(output_file_path, 'w+') as f:
        for embedding in combined_emb:
            f.write(' '.join(map(str, embedding)))
            f.write('\n')


def combine_id_emb(raw_emb_file, pca_emb):

    osmids = []
    combined_embeddings = []

    with open(raw_emb_file) as f:
        for line in f:
            id_ = line.strip().split(' ')
            osmids.append(id_[0])

    assert len(osmids) == len(pca_emb)

    for i in range(len(osmids)):
        _id = []
        _emb = []
        _id.append(osmids[i])
        for item in pca_emb[i]:
            _emb.append(str(item))
        _combined_emb = _id + _emb
        combined_embeddings.append(_combined_emb)

    return combined_embeddings


pca = PCA(n_components=dimension)
print('training pca model: ')
pca_transform = pca.fit_transform(x)

save_embeddings(pca_transform, emb_file_path, pca_emb_file_path)
print('training done! and the embeddings saved!')
