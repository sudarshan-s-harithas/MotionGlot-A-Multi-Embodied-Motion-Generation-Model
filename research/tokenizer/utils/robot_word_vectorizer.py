import numpy as np
import pickle
from os.path import join as pjoin

POS_enumerator = {
    'VERB': 0,
    'NOUN': 1,
    'DET': 2,
    'ADP': 3,
    'NUM': 4,
    'AUX': 5,
    'PRON': 6,
    'ADJ': 7,
    'ADV': 8,
    'Loc_VIP': 9,
    'Body_VIP': 10,
    'Obj_VIP': 11,
    'Act_VIP': 12,
    'Desc_VIP': 13,
    'OTHER': 14,
}

Loc_list = ('left', 'right', 'clockwise', 'counter-clockwise', 'anticlockwise', 'forward', 'back', 'backward',
            'up', 'down', 'straight', 'curve' )

Body_list = ('Robot', 'robot' )


Act_list = ('walk', 'turns', 'Does not move' , 'Spins on the spot')



VIP_dict = {
    'Loc_VIP': Loc_list,
    'Body_VIP': Body_list,
    'Act_VIP': Act_list
}


class WordVectorizer(object):
    def __init__(self, meta_root, prefix):
        vectors = np.load(pjoin(meta_root, '%s_data.npy'%prefix))
        words = pickle.load(open(pjoin(meta_root, '%s_words.pkl'%prefix), 'rb'))
        self.word2idx = pickle.load(open(pjoin(meta_root, '%s_idx.pkl'%prefix), 'rb'))
        self.word2vec = {w: vectors[self.word2idx[w]] for w in words}

    def _get_pos_ohot(self, pos):
        pos_vec = np.zeros(len(POS_enumerator))
        if pos in POS_enumerator:
            pos_vec[POS_enumerator[pos]] = 1
        else:
            pos_vec[POS_enumerator['OTHER']] = 1
        return pos_vec

    def __len__(self):
        return len(self.word2vec)

    def __getitem__(self, item):

        word, pos = item.split('\n')
        if word in self.word2vec:
            word_vec = self.word2vec[word]
            vip_pos = None
            for key, values in VIP_dict.items():
                if word in values:
                    vip_pos = key
                    break
            if vip_pos is not None:
                pos_vec = self._get_pos_ohot(vip_pos)
            else:
                pos_vec = self._get_pos_ohot(pos)
        else:
            word_vec = self.word2vec['unk']
            pos_vec = self._get_pos_ohot('OTHER')
        
        return word_vec, pos_vec


# class WordVectorizerV2(WordVectorizer):
#     def __init__(self, meta_root, prefix):
#         super(WordVectorizerV2, self).__init__(meta_root, prefix)
#         self.idx2word = {self.word2idx[w]: w for w in self.word2idx}

#     def __getitem__(self, item):
#         word_vec, pose_vec = super(WordVectorizerV2, self).__getitem__(item)
#         word, pos = item.split(' ')
#         print(word)
#         if word in self.word2vec:
#             return word_vec, pose_vec, self.word2idx[word]
#         else:
#             return word_vec, pose_vec, self.word2idx['unk']

#     def itos(self, idx):
#         if idx == len(self.idx2word):
#             return "pad"
#         return self.idx2word[idx]