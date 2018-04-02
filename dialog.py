# 어휘 사전과 워드 임베딩을 만들고, 학습을 위해 대화 데이터를 읽어들이는 유틸리티들의 모음
import tensorflow as tf
import numpy as np
import re
import codecs

from config import FLAGS


class Dialog():

    ''' Word2Vec 4가지 symbol
        1. PAD : input이 같은 width 갖게끔
        2. EOS : Decoder에게 Sentence가 끝나는 지점 알려주며, Decoder의 output의 끝.
        3. UNK : Training 과정에서 잘 나타나지 않는 단어들을 무시함으로서 효율을 높일 수 있음.
        4. GO : Decoder의 처음 timestamp의 입력으로 decoder에게 output을 생성해야 한다고 알림.
    '''
    _PAD_ = "_PAD_"  # 빈칸 채우는 심볼
    _STA_ = "_STA_"  # 디코드 입력 시퀀스의 시작 심볼
    _EOS_ = "_EOS_"  # 디코드 입출력 시퀀스의 종료 심볼
    _UNK_ = "_UNK_"  # 사전에 없는 단어를 나타내는 심볼

    _PAD_ID_ = 0
    _STA_ID_ = 1
    _EOS_ID_ = 2
    _UNK_ID_ = 3
    _PRE_DEFINED_ = [_PAD_ID_, _STA_ID_, _EOS_ID_, _UNK_ID_]

    def __init__(self):
        self.vocab_list = []
        self.vocab_dict = {}
        self.vocab_size = 0
        self.examples = []

        self._index_in_epoch = 0

    def decode(self, indices, string=False):
        tokens = [[self.vocab_list[i] for i in dec] for dec in indices]

        if string:
            return self.decode_to_string(tokens[0])
        else:
            return tokens

    def decode_to_string(self, tokens):
        text = ' '.join(tokens)
        return text.strip()

    def cut_eos(self, indices):
        eos_idx = indices.index(self._EOS_ID_)
        return indices[:eos_idx]

    def is_eos(self, voc_id):
        return voc_id == self._EOS_ID_

    def is_defined(self, voc_id):
        return voc_id in self._PRE_DEFINED_

    def max_len(self, batch_set):
        max_len_input = 0
        max_len_output = 0

        for i in range(0, len(batch_set), 2):
            len_input = len(batch_set[i])
            len_output = len(batch_set[i+1])
            if len_input > max_len_input:
                max_len_input = len_input
            if len_output > max_len_output:
                max_len_output = len_output

        return max_len_input, max_len_output + 1

    def pad(self, seq, max_len, start=None, eos=None):
        if start:
            # 디코더의 입력이 시작됨을 알려주는 심볼 추가
            padded_seq = [self._STA_ID_] + seq
        elif eos:
            # 디코더의 출력이 끝났음을 알려주는 심볼 추가
            padded_seq = seq + [self._EOS_ID_]
        else:
            padded_seq = seq

        if len(padded_seq) < max_len:
            # 오른쪽에 패드
            return padded_seq + ([self._PAD_ID_] * (max_len - len(padded_seq)))
        else:
            return padded_seq

    def pad_left(self, seq, max_len):
        # 왼쪽에 패드
        if len(seq) < max_len:
            return ([self._PAD_ID_] * (max_len - len(seq))) + seq
        else:
            return seq

    def transform(self, input, output, input_max, output_max):
        enc_input = self.pad(input, input_max)
        dec_input = self.pad(output, output_max, start=True)
        target = self.pad(output, output_max, eos=True)

        # 구글 방식으로 입력을 인코더에 역순으로 입력한다.
        enc_input.reverse()

        # np.eye() 는 나머지를 0으로 다 채우고 대각선으로 1이 채워진 2차원 numpy array 를 만든다.
        # one-hot-encoding
        enc_input = np.eye(self.vocab_size)[enc_input]
        dec_input = np.eye(self.vocab_size)[dec_input]
        
        return enc_input, dec_input, target

    def next_batch(self, batch_size):
        enc_input = []
        dec_input = []
        target = []

        start = self._index_in_epoch # 시작점

        # index 옮겨놓기
        if self._index_in_epoch + batch_size < len(self.examples) - 1:
            self._index_in_epoch = self._index_in_epoch + batch_size
        else:
            self._index_in_epoch = 0

        # 미니 배치 셋 
        batch_set = self.examples[start:start+batch_size]

        # 작은 데이터셋을 실험하기 위한 꼼수
        # 현재의 답변을 다음 질문의 질문으로 하고, 다음 질문을 답변으로 하여 데이터를 늘린다.?
        if FLAGS.data_loop is True:
            batch_set = batch_set + batch_set[1:] + batch_set[0:1]

        # TODO: 구글처럼 버킷을 이용한 방식으로 변경
        # 간단하게 만들기 위해 구글처럼 버킷을 쓰지 않고 같은 배치는 같은 사이즈를 사용하도록 만듬
        max_len_input, max_len_output = self.max_len(batch_set)

        for i in range(0, len(batch_set) - 1, 2):
            # input: batch_set[i]
            # output: batch_set[i+1]
            enc, dec, tar = self.transform(batch_set[i], batch_set[i+1],
                                           max_len_input, max_len_output)

            enc_input.append(enc)
            dec_input.append(dec)
            target.append(tar)

        return enc_input, dec_input, target

    def tokens_to_ids(self, tokens):
        ids = []
        for t in tokens:
            if t in self.vocab_dict: # 단어장에 있으면 ids 변경
                ids.append(self.vocab_dict[t])
            else: # 단어장에 없으면 UNKNOWN
                ids.append(self._UNK_ID_)

        return ids

    def ids_to_tokens(self, ids):
        tokens = []
        for i in ids: # ids -> 단어 대체
            tokens.append(self.vocab_list[i])

        return tokens

    def load_examples(self, data_path):
        self.examples = []
        with open(data_path, 'r', encoding='utf-8') as content_file:
            # 데이터 파일 읽어 들여서 한 줄씩 tokenizer -> ids(단어장의 인덱스) 변경
            # examples 에는 ids(e.g. [3, 2, 5, UNK, ....])저장
            for line in content_file:
                tokens = self.tokenizer(line.strip())
                ids = self.tokens_to_ids(tokens)
                self.examples.append(ids)

    def tokenizer(self, sentence):
        # 공백으로 나누고 특수문자는 따로 뽑아낸다.
        words = []
        _TOKEN_RE_ = re.compile("([.,!?\"':;)(])")

        for fragment in sentence.strip().split():
            words.extend(_TOKEN_RE_.split(fragment))

        return [w for w in words if w]

    def build_vocab(self, data_path, vocab_path):
        with open(data_path, 'r', encoding='utf-8') as content_file:
            # 데이터 파일을 읽어 들여서 tokenize후 단어들을 List로 words에 저장
            content = content_file.read()
            words = self.tokenizer(content)
            words = list(set(words))

        with open(vocab_path, 'w') as vocab_file:
            # 단어장 파일에 사용된 단어들 저장
            for w in words:
                vocab_file.write(w + '\n')

    def load_vocab(self, vocab_path):
        self.vocab_list = self._PRE_DEFINED_ + []
        # 단어장 파일 읽어와서 vocab_list에 저장
        with open(vocab_path, 'r', encoding='utf-8') as vocab_file:
            for line in vocab_file:
                self.vocab_list.append(line.strip())

        # {'_PAD_': 0, '_STA_': 1, '_EOS_': 2, '_UNK_': 3, 'Hello': 4, 'World': 5, ...}
        self.vocab_dict = {n: i for i, n in enumerate(self.vocab_list)} # 단어:숫자 매치
        self.vocab_size = len(self.vocab_list) # 총 단어 수


def main(_):
    dialog = Dialog()

    if FLAGS.data_path and FLAGS.voc_test:
        print("다음 데이터로 어휘 사전을 테스트합니다.", FLAGS.data_path)
        dialog.load_vocab(FLAGS.voc_path)
        dialog.load_examples(FLAGS.data_path)

        enc, dec, target = dialog.next_batch(10)
        print(target)
        enc, dec, target = dialog.next_batch(10)
        print(target)

    elif FLAGS.data_path and FLAGS.voc_build:
        print("다음 데이터에서 어휘 사전을 생성합니다.", FLAGS.data_path)
        dialog.build_vocab(FLAGS.data_path, FLAGS.voc_path)

    elif FLAGS.voc_test:
        dialog.load_vocab(FLAGS.voc_path)
        print(dialog.vocab_dict)


if __name__ == "__main__":
    tf.app.run()
