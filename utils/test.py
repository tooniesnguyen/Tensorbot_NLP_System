from nltk.translate import bleu_score
from nltk.tokenize import TweetTokenizer

# Khởi tạo đối tượng tokenizer
tokenizer = TweetTokenizer()

def calc_bleu_many(cand_seq, ref_sequences):
    # Tạo đối tượng SmoothingFunction
    sf = bleu_score.SmoothingFunction()
    # Tính toán BLEU score cho nhiều chuỗi dự đoán và nhiều chuỗi tham chiếu
    return bleu_score.sentence_bleu(ref_sequences, cand_seq,
                                    smoothing_function=sf.method1,
                                    weights=(0.5, 0.5))

# Chuỗi dự đoán
cand_seq = "current popular methods are statistical"
# Danh sách các chuỗi tham chiếu
ref_sequences = [
    "currently popular approaches include statistical methods",
    "current popular methods involve statistics",
    "statistical methods are currently popular"
]

# Tách từ trong chuỗi dự đoán và các chuỗi tham chiếu
cand_tokens = tokenizer.tokenize(cand_seq)
ref_tokens_list = [tokenizer.tokenize(ref_seq) for ref_seq in ref_sequences]

# Tính toán BLEU score cho nhiều chuỗi tham chiếu
bleu_score_many = calc_bleu_many(cand_tokens, ref_tokens_list)

# In kết quả BLEU score
print("BLEU score:", bleu_score_many)
