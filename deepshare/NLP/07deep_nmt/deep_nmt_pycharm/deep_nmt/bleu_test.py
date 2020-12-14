from nltk.translate.bleu_score import sentence_bleu
reference = [[1, 2, 3, 1, 5, 6,7]]
candidate = [1,1,1,1,1,1,1]
score = sentence_bleu(reference, candidate, weights=(1, 0, 0, 0))
print(score)

