def splitscore(file_dir):
    score = []
    Prefix_str = []
    f = open(file_dir)
    for line in f:
        s =line.split()
        score.append(float(s[-1]))
        s = s[0] + ' ' + s[1] + ' ' + s[2] + ' '
        Prefix_str.append(s)
    return score,Prefix_str

file_dir1='submission/2019-01-28_15:45:05_fishnet150_52_submission.txt'
score1,Prefix_str = splitscore(file_dir1)
file_dir2 = 'submission/2019-02-13_15:22:05_FeatherNet54-se_69_submission.txt'
score2,Prefix_str = splitscore(file_dir2)
# print(Prefix_str[1])
file_dir3 = 'submission/2019-03-01_22:25:43_fishnet150_27_submission.txt'
score3,Prefix_str = splitscore(file_dir3)
#
file_dir4 = 'submission/2019-02-13_13:30:12_FeatherNet54_41_submission.txt'
score4,Prefix_str = splitscore(file_dir4)
#
file_dir5 = 'submission/2019-02-13_14:13:43_fishnet150_16_submission.txt'
score5,Prefix_str = splitscore(file_dir5)

file_dir6 = 'submission/2019-02-16_19:31:04_moilenetv2_5_submission.txt'
score6,Prefix_str = splitscore(file_dir6)
file_dir7 = 'submission/2019-02-16_19:30:02_moilenetv2_7_submission.txt'
score7,Prefix_str = splitscore(file_dir7)
file_dir8 = 'submission/2019-02-16_19:28:47_moilenetv2_6_submission.txt'
score8,Prefix_str = splitscore(file_dir8)


file_dir9 = 'submission/2019-03-01_17:10:11_mobilelitenetB_48_submission.txt'
score9,Prefix_str = splitscore(file_dir9)
file_dir10 = 'submission/2019-03-01_17:38:27_mobilelitenetA_51_submission.txt'
score10,Prefix_str = splitscore(file_dir10)

# scores =[score1,score2,score3,score4,score5,score6,score7,score8,score9]
scores = [score1,score2,score3,score4,score5,score6,score7,score8,score9,score10]

def Average(lst):
    return sum(lst) / len(lst)
def fecth_ensembled_score(scores, threshold):
    ensembled_score  = []
    for i in range(len(score1)):
        line_socres = [scores[j][i] for j in range(len(scores))]
        mean_socre = Average(line_socres)
        if mean_socre > threshold:
            ensembled_score.append(max(line_socres))
        else:
            ensembled_score.append(min(line_socres))
    return ensembled_score

def num_err(ensembled_score,threshold,real_scores):
    count = 0
    for i in range(len(real_scores)):
        if real_scores[i] == (ensembled_score[i]>0.5):
            pass
        else:
            count = count + 1
    if count < 50:
        print('threshold: {:.3f} num_errors is {}'.format(threshold,count))
    return count

# submission_ensembled_file_dir='data/val_label.txt'
submission_ensembled_file_dir='data/test_private_list.txt'
real_scores,Prefix_str = splitscore(submission_ensembled_file_dir)
print('img num in test: ',len(real_scores))
      
def get_best_threshold():
    min_count = 10000000
    best_threshold = 0.0
    for i in range(100):
        threshold = i / 100
        ensembled_score = fecth_ensembled_score(scores, threshold)
        count = num_err(ensembled_score,threshold,real_scores)
        if count < min_count:
            min_count = count
            best_threshold = threshold
    return best_threshold
      
best_threshold = get_best_threshold()
print('best threshold is :',best_threshold)
submission_ensembled_file_dir='submission/final_submission.txt'
ensembled_file = open(submission_ensembled_file_dir,'a')
ensembled_score = fecth_ensembled_score(scores, best_threshold)
for i in range(len(ensembled_score)):
    ensembled_file.write(Prefix_str[i]+str(ensembled_score[i])+'\n')
ensembled_file.close()
