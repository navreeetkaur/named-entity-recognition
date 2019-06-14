from itertools import chain
import nltk
import sklearn_crfsuite
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics
import re
import string
import pickle
import sys

input_file = sys.argv[1]
output_file = sys.argv[2]
print(input_file, output_file)

print("Reading input file")

test_data = []
sent = []
with open(input_file, 'r') as f:
    for line in f:
        line = line.split()
        if len(line)==0:
            test_data.append(sent)
            sent = []
        else:
            sent.append((line[0], nltk.pos_tag([line[0]])[0][1]))

print(test_data[:3])
print("Loading gazet_dict")
gazet_dict = pickle.load(open('gazet_dict.pkl','rb'))

data_A = gazet_dict['A']
data_L = gazet_dict['L']
data_N = gazet_dict['N']
data_LA = gazet_dict['LA']

def word2features(sent, i):
    word = sent[i][0]
    postag = sent[i][1]

    features = {
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'postag': postag,
        'postag[:2]': postag[:2],
        '--bhk': 'bhk' in word.lower(),
        '#...':word[0]=="#",
        '@...': word[0]=='@',
        '+...': word[0]=='+',
        '0...': word[0]=='0',
        'A': word in data_A,
        'L':word in data_L,
        'LA':word in data_LA,
        'N':word in data_N,
        'C_':re.match('[0-9]\-[A-Za-z]+', word) is not None,
        'sqft': 'sq' in word.lower() or 'ft' in word.lower() or 'mtr' in word.lower() or 'meter' in word.lower() or 'yard' in word.lower() or 'sqft' in word.lower(),
        '\sq': '/sq' in word.lower(),
        'punkt':word in set(string.punctuation),
        "cr": 'cr' in word.lower(),
        'lac': 'lac' in word.lower(),
        'lakh' :'lakh' in word.lower(),
        "crore": 'crore' in word.lower(),
        'per': 'per' in word,
        'tel-number': re.match('[0-9]{3}-[0-9]{4}-[0-9]{3}', word.lower()) is not None,
        'tel-number-2': re.match('[0-9]{10}', word.lower()) is not None,
        "/-" :  '/-' in word,
        'Sector': "sec" in word.lower(),
        
        'abc123': word.isalnum(),
        '#abc123':word[0]=='#' and word[1:].isalnum(),
        '#abc':word[0]=='#' and word[1:].isalpha(),
        'mobile_num': len(word)>9 and word[1:].isdigit(),
        'other_num':word.isdigit() and len(word)<10,
        'len(word)':len(word),
        'islink': word[:7]=="http://",
        'num_,_.': re.match('[0-9]+,[0-9]+|[0-9]+\.[0-9]+', word.lower()) is not None,
        'N.':len(word)==1 and word.isupper(),
        'sector': 'sector' or 'sec' in word.lower(),
        'city' : 'city' or 'society' in word.lower(),
        "Place": 'noida' or 'gr' or 'greater' or 'yamuna' or 'dadri' or 'ansal'or 'amrapali'or 'vihar' or 'amarpali' in word.lower(),
        'block': 'block' in word.lower(),
        'delhi': 'delhi' or 'gurgaon' or 'faridabad'in word.lower(),
    }
    
    if i > 0:
        word1 = sent[i-1][0]
        postag1 = sent[i-1][1]
        features.update({
                   '-1:word1.lower()': word1.lower(),
        '-1:word1[-3:]': word1[-3:],
        '-1:word1[-2:]': word1[-2:],
        '-1:word1.isupper()': word1.isupper(),
        '-1:word1.istitle()': word1.istitle(),
        '-1:word1.isdigit()': word1.isdigit(),
        '-1:postag1': postag1,
        '-1:postag1[:2]': postag1[:2],
        '-1:--bhk': 'bhk' in word1.lower(),
        '-1:#...':word1[0]=="#",
        '-1:@...': word1[0]=='@',
        '-1:+...': word1[0]=='+',
        '-1:0...': word1[0]=='0',
        '-1:A': word1 in data_A,
        '-1:L':word1 in data_L,
        '-1:LA':word1 in data_LA,
        '-1:N':word1 in data_N,
        '-1:C_':re.match('[0-9]\-[A-Za-z]+', word1) is not None,
        '-1:sqft': 'sq' in word1.lower() or 'ft' in word1.lower() or 'mtr' in word1.lower() or 'meter' in word1.lower() or 'yard' in word1.lower() or 'sqft' in word1.lower(),
        '-1:\sq': '/sq' in word1.lower(),
        '-1:punkt':word1 in set(string.punctuation),
        "-1:cr": 'cr' in word1.lower(),
        '-1:lac': 'lac' in word1.lower(),
        '-1:lakh' :'lakh' in word1.lower(),
        "-1:crore": 'crore' in word1.lower(),
        '-1:per': 'per' in word1,
        '-1:tel-number': re.match('[0-9]{3}-[0-9]{4}-[0-9]{3}', word1.lower()) is not None,
        '-1:tel-number-2': re.match('[0-9]{10}', word1.lower()) is not None,
        "-1:/-" :  '/-' in word1,
        '-1:Sector': "sec" in word1.lower()
            
        })
    

    if i < len(sent)-1:
        word1 = sent[i+1][0]
        postag1 = sent[i+1][1]
        features.update({
                    '+1:word1.lower()': word1.lower(),
        '+1:word1[-3:]': word1[-3:],
        '+1:word1[-2:]': word1[-2:],
        '+1:word1.isupper()': word1.isupper(),
        '+1:word1.istitle()': word1.istitle(),
        '+1:word1.isdigit()': word1.isdigit(),
        '+1:postag1': postag1,
        '+1:postag1[:2]': postag1[:2],
        '+1:--bhk': 'bhk' in word1.lower(),
        '+1:#...':word1[0]=="#",
        '+1:@...': word1[0]=='@',
        '+1:+...': word1[0]=='+',
        '+1:0...': word1[0]=='0',
        '+1:A': word1 in data_A,
        '+1:L':word1 in data_L,
        '+1:LA':word1 in data_LA,
        '+1:N':word1 in data_N,
        '+1:C_':re.match('[0-9]\-[A-Za-z]+', word1) is not None,
        '+1:sqft': 'sq' in word1.lower() or 'ft' in word1.lower() or 'mtr' in word1.lower() or 'meter' in word1.lower() or 'yard' in word1.lower() or 'sqft' in word1.lower(),
        '+1:\sq': '/sq' in word1.lower(),
        '+1:punkt':word1 in set(string.punctuation),
        "+1:cr": 'cr' in word1.lower(),
        '+1:lac': 'lac' in word1.lower(),
        '+1:lakh' :'lakh' in word1.lower(),
        "+1:crore": 'crore' in word1.lower(),
        '+1:per': 'per' in word1,
        '+1:tel-number': re.match('[0-9]{3}-[0-9]{4}-[0-9]{3}', word1.lower()) is not None,
        '+1:tel-number-2': re.match('[0-9]{10}', word1.lower()) is not None,
        "+1:/-" :  '/-' in word1,
        '+1:Sector': "sec" in word1.lower()

        })

    return features


def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

def sent2labels(sent):
    return [label for token, postag, label in sent]

def sent2tokens(sent):
    return [token for token, postag, label in sent]

print("Loading model")
crf = pickle.load(open('model.pkl','rb'))

print("making features")
x = [sent2features(s) for s in test_data]
print("making predictions")
y_pred = crf.predict(x)
print("writing to file")
with open(output_file, 'w') as f:
    for i,sent in enumerate(test_data):
        for j,words in enumerate(sent):
            f.write(words[0]+" "+y_pred[i][j]+"\n")
        f.write("\n")
