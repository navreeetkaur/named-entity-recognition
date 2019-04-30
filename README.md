# Named-entity-recognition
### Assignment 2 - COL772(Spring'19): Natural Language Processing
#### Creator: Navreet Kaur[2015TT10917]
 
#### Motivation:
The motivation of this assignment is to get practice with sequence labeling tasks such as Named Entity Recognition. More precisely you will experiment with the BiLSTM-CRF model and various features on real estate text.

#### Scenario: 
Different real estate agents share noisy text messages on a real estate platform to inform buyers about new properties available for sale. We will call these text messages, shouts. As a company interested in automating real estate information, our first step is to perform NER on these shouts so that important and relevant information can be extracted downstream.

#### Problem Statement: 
The goal of the assignment is to build an NER system for shouts. The input of the code will be a set of tokenized shouts and the output will be a label for each token in the sentence. Labels will be from 8 classes:
- Locality (L)
- Total Price (P)
- Land Area (LA)
- Cost per land area (C)
- Contact name (N)
- Contact telephone (T)
- Attributes of the property (A)
- Other (O)

#### The Task: 
You need to write a sequence tagger that labels the given shouts in a tokenized test file. The tokenized test file follows the same format as training except that it does not have the final label in the input. Your output should label the test file in the same format as the training data.
