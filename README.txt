# Digit Recognition

A kNN machine learning project, that trains the data before using it.

Only works wit text-file inputs 
 - 32 rows X 32 chars. 
 - for each char 0 is white and 1 is black
 - find examples in digits/testDigits
 - the fileName must start with *_ where * is the digit drawn in the file

The train.train() function will merge the files in digits/testDigits

Running run.py will select 200 random input files from digits/testDigits and will try to guess what is drawn on them
It will write it's acuracy and point to the found mistakes.

Example output:

> Num #0 : 100% accuracy
Num #1 : 100% accuracy
Num #2 : 90% accuracy
Num #3 : 64% accuracy
Num #4 : 86% accuracy
Num #5 : 95% accuracy
Num #6 : 100% accuracy
Num #7 : 100% accuracy
Num #8 : 95% accuracy
Num #9 : 92% accuracy

> 13/200 mistakes were found.
Total Accuracy : 93.5 %

> MISTAKES:0 : []
1 : []
2 : [9, 9]
3 : [9, 8, 9, 5, 8]
4 : [7, 1, 7]
5 : [9]
6 : []
7 : []
8 : [6]
9 : [7]

