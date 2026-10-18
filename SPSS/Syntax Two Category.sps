* Encoding: UTF-8.
*======================================================.
* Calculate DASS-21 Scores (Adjusted by multiplying ×2).
*======================================================.

COMPUTE D_Score = SUM(DASS21_d1,DASS21_d2,DASS21_d3,
                      DASS21_d4,DASS21_d5,DASS21_d6,DASS21_d7) * 2.

COMPUTE A_Score = SUM(DASS21_a1,DASS21_a2,DASS21_a3,
                      DASS21_a4,DASS21_a5,DASS21_a6,DASS21_a7) * 2.

VARIABLE LABELS
D_Score 'DASS-21 Depression Score'
A_Score 'DASS-21 Anxiety Score'.

EXECUTE.


*======================================================.
* Create CMD Variable.
* CMD = 1 if Depression >=10 OR Anxiety >=8.
*======================================================.

COMPUTE CMD = 0.

IF (D_Score >= 10 OR A_Score >= 8) CMD = 1.

VARIABLE LABELS CMD
'Common Mental Disorder (0=No, 1=Yes)'.

VALUE LABELS CMD
0 'No CMD'
1 'CMD Present'.

EXECUTE.
