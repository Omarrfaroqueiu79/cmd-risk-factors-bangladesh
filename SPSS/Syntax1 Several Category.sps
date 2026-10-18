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
* Create Multiple-Category CMD Severity Variable.
*======================================================.

COMPUTE CMD_Severity = 0.

* Mild CMD.
IF ((D_Score >= 10 AND D_Score <= 13) OR
    (A_Score >= 8 AND A_Score <= 9))
CMD_Severity = 1.

* Moderate CMD.
IF ((D_Score >= 14 AND D_Score <= 20) OR
    (A_Score >= 10 AND A_Score <= 14))
CMD_Severity = 2.

* Severe CMD.
IF ((D_Score >= 21) OR
    (A_Score >= 15))
CMD_Severity = 3.

VARIABLE LABELS CMD_Severity
'Common Mental Disorder Severity'.

VALUE LABELS CMD_Severity
0 'No CMD'
1 'Mild CMD'
2 'Moderate CMD'
3 'Severe CMD'.

EXECUTE.
