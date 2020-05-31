# Spectrum Data Science and Machine Learning Internship Project Report
## Student Grade Prediction Machine Learning using Linear regression, Decision tree and random Forest

In this internship, I have done a project based on the student’s data and their previous math-marks.
I imported different machine learning libraries like:

    1. numpy,

    2. pandas,

    3. sklearn,

    4. statsmodel.api,

    5. matplotlib,

    6. seaborn. etc..

I have added a new column by adding 'G1','G2','G3' column. After that, I pre-processed the data in a dataframe , and changed the binary attributes into 0 or 1, encoded the nominal values. Then I have used various Machine Learning models and plotted the various graphs.

In Linear Regression I used Ordinary Least Square(OLS) method for Backward Elimination of the attributes to find out which attributes affected the most.

## Various models used:
#### 1. Linear Regression
##### Result:
    Train Score:  0.9781480282081054
    Test Score:  0.9601885694498354
    Predict Score:  0.9976196999704539
##### Plot:
![Leaner Regression](https://github.com/svjenar/Spectrum/blob/master/task3_final/linear.png)
#### 2. Decision Tree
##### Result:
    Train Score:  1.0
    Test Score:  0.94715
    Predict Score:  1.0
##### Plot:
![Decision Tree](https://github.com/svjenar/Spectrum/blob/master/task3_final/dtree.png)
#### 3. Random Forest Regression
##### Result:
    Train Score:  0.9747949479943939
    Test Score:  0.9624482395133013
    Predict Score:  0.9977051501448246

![Random Forest](https://github.com/svjenar/Spectrum/blob/master/task3_final/randomforest.png)

#### Other plots are:
![plot1](https://github.com/svjenar/Spectrum/blob/master/task3_final/G1Vsfinal_grade.png)
![plot2](https://github.com/svjenar/Spectrum/blob/master/task3_final/G2Vsfinal_grade.png)
![plot3](https://github.com/svjenar/Spectrum/blob/master/task3_final/StudytimeVsfinal_grade(sex).png)
![plot4](https://github.com/svjenar/Spectrum/blob/master/task3_final/absenceVsfinal_grade.png)
![plot5](https://github.com/svjenar/Spectrum/blob/master/task3_final/ageVsfinal_grade(sex).png)



### Submitted By:
#### Name: Sidhartha Bibekananda Dash
#### Regd No: 1701106295
