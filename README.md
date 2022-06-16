# LinearRegression
Residen is a copy of the Residential-Building-Dataset, the variables names have been changed for use in R. The excel file also contains a description of the variables. More information on the dataset can be obtained on the UCI webpage: https://archive.ics.uci.edu/ml/datasets/Residential+Building+Data+Set 

 Fit a linear regression model to explain the ”actual sales price”
(V104) in terms of other variables (excluding the variable ”actual
construction costs” (V105)) using backwards selection, stepwise selection, Ridge regression and LASSO regression.

# Result
![result](https://user-images.githubusercontent.com/107531850/173973728-be6cdf7c-7298-4e92-b46b-0ccf1687e760.PNG)

Overall, Ridge regression outperforms than LASSO regression, backwards linear regression and stepwise linear regression. Ridge regression is the parsimonious model with less explanatory variables, smaller test MSE.
