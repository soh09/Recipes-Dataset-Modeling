# How Good Are The Recipes?

A Data Science Project For DSC 80 At The University of California, San Diego

by So Hirota (hirotaso92602@gmail.com)

Published 3/20/2023

Part 1 (EDA) can be found here: https://soh09.github.io/Recipes-Dataset-Analysis/

------

## Framing the Problem

In Part 1 of this project, I performed EDA in addition to some permutation and hypothesis testing on the recipes and interactions dataset. In this part of the project, I will further explore the data and create a machine learning model to predict the average rating of a recipe.

#### Can we predict the average rating of a recipe?

To answer this question,
1. I'll make a **regression** model
2. This regression model will predict the average rating of a certain recipe, given a few features from the recipes and interactions dataset.
3. This regression model will be evaluted using the **R^2** score. I chose this metric because R^2 is an easily interpretable metric that tells me the proportion (from 0.0 to 1.0) of variation in the data that the model is able to explain. 1.0 would means that model's predictions are identitcal to actual response variable. 

### Recap: The Data
The recipes dataset contains two .csv files: the RAW_recpies and the RAW_interactions dataset.

RAW_recipes.csv contains `83782 rows` and `12 columns`. The rows represent the recipes, and the columns contain `name`, `id`, `minutes`, `contributor_id`, `submitted`, `tags`, `nutrition`, `n_steps`, `steps`, `description`, `ingredients`, `n_ingredients`. `nutrition` is in "Percentage Daily Value (PDV)" besides `calories (#)`, which is kilocalories. 

| column name | meaning |
|-----|-----|
| `name` | the name of the recipe |
| `id` | the id of the recipe |
| `minutes` | the time it takes to make the recipe |
| `contributor_id` | the id of the recipe contributor |
| `submitted` | the date the recipe was submitted, in YY-MM-DD format |
| `tags` | the tags associated with the recipe |
| `nutrition` | nutritional information, in order of calories (#), total fat (PDV), sugar (PDV), sodium (PDV), protein (PDV), saturated fat (PDV), carbohydrates (PDV) |
| `n_steps` | the number of steps the recipe requires |
| `steps` | the descriptions of each step |
| `description` | the description of the recipe |
| `ingredients` | the ingredients of the recipe | 
| `n_ingredients` | the number of ingredients required to make the recipe | 

RAW_interactions.csv contains `731927 rows` and `5 columns`. The rows represent an individual review of a recipe, and the columns contain `user_id`, `recipe_id`, `date`, `rating`, `review`.

| columns name | meaning |
|-----|-----|
| `user_id` | user id of the user who posted a review |
| `recipe_id` | recipe id for the review, same as the ones in RAW_recipes.csv |
| `date` | the date that the reivew was posted |
| `rating` | the star rating of the recipe, from 1 - 5 |
| `review` | the text review of the recipe |

### The Question

### Preparing the Data
- Recipes data
    1. Converting "fake lists" from strings to actual lists
    1. Breaking up the nutrition column into its individual components
    2. Dropping `name`, `contributor_id`, `steps`, `submitted`, and `description`. I'm keeping tags because maybe I could use them for feature engineering down the line.
    3. Adding the average ratings column, which is derived from the interactions data. This will be the response variable (y) for the model.
    4. Adding a column that contains all the review comments stored as a list for each recipe
- Interactions data
    1. Drop rows with nan in `review` column

Much of the code used for this was from Part 1 of the project, which I have linked at the top.

After data cleaning, the two dataframe looks like this.

```python
>>> recipes.shape
(9707, 13)
```
```python
>>> recipes.head(2)
```

| id     | minutes | tags                   | n_steps | n_ingredients | calories (#) | total fat (%) | sugar (%) | sodium (%) | protein (%) | sat fats (%) | carbs (%) | rating      | review                    |
|:-------|:--------|:-----------------------|:--------|:--------------|:-------------|:--------------|:----------|:-----------|:------------|:-------------|:----------|:------------|:--------------------------|
| 333281 | 75      | ['time-to-make', 'cours| 6       | 9             | 1582.6	     |   88.0        |  402.0    |  27.0      |  96.0       |  156.0       |  73.0     |  4.400000   |[Loved it and will make    |
| 453467 | 5       | ['15-minutes-or-less', | 2       | 11            | 94.7         |   0.0         |  70.0     |  0.0       |  2.0        |  0.0         |  7.0      |  4.800000   |[Love the anise and orange |

```python
>>> interactions.shape
(375987, 3)
```
```python
>>> interactions.head(2)
```

|   recipe_id |   rating | review                                                   |
|------------:|---------:|:---------------------------------------------------------|
|       79222 |        4 | Oh, how wonderful!  I doubled the crab, and added some   |
|       79222 |        5 | Along with the onions we added in a square of salt pork, |
|       79222 |        4 | I made this last nite and it was pretty good.  I will    |



## The Baseline Model
The mode will be as described below.

- Features: 10 quantitative/numerical features
    1. minutes
    2. n_steps, n_ingredients
    3. all of the nutrition columns
- Response variable: the rating column
- No encoding is necessary at this stage, since all the input features will be numerical
- Transformation
    - I won't use any transformations at this point

In order to test the model's capacity to generalize to unseen data, I performed a train test split and trained the model only on the training data.

```python
>>> model.score(X_test, y_test)
0.0038402175285358053
```

### Baseline Model: Result

R^2 score is consistently close to 0.

This word is a very poor model because the r^2 is very close to 0. This means the model is unable to explain any of the variance present in the data. I think that this model has a very low R^2 score because the features that are fed into it are not correlated the rating very well. I will try to transform columns to create these relationships for the final model.


## Final Model

Findings and Possible Improvements
- The nutrition columns (heatmap can be seen below)
    1. calories has high correlation with total fat, saturated fats, and carbs. Perhaps, if we include calories, saturated fats and carbs column are redundant.
    2. sodium has a low correlation with all columns
    3. sugar could also be included because it has relativly low correlation with other columns besides carbs
    4. To simplify the mode, we can probably drop the other nutrition labels

<iframe src = 'assets/heatmap.png' width = 800 height = 800 frameborder = 0> </iframe>

- Incorporating data from the interactions dataset
    1. Create three columns, each corresponding to the number of good, neutral, and bad reviews for a recipe
        - I will derive this feature by performing sentiment analysis on the reviews. The two aforementioned columns will contain the number of positive and negative reviews, respectively.

Changes I am making between the baseline and final mode
1. Of the nutrition columns, I will only keep `calories`, `sodium`, and `sugar`. This is because the other nutrition columns were highly correlated to other columns. By dropping these columns, we can prevent multicolinarity. While this change may not necessarily improve the R^2 score of the mode, it will reducing the dimensionality and make the model less complex. This change will most likely not negatively impact the R^2 at the very least.
2. I will add three columns, where each encode the number of good, neutral, and bad reviews for a recipe. This should provide more information to the model, and help it make better predictions about the average rating of the recipe. I think these features would improve the accuracy because the review comment should overall reflect the rating that that the individual provided the recipe with. Average rating is a number that is derived from the ratings that these individuals have provided, so the number of good, neutral, and bad reviews should serve as a good indicator for how highly rated a recipe is.

Models
1. Linear Regression model
    - Rationale: if there are any simple linear relationships that the model can pick up on, then the R^2 score could be pretty high. The sentiment analysis columns are good candidates for linear relationships.
    - Hyperparamters: combinations of standarizing minutes, n_ingredients, n_steps, and calories
2. K-Nearest Neighbor Regression model
    - Rationale: since rating is a categorical variable in some sense, then it could make sense to use a KNN algorithm to figure out the "clusters" (which are the rating categories, like 1 star, 2 stars, etc). This could maybe lead to accurate predictions.
    - hyperparamters: n_neighbors, p (how distance is calculated)
3. Decision Tree Regressor model
    - Rationale: Individuals giving the recipe a ratings is a decision that a user has to make, so perhaps a decision tree model will be able to capture that nuance well.
    - hyperparameters: depth, mininimum sample splits, scoring criterion

