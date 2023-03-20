# How Good Are The Recipes?

A Data Science Project For DSC 80 At The University of California, San Diego

by So Hirota (hirotaso92602@gmail.com)

Published 3/20/2023

Part 1 (EDA) can be found here: https://soh09.github.io/Recipes-Dataset-Analysis/

------

## Framing the Problem

In Part 1 of this project, I performed EDA in addition to some permutation and hypothesis testing on the recipes and interactions dataset. In this part of the project, I will further explore the data and create a machine learning model to predict the average rating of a recipe.

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

#### Can we predict the average rating of a recipe?

To answer this question,
1. I'll make a **regression** model
2. This regression model will predict the average rating of a certain recipe, given a few features from the recipes and interactions dataset.
3. This regression model will be evaluted using the **R^2** score. I chose this metric because R^2 is an easily interpretable metric that tells me the proportion (from 0.0 to 1.0) of variation in the data that the model is able to explain. 1.0 would means that model's predictions are identitcal to actual response variable. 

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



