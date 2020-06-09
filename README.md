# Wine Classifier

Wine dataset: https://www.kaggle.com/dell4010/wine-dataset/data


```python
import os
import pandas as pd 

data = pd.read_csv(os.path.join('Data','wine_dataset.csv'))

```


```python
data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>fixed_acidity</th>
      <th>volatile_acidity</th>
      <th>citric_acid</th>
      <th>residual_sugar</th>
      <th>chlorides</th>
      <th>free_sulfur_dioxide</th>
      <th>total_sulfur_dioxide</th>
      <th>density</th>
      <th>pH</th>
      <th>sulphates</th>
      <th>alcohol</th>
      <th>quality</th>
      <th>style</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>7.4</td>
      <td>0.70</td>
      <td>0.00</td>
      <td>1.9</td>
      <td>0.076</td>
      <td>11.0</td>
      <td>34.0</td>
      <td>0.9978</td>
      <td>3.51</td>
      <td>0.56</td>
      <td>9.4</td>
      <td>5</td>
      <td>red</td>
    </tr>
    <tr>
      <th>1</th>
      <td>7.8</td>
      <td>0.88</td>
      <td>0.00</td>
      <td>2.6</td>
      <td>0.098</td>
      <td>25.0</td>
      <td>67.0</td>
      <td>0.9968</td>
      <td>3.20</td>
      <td>0.68</td>
      <td>9.8</td>
      <td>5</td>
      <td>red</td>
    </tr>
    <tr>
      <th>2</th>
      <td>7.8</td>
      <td>0.76</td>
      <td>0.04</td>
      <td>2.3</td>
      <td>0.092</td>
      <td>15.0</td>
      <td>54.0</td>
      <td>0.9970</td>
      <td>3.26</td>
      <td>0.65</td>
      <td>9.8</td>
      <td>5</td>
      <td>red</td>
    </tr>
    <tr>
      <th>3</th>
      <td>11.2</td>
      <td>0.28</td>
      <td>0.56</td>
      <td>1.9</td>
      <td>0.075</td>
      <td>17.0</td>
      <td>60.0</td>
      <td>0.9980</td>
      <td>3.16</td>
      <td>0.58</td>
      <td>9.8</td>
      <td>6</td>
      <td>red</td>
    </tr>
    <tr>
      <th>4</th>
      <td>7.4</td>
      <td>0.70</td>
      <td>0.00</td>
      <td>1.9</td>
      <td>0.076</td>
      <td>11.0</td>
      <td>34.0</td>
      <td>0.9978</td>
      <td>3.51</td>
      <td>0.56</td>
      <td>9.4</td>
      <td>5</td>
      <td>red</td>
    </tr>
  </tbody>
</table>
</div>




```python
data['style'] = data['style'].replace('red', 0)
```


```python
data['style'] = data['style'].replace('white', 1)
```


```python
# separando entre alvo  Y = style e X = tudo menos a coluna Style
# axis = 1 operação nas colunas

y = data['style']
x = data.drop('style', axis = 1)
```


```python
from sklearn.model_selection import train_test_split

#cria conjunto de dados para treino e para teste
x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size = 0.3)

#test_size = 0.3 -> 30% do conjunto de dados vai ser dedicado para testes 

#train_test_split() -> 4 saidas, 2 para treino 2 para teste 
```

## Decision tree



```python
from sklearn.ensemble import ExtraTreesClassifier

modelo = ExtraTreesClassifier()
modelo.fit(x_treino, y_treino)

resultado = modelo.score(x_teste, y_teste)
print ("Acurácia:", resultado )
```

    Acurácia: 0.9964102564102564


# Prevendo


```python
previsoes = modelo.predict(x_teste[300:305])
previsoes
```




    array([0, 1, 1, 0, 0])




```python
y_teste[300:305]
```




    702     0
    4438    1
    6436    1
    1039    0
    253     0
    Name: style, dtype: int64




```python
x_teste[300:305]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>fixed_acidity</th>
      <th>volatile_acidity</th>
      <th>citric_acid</th>
      <th>residual_sugar</th>
      <th>chlorides</th>
      <th>free_sulfur_dioxide</th>
      <th>total_sulfur_dioxide</th>
      <th>density</th>
      <th>pH</th>
      <th>sulphates</th>
      <th>alcohol</th>
      <th>quality</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>702</th>
      <td>7.0</td>
      <td>0.640</td>
      <td>0.02</td>
      <td>2.1</td>
      <td>0.067</td>
      <td>9.0</td>
      <td>23.0</td>
      <td>0.99700</td>
      <td>3.47</td>
      <td>0.67</td>
      <td>9.4</td>
      <td>6</td>
    </tr>
    <tr>
      <th>4438</th>
      <td>5.8</td>
      <td>0.220</td>
      <td>0.29</td>
      <td>0.9</td>
      <td>0.034</td>
      <td>34.0</td>
      <td>89.0</td>
      <td>0.98936</td>
      <td>3.14</td>
      <td>0.36</td>
      <td>11.1</td>
      <td>7</td>
    </tr>
    <tr>
      <th>6436</th>
      <td>6.5</td>
      <td>0.280</td>
      <td>0.38</td>
      <td>7.8</td>
      <td>0.031</td>
      <td>54.0</td>
      <td>216.0</td>
      <td>0.99154</td>
      <td>3.03</td>
      <td>0.42</td>
      <td>13.1</td>
      <td>6</td>
    </tr>
    <tr>
      <th>1039</th>
      <td>8.9</td>
      <td>0.500</td>
      <td>0.21</td>
      <td>2.2</td>
      <td>0.088</td>
      <td>21.0</td>
      <td>39.0</td>
      <td>0.99692</td>
      <td>3.33</td>
      <td>0.83</td>
      <td>11.1</td>
      <td>6</td>
    </tr>
    <tr>
      <th>253</th>
      <td>7.7</td>
      <td>0.775</td>
      <td>0.42</td>
      <td>1.9</td>
      <td>0.092</td>
      <td>8.0</td>
      <td>86.0</td>
      <td>0.99590</td>
      <td>3.23</td>
      <td>0.59</td>
      <td>9.5</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>




```python

```
