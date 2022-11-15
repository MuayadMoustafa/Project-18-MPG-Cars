# MPG Cars

### Introduction:

The following exercise utilizes data from [UC Irvine Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Auto+MPG)

### Step 1. Import the necessary libraries


```python
import pandas as pd 
import numpy as np 
```

### Step 2. Import the first dataset [cars1](https://raw.githubusercontent.com/guipsamora/pandas_exercises/master/05_Merge/Auto_MPG/cars1.csv) and [cars2](https://raw.githubusercontent.com/guipsamora/pandas_exercises/master/05_Merge/Auto_MPG/cars2.csv).  

   ### Step 3. Assign each to a variable called cars1 and cars2


```python
cars1 = pd.read_csv("https://raw.githubusercontent.com/guipsamora/pandas_exercises/master/05_Merge/Auto_MPG/cars1.csv")
cars1
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
      <th>mpg</th>
      <th>cylinders</th>
      <th>displacement</th>
      <th>horsepower</th>
      <th>weight</th>
      <th>acceleration</th>
      <th>model</th>
      <th>origin</th>
      <th>car</th>
      <th>Unnamed: 9</th>
      <th>Unnamed: 10</th>
      <th>Unnamed: 11</th>
      <th>Unnamed: 12</th>
      <th>Unnamed: 13</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>18.0</td>
      <td>8</td>
      <td>307</td>
      <td>130</td>
      <td>3504</td>
      <td>12.0</td>
      <td>70</td>
      <td>1</td>
      <td>chevrolet chevelle malibu</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>15.0</td>
      <td>8</td>
      <td>350</td>
      <td>165</td>
      <td>3693</td>
      <td>11.5</td>
      <td>70</td>
      <td>1</td>
      <td>buick skylark 320</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>18.0</td>
      <td>8</td>
      <td>318</td>
      <td>150</td>
      <td>3436</td>
      <td>11.0</td>
      <td>70</td>
      <td>1</td>
      <td>plymouth satellite</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>16.0</td>
      <td>8</td>
      <td>304</td>
      <td>150</td>
      <td>3433</td>
      <td>12.0</td>
      <td>70</td>
      <td>1</td>
      <td>amc rebel sst</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>17.0</td>
      <td>8</td>
      <td>302</td>
      <td>140</td>
      <td>3449</td>
      <td>10.5</td>
      <td>70</td>
      <td>1</td>
      <td>ford torino</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>193</th>
      <td>24.0</td>
      <td>6</td>
      <td>200</td>
      <td>81</td>
      <td>3012</td>
      <td>17.6</td>
      <td>76</td>
      <td>1</td>
      <td>ford maverick</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>194</th>
      <td>22.5</td>
      <td>6</td>
      <td>232</td>
      <td>90</td>
      <td>3085</td>
      <td>17.6</td>
      <td>76</td>
      <td>1</td>
      <td>amc hornet</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>195</th>
      <td>29.0</td>
      <td>4</td>
      <td>85</td>
      <td>52</td>
      <td>2035</td>
      <td>22.2</td>
      <td>76</td>
      <td>1</td>
      <td>chevrolet chevette</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>196</th>
      <td>24.5</td>
      <td>4</td>
      <td>98</td>
      <td>60</td>
      <td>2164</td>
      <td>22.1</td>
      <td>76</td>
      <td>1</td>
      <td>chevrolet woody</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>197</th>
      <td>29.0</td>
      <td>4</td>
      <td>90</td>
      <td>70</td>
      <td>1937</td>
      <td>14.2</td>
      <td>76</td>
      <td>2</td>
      <td>vw rabbit</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>198 rows × 14 columns</p>
</div>




```python
cars2 = pd.read_csv("https://raw.githubusercontent.com/guipsamora/pandas_exercises/master/05_Merge/Auto_MPG/cars2.csv")
cars2
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
      <th>mpg</th>
      <th>cylinders</th>
      <th>displacement</th>
      <th>horsepower</th>
      <th>weight</th>
      <th>acceleration</th>
      <th>model</th>
      <th>origin</th>
      <th>car</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>33.0</td>
      <td>4</td>
      <td>91</td>
      <td>53</td>
      <td>1795</td>
      <td>17.4</td>
      <td>76</td>
      <td>3</td>
      <td>honda civic</td>
    </tr>
    <tr>
      <th>1</th>
      <td>20.0</td>
      <td>6</td>
      <td>225</td>
      <td>100</td>
      <td>3651</td>
      <td>17.7</td>
      <td>76</td>
      <td>1</td>
      <td>dodge aspen se</td>
    </tr>
    <tr>
      <th>2</th>
      <td>18.0</td>
      <td>6</td>
      <td>250</td>
      <td>78</td>
      <td>3574</td>
      <td>21.0</td>
      <td>76</td>
      <td>1</td>
      <td>ford granada ghia</td>
    </tr>
    <tr>
      <th>3</th>
      <td>18.5</td>
      <td>6</td>
      <td>250</td>
      <td>110</td>
      <td>3645</td>
      <td>16.2</td>
      <td>76</td>
      <td>1</td>
      <td>pontiac ventura sj</td>
    </tr>
    <tr>
      <th>4</th>
      <td>17.5</td>
      <td>6</td>
      <td>258</td>
      <td>95</td>
      <td>3193</td>
      <td>17.8</td>
      <td>76</td>
      <td>1</td>
      <td>amc pacer d/l</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>195</th>
      <td>27.0</td>
      <td>4</td>
      <td>140</td>
      <td>86</td>
      <td>2790</td>
      <td>15.6</td>
      <td>82</td>
      <td>1</td>
      <td>ford mustang gl</td>
    </tr>
    <tr>
      <th>196</th>
      <td>44.0</td>
      <td>4</td>
      <td>97</td>
      <td>52</td>
      <td>2130</td>
      <td>24.6</td>
      <td>82</td>
      <td>2</td>
      <td>vw pickup</td>
    </tr>
    <tr>
      <th>197</th>
      <td>32.0</td>
      <td>4</td>
      <td>135</td>
      <td>84</td>
      <td>2295</td>
      <td>11.6</td>
      <td>82</td>
      <td>1</td>
      <td>dodge rampage</td>
    </tr>
    <tr>
      <th>198</th>
      <td>28.0</td>
      <td>4</td>
      <td>120</td>
      <td>79</td>
      <td>2625</td>
      <td>18.6</td>
      <td>82</td>
      <td>1</td>
      <td>ford ranger</td>
    </tr>
    <tr>
      <th>199</th>
      <td>31.0</td>
      <td>4</td>
      <td>119</td>
      <td>82</td>
      <td>2720</td>
      <td>19.4</td>
      <td>82</td>
      <td>1</td>
      <td>chevy s-10</td>
    </tr>
  </tbody>
</table>
<p>200 rows × 9 columns</p>
</div>



### Step 4. Oops, it seems our first dataset has some unnamed blank columns, fix cars1


```python
cars1 = cars1.dropna(axis=1)
cars1
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
      <th>mpg</th>
      <th>cylinders</th>
      <th>displacement</th>
      <th>horsepower</th>
      <th>weight</th>
      <th>acceleration</th>
      <th>model</th>
      <th>origin</th>
      <th>car</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>18.0</td>
      <td>8</td>
      <td>307</td>
      <td>130</td>
      <td>3504</td>
      <td>12.0</td>
      <td>70</td>
      <td>1</td>
      <td>chevrolet chevelle malibu</td>
    </tr>
    <tr>
      <th>1</th>
      <td>15.0</td>
      <td>8</td>
      <td>350</td>
      <td>165</td>
      <td>3693</td>
      <td>11.5</td>
      <td>70</td>
      <td>1</td>
      <td>buick skylark 320</td>
    </tr>
    <tr>
      <th>2</th>
      <td>18.0</td>
      <td>8</td>
      <td>318</td>
      <td>150</td>
      <td>3436</td>
      <td>11.0</td>
      <td>70</td>
      <td>1</td>
      <td>plymouth satellite</td>
    </tr>
    <tr>
      <th>3</th>
      <td>16.0</td>
      <td>8</td>
      <td>304</td>
      <td>150</td>
      <td>3433</td>
      <td>12.0</td>
      <td>70</td>
      <td>1</td>
      <td>amc rebel sst</td>
    </tr>
    <tr>
      <th>4</th>
      <td>17.0</td>
      <td>8</td>
      <td>302</td>
      <td>140</td>
      <td>3449</td>
      <td>10.5</td>
      <td>70</td>
      <td>1</td>
      <td>ford torino</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>193</th>
      <td>24.0</td>
      <td>6</td>
      <td>200</td>
      <td>81</td>
      <td>3012</td>
      <td>17.6</td>
      <td>76</td>
      <td>1</td>
      <td>ford maverick</td>
    </tr>
    <tr>
      <th>194</th>
      <td>22.5</td>
      <td>6</td>
      <td>232</td>
      <td>90</td>
      <td>3085</td>
      <td>17.6</td>
      <td>76</td>
      <td>1</td>
      <td>amc hornet</td>
    </tr>
    <tr>
      <th>195</th>
      <td>29.0</td>
      <td>4</td>
      <td>85</td>
      <td>52</td>
      <td>2035</td>
      <td>22.2</td>
      <td>76</td>
      <td>1</td>
      <td>chevrolet chevette</td>
    </tr>
    <tr>
      <th>196</th>
      <td>24.5</td>
      <td>4</td>
      <td>98</td>
      <td>60</td>
      <td>2164</td>
      <td>22.1</td>
      <td>76</td>
      <td>1</td>
      <td>chevrolet woody</td>
    </tr>
    <tr>
      <th>197</th>
      <td>29.0</td>
      <td>4</td>
      <td>90</td>
      <td>70</td>
      <td>1937</td>
      <td>14.2</td>
      <td>76</td>
      <td>2</td>
      <td>vw rabbit</td>
    </tr>
  </tbody>
</table>
<p>198 rows × 9 columns</p>
</div>



### Step 5. What is the number of observations in each dataset?


```python
cars1.shape[0],cars2.shape[0]
```




    (198, 200)



### Step 6. Join cars1 and cars2 into a single DataFrame called cars


```python
car = pd.concat([cars1,cars2])
car 
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
      <th>mpg</th>
      <th>cylinders</th>
      <th>displacement</th>
      <th>horsepower</th>
      <th>weight</th>
      <th>acceleration</th>
      <th>model</th>
      <th>origin</th>
      <th>car</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>18.0</td>
      <td>8</td>
      <td>307</td>
      <td>130</td>
      <td>3504</td>
      <td>12.0</td>
      <td>70</td>
      <td>1</td>
      <td>chevrolet chevelle malibu</td>
    </tr>
    <tr>
      <th>1</th>
      <td>15.0</td>
      <td>8</td>
      <td>350</td>
      <td>165</td>
      <td>3693</td>
      <td>11.5</td>
      <td>70</td>
      <td>1</td>
      <td>buick skylark 320</td>
    </tr>
    <tr>
      <th>2</th>
      <td>18.0</td>
      <td>8</td>
      <td>318</td>
      <td>150</td>
      <td>3436</td>
      <td>11.0</td>
      <td>70</td>
      <td>1</td>
      <td>plymouth satellite</td>
    </tr>
    <tr>
      <th>3</th>
      <td>16.0</td>
      <td>8</td>
      <td>304</td>
      <td>150</td>
      <td>3433</td>
      <td>12.0</td>
      <td>70</td>
      <td>1</td>
      <td>amc rebel sst</td>
    </tr>
    <tr>
      <th>4</th>
      <td>17.0</td>
      <td>8</td>
      <td>302</td>
      <td>140</td>
      <td>3449</td>
      <td>10.5</td>
      <td>70</td>
      <td>1</td>
      <td>ford torino</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>195</th>
      <td>27.0</td>
      <td>4</td>
      <td>140</td>
      <td>86</td>
      <td>2790</td>
      <td>15.6</td>
      <td>82</td>
      <td>1</td>
      <td>ford mustang gl</td>
    </tr>
    <tr>
      <th>196</th>
      <td>44.0</td>
      <td>4</td>
      <td>97</td>
      <td>52</td>
      <td>2130</td>
      <td>24.6</td>
      <td>82</td>
      <td>2</td>
      <td>vw pickup</td>
    </tr>
    <tr>
      <th>197</th>
      <td>32.0</td>
      <td>4</td>
      <td>135</td>
      <td>84</td>
      <td>2295</td>
      <td>11.6</td>
      <td>82</td>
      <td>1</td>
      <td>dodge rampage</td>
    </tr>
    <tr>
      <th>198</th>
      <td>28.0</td>
      <td>4</td>
      <td>120</td>
      <td>79</td>
      <td>2625</td>
      <td>18.6</td>
      <td>82</td>
      <td>1</td>
      <td>ford ranger</td>
    </tr>
    <tr>
      <th>199</th>
      <td>31.0</td>
      <td>4</td>
      <td>119</td>
      <td>82</td>
      <td>2720</td>
      <td>19.4</td>
      <td>82</td>
      <td>1</td>
      <td>chevy s-10</td>
    </tr>
  </tbody>
</table>
<p>398 rows × 9 columns</p>
</div>



### Step 7. Oops, there is a column missing, called owners. Create a random number Series from 15,000 to 73,000.


```python
owners = pd.Series(np.random.randint(15000, 73000, car.shape[0]))
owners
```




    0      34813
    1      19521
    2      58311
    3      32809
    4      57738
           ...  
    393    17063
    394    22313
    395    45045
    396    67461
    397    18697
    Length: 398, dtype: int32



### Step 8. Add the column owners to cars


```python
car["owners"] = owners
car 
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
      <th>mpg</th>
      <th>cylinders</th>
      <th>displacement</th>
      <th>horsepower</th>
      <th>weight</th>
      <th>acceleration</th>
      <th>model</th>
      <th>origin</th>
      <th>car</th>
      <th>owners</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>18.0</td>
      <td>8</td>
      <td>307</td>
      <td>130</td>
      <td>3504</td>
      <td>12.0</td>
      <td>70</td>
      <td>1</td>
      <td>chevrolet chevelle malibu</td>
      <td>34813</td>
    </tr>
    <tr>
      <th>1</th>
      <td>15.0</td>
      <td>8</td>
      <td>350</td>
      <td>165</td>
      <td>3693</td>
      <td>11.5</td>
      <td>70</td>
      <td>1</td>
      <td>buick skylark 320</td>
      <td>19521</td>
    </tr>
    <tr>
      <th>2</th>
      <td>18.0</td>
      <td>8</td>
      <td>318</td>
      <td>150</td>
      <td>3436</td>
      <td>11.0</td>
      <td>70</td>
      <td>1</td>
      <td>plymouth satellite</td>
      <td>58311</td>
    </tr>
    <tr>
      <th>3</th>
      <td>16.0</td>
      <td>8</td>
      <td>304</td>
      <td>150</td>
      <td>3433</td>
      <td>12.0</td>
      <td>70</td>
      <td>1</td>
      <td>amc rebel sst</td>
      <td>32809</td>
    </tr>
    <tr>
      <th>4</th>
      <td>17.0</td>
      <td>8</td>
      <td>302</td>
      <td>140</td>
      <td>3449</td>
      <td>10.5</td>
      <td>70</td>
      <td>1</td>
      <td>ford torino</td>
      <td>57738</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>195</th>
      <td>27.0</td>
      <td>4</td>
      <td>140</td>
      <td>86</td>
      <td>2790</td>
      <td>15.6</td>
      <td>82</td>
      <td>1</td>
      <td>ford mustang gl</td>
      <td>15479</td>
    </tr>
    <tr>
      <th>196</th>
      <td>44.0</td>
      <td>4</td>
      <td>97</td>
      <td>52</td>
      <td>2130</td>
      <td>24.6</td>
      <td>82</td>
      <td>2</td>
      <td>vw pickup</td>
      <td>19136</td>
    </tr>
    <tr>
      <th>197</th>
      <td>32.0</td>
      <td>4</td>
      <td>135</td>
      <td>84</td>
      <td>2295</td>
      <td>11.6</td>
      <td>82</td>
      <td>1</td>
      <td>dodge rampage</td>
      <td>24516</td>
    </tr>
    <tr>
      <th>198</th>
      <td>28.0</td>
      <td>4</td>
      <td>120</td>
      <td>79</td>
      <td>2625</td>
      <td>18.6</td>
      <td>82</td>
      <td>1</td>
      <td>ford ranger</td>
      <td>42315</td>
    </tr>
    <tr>
      <th>199</th>
      <td>31.0</td>
      <td>4</td>
      <td>119</td>
      <td>82</td>
      <td>2720</td>
      <td>19.4</td>
      <td>82</td>
      <td>1</td>
      <td>chevy s-10</td>
      <td>15689</td>
    </tr>
  </tbody>
</table>
<p>398 rows × 10 columns</p>
</div>




```python
car.describe()
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
      <th>mpg</th>
      <th>cylinders</th>
      <th>displacement</th>
      <th>weight</th>
      <th>acceleration</th>
      <th>model</th>
      <th>origin</th>
      <th>owners</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>398.000000</td>
      <td>398.000000</td>
      <td>398.000000</td>
      <td>398.000000</td>
      <td>398.000000</td>
      <td>398.000000</td>
      <td>398.000000</td>
      <td>398.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>23.514573</td>
      <td>5.454774</td>
      <td>193.427136</td>
      <td>2970.424623</td>
      <td>15.568090</td>
      <td>76.010050</td>
      <td>1.572864</td>
      <td>44485.256281</td>
    </tr>
    <tr>
      <th>std</th>
      <td>7.815984</td>
      <td>1.701004</td>
      <td>104.268683</td>
      <td>846.841774</td>
      <td>2.757689</td>
      <td>3.697627</td>
      <td>0.802055</td>
      <td>16712.430261</td>
    </tr>
    <tr>
      <th>min</th>
      <td>9.000000</td>
      <td>3.000000</td>
      <td>68.000000</td>
      <td>1613.000000</td>
      <td>8.000000</td>
      <td>70.000000</td>
      <td>1.000000</td>
      <td>15165.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>17.500000</td>
      <td>4.000000</td>
      <td>104.250000</td>
      <td>2223.750000</td>
      <td>13.825000</td>
      <td>73.000000</td>
      <td>1.000000</td>
      <td>30831.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>23.000000</td>
      <td>4.000000</td>
      <td>148.500000</td>
      <td>2803.500000</td>
      <td>15.500000</td>
      <td>76.000000</td>
      <td>1.000000</td>
      <td>44198.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>29.000000</td>
      <td>8.000000</td>
      <td>262.000000</td>
      <td>3608.000000</td>
      <td>17.175000</td>
      <td>79.000000</td>
      <td>2.000000</td>
      <td>59600.250000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>46.600000</td>
      <td>8.000000</td>
      <td>455.000000</td>
      <td>5140.000000</td>
      <td>24.800000</td>
      <td>82.000000</td>
      <td>3.000000</td>
      <td>72816.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
car.index
```




    Int64Index([  0,   1,   2,   3,   4,   5,   6,   7,   8,   9,
                ...
                190, 191, 192, 193, 194, 195, 196, 197, 198, 199],
               dtype='int64', length=398)




```python
car.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 398 entries, 0 to 199
    Data columns (total 10 columns):
     #   Column        Non-Null Count  Dtype  
    ---  ------        --------------  -----  
     0   mpg           398 non-null    float64
     1   cylinders     398 non-null    int64  
     2   displacement  398 non-null    int64  
     3   horsepower    398 non-null    object 
     4   weight        398 non-null    int64  
     5   acceleration  398 non-null    float64
     6   model         398 non-null    int64  
     7   origin        398 non-null    int64  
     8   car           398 non-null    object 
     9   owners        398 non-null    int32  
    dtypes: float64(2), int32(1), int64(5), object(2)
    memory usage: 32.6+ KB
    
