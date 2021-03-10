# About This Project 
I developed Recurrent Neural Network(RNN) that can perform prediction of crypto currency prices.
The dataset being used is web scraped cryptocurrency data.


# 1. Data Cleaning 


```python
# installing
%matplotlib inline
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
```


```python
bitmex_price = pd.read_csv('bitmex_prices .csv')
```

### Checking missing values


```python
bitmex_price.isnull().sum()
```




    date      0
    xbtusd    0
    dtype: int64



**Observation** <br>
No missing values,data is already cleaned. but we want to make sure
1. datetypes is stored appropriately
2. The data is in correct order


```python
bitmex_price.head()
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
      <th>date</th>
      <th>xbtusd</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2019-04-13 10:55:00</td>
      <td>5053.25</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2019-04-13 10:56:00</td>
      <td>5053.25</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2019-04-13 10:57:00</td>
      <td>5053.25</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2019-04-13 10:58:00</td>
      <td>5052.25</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2019-04-13 10:59:00</td>
      <td>5049.25</td>
    </tr>
  </tbody>
</table>
</div>




```python
bitmex_price.dtypes
```




    date       object
    xbtusd    float64
    dtype: object



**currently the date column is object data type, converting it to datetime to allow more manipulation on data**


```python
bitmex_price['date'] = pd.to_datetime(bitmex_price['date'])
bitmex_price.dtypes
```




    date      datetime64[ns]
    xbtusd           float64
    dtype: object




```python
bitmex_price['date'][1] - bitmex_price['date'][0]
```




    Timedelta('0 days 00:01:00')



### Making sure the date was correctly sorted 


```python
temp = bitmex_price.sort_values(by='date')
```


```python
(temp['date'] == bitmex_price['date']).sum() == len(bitmex_price)
```




    True



the data is in correct order

# 2. Data Analysis & Visualization


```python
bitmex_price.describe()
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
      <th>xbtusd</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>121922.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>7861.513337</td>
    </tr>
    <tr>
      <th>std</th>
      <td>2125.928309</td>
    </tr>
    <tr>
      <th>min</th>
      <td>4958.250000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>5679.750000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>7938.750000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>8873.250000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>13888.250000</td>
    </tr>
  </tbody>
</table>
</div>



## The Prices from beginning  to the end


```python
plt.plot(bitmex_price['xbtusd'])
```




    [<matplotlib.lines.Line2D at 0x1a3f0e84a8>]




    
![png](Crypto%20Price%20Prediction%28EDA%20%26%20Feature%20Engineering%29_files/Crypto%20Price%20Prediction%28EDA%20%26%20Feature%20Engineering%29_18_1.png)
    


## The Prices of last 30 minutes


```python
plt.plot(bitmex_price['xbtusd'][-30:])
```




    [<matplotlib.lines.Line2D at 0x1a3f14fb38>]




    
![png](Crypto%20Price%20Prediction%28EDA%20%26%20Feature%20Engineering%29_files/Crypto%20Price%20Prediction%28EDA%20%26%20Feature%20Engineering%29_20_1.png)
    



```python
def to_hour(df, agg=np.mean):
    df = df.copy()
    counter = 0
    minunte_prices = []
    hour_prices = []
    for x in df['xbtusd']:
        minunte_prices.append(x)
        if len(minunte_prices) == 60:
            hour_prices.append(agg(minunte_prices))
            minunte_prices = []
    hour_prices.append(agg(minunte_prices))
    return pd.Series(hour_prices)

hour_avg_prices = to_hour(bitmex_price, np.mean)
hour_max_prices = to_hour(bitmex_price, np.max)
hour_min_prices = to_hour(bitmex_price, np.min)
```

## The Average  Prices by Hour


```python
plt.plot(hour_avg_prices)
plt.plot(hour_max_prices)
plt.plot(hour_min_prices)

plt.legend(['average', 'max', 'min'], loc='upper left')
```




    <matplotlib.legend.Legend at 0x1a4718be80>




    
![png](Crypto%20Price%20Prediction%28EDA%20%26%20Feature%20Engineering%29_files/Crypto%20Price%20Prediction%28EDA%20%26%20Feature%20Engineering%29_23_1.png)
    


### last 24 hours


```python
plt.plot(hour_avg_prices[-24:])
plt.plot(hour_max_prices[-24:])
plt.plot(hour_min_prices[-24:])

plt.legend(['average', 'max', 'min'], loc='upper right')
```




    <matplotlib.legend.Legend at 0x1a471e5128>




    
![png](Crypto%20Price%20Prediction%28EDA%20%26%20Feature%20Engineering%29_files/Crypto%20Price%20Prediction%28EDA%20%26%20Feature%20Engineering%29_25_1.png)
    



```python
def to_day(df, agg=np.mean):
    hours = to_hour(df, agg)
    hour_prices = []
    day_prices = []
    for hour in hours:
        hour_prices.append(hour)
        if len(hour_prices) == 24:
            day_prices.append(agg(hour_prices))
            hour_prices = []
    day_prices.append(agg(hour_prices))
    return pd.Series(day_prices)

day_avg = to_day(bitmex_price, np.mean)
day_max = to_day(bitmex_price, np.max)
day_min = to_day(bitmex_price, np.min)
```

## Prices By Days


```python
plt.plot(day_avg)
plt.plot(day_max)
plt.plot(day_min)

plt.legend(['average', 'max', 'min'], loc='upper left')
```




    <matplotlib.legend.Legend at 0x1a4057cf60>




    
![png](Crypto%20Price%20Prediction%28EDA%20%26%20Feature%20Engineering%29_files/Crypto%20Price%20Prediction%28EDA%20%26%20Feature%20Engineering%29_28_1.png)
    


### last 7 days


```python
plt.plot(day_avg[-7:])
plt.plot(day_max[-7:])
plt.plot(day_min[-7:])

plt.legend(['average', 'max', 'min'], loc='upper left')
```




    <matplotlib.legend.Legend at 0x1a44cfc080>




    
![png](Crypto%20Price%20Prediction%28EDA%20%26%20Feature%20Engineering%29_files/Crypto%20Price%20Prediction%28EDA%20%26%20Feature%20Engineering%29_30_1.png)
    


# 3. Feature Engineering

**Set Target Variable** <br>
We want to predict the next minute price will rise or not. Therefore, if current minutes' price is higher than the last minute, it is 1, otherwise it is 0


```python
y = bitmex_price['xbtusd'].shift(-1)
bitmex_price['target'] = y
bitmex_price.drop(bitmex_price.index[0] , inplace=True)
bitmex_price.drop(bitmex_price.index[-1] , inplace=True)
bitmex_price.head()
```

# Model Building

We want to predict the minute by minute crypto currency prices based on the prices on the past time stamps. This problem is sequential prediction. It is one of the hardest problems in data science industry. Often **Long Short Term Memory**, a variant of recurrent neural network, is used to solve sequential prediction. https://en.wikipedia.org/wiki/Long_short-term_memory

**Because of the nature of Long Short Term Memory, k-cross validation is inappropriate in this case** if you do k-cross validation on this problem, you would be **predicting past stock prices based on future stock prices** For example, LSTM will predict the stock price on Friday based on Mon-Thursday's prices, therefore Friday is appropriate validation set in this case. However if you do k-cross validation, they will make Monday's stock price as one of validation sets, you will predict it based on Tuesday-Friday stock prices. LSTM does not work like that and I do not think predicting past stock prices will bring much business values. For the same reason, shuffling the data set is also inappropriate to this problem

A validation set for LSTM must be continuation from the training set, therefore I will use a single validation=test set


```python
%matplotlib inline
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
import matplotlib.pyplot as plt
```


```python
dataset = pd.read_csv('bitmex_prices .csv')

```

## Feature Selection
Because we are now using LSTM, the custom features I created do not have much use. It is because, LSTM model stores the past stock prices in itself and optimize it as its weights. Instead of using feature such as "difference from average price of last 60 mins", I will just make the model predict stock price based on past 60 minutes prices


```python
prices = dataset['xbtusd'].to_numpy()
prices
```




    array([ 5053.25,  5053.25,  5053.25, ..., 12535.25, 12536.75, 12536.75])




```python
scaler = MinMaxScaler(feature_range=(0,1))
prices_scaled = scaler.fit_transform(prices.reshape(-1, 1))
prices_scaled.shape[0]*0.8
```

**The data is now scaled to the range 0-1**


```python
plt.plot(prices_scaled[-1000:])
```




    [<matplotlib.lines.Line2D at 0x7fab870e77b8>]




    
![png](Feature%20Selection%20%26%20Model_Building%20_files/Feature%20Selection%20%26%20Model_Building%20_10_1.png)
    


Using last 60 minutes time steps to predict next price <br>
for example, a sample x contains the price at 1st minute to 60th minute stamp, then the corresponding y will be the price at 61st minute


```python
x, y = [], []
for i in range(60,len(prices_scaled)):
    x.append(prices_scaled[i-60:i])
    y.append(prices_scaled[i])
x, y = np.array(x), np.array(y)

```


```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, shuffle=False)
y_train.shape
```




    (91396, 1)




```python
X_train = np.reshape(X_train, (X_train.shape[0],X_train.shape[1],1))
```


```python
X_train.shape
```




    (91396, 60, 1)



**Long Shot Term Memory Model**


```python
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1],1)))
model.add(LSTM(units=50))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.summary()
model.fit(X_train, y_train, epochs=1, batch_size=1, verbose=1)
```

    WARNING: Logging before flag parsing goes to stderr.
    W0719 14:27:19.716487 140375785535360 deprecation_wrapper.py:119] From /usr/local/lib/python3.6/dist-packages/keras/backend/tensorflow_backend.py:74: The name tf.get_default_graph is deprecated. Please use tf.compat.v1.get_default_graph instead.
    
    W0719 14:27:19.765002 140375785535360 deprecation_wrapper.py:119] From /usr/local/lib/python3.6/dist-packages/keras/backend/tensorflow_backend.py:517: The name tf.placeholder is deprecated. Please use tf.compat.v1.placeholder instead.
    
    W0719 14:27:19.777117 140375785535360 deprecation_wrapper.py:119] From /usr/local/lib/python3.6/dist-packages/keras/backend/tensorflow_backend.py:4138: The name tf.random_uniform is deprecated. Please use tf.random.uniform instead.
    
    W0719 14:27:20.242657 140375785535360 deprecation_wrapper.py:119] From /usr/local/lib/python3.6/dist-packages/keras/optimizers.py:790: The name tf.train.Optimizer is deprecated. Please use tf.compat.v1.train.Optimizer instead.
    


    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    lstm_1 (LSTM)                (None, 60, 50)            10400     
    _________________________________________________________________
    lstm_2 (LSTM)                (None, 50)                20200     
    _________________________________________________________________
    dense_1 (Dense)              (None, 1)                 51        
    =================================================================
    Total params: 30,651
    Trainable params: 30,651
    Non-trainable params: 0
    _________________________________________________________________


    W0719 14:27:20.544457 140375785535360 deprecation.py:323] From /usr/local/lib/python3.6/dist-packages/tensorflow/python/ops/math_grad.py:1250: add_dispatch_support.<locals>.wrapper (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    Use tf.where in 2.0, which has the same broadcast rule as np.where
    W0719 14:27:21.760176 140375785535360 deprecation_wrapper.py:119] From /usr/local/lib/python3.6/dist-packages/keras/backend/tensorflow_backend.py:986: The name tf.assign_add is deprecated. Please use tf.compat.v1.assign_add instead.
    
    W0719 14:27:21.881016 140375785535360 deprecation_wrapper.py:119] From /usr/local/lib/python3.6/dist-packages/keras/backend/tensorflow_backend.py:973: The name tf.assign is deprecated. Please use tf.compat.v1.assign instead.
    


    Epoch 1/1
    91396/91396 [==============================] - 5605s 61ms/step - loss: 2.0106e-05





    <keras.callbacks.History at 0x7fab83b2fef0>




```python
from keras.models import load_model

model.save('lstm_model.h5')
```


```python
predictions = model.predict(X_test, verbose=1)
```

    30466/30466 [==============================] - 14s 457us/step


**Scale back the predictions to actual prices**


```python
predictions = scaler.inverse_transform(predictions)
actual_price = scaler.inverse_transform(y_test)
```

## Prediction on the whole validation set


```python
plt.plot(predictions)
plt.plot(actual_price)
plt.legend(['my prediction', 'actual'], loc='upper right')
```




    <matplotlib.legend.Legend at 0x7fab81a4c7b8>




    
![png](Feature%20Selection%20%26%20Model_Building%20_files/Feature%20Selection%20%26%20Model_Building%20_23_1.png)
    


## First 5000 next minute predictions


```python
plt.plot(predictions[:5000])
plt.plot(actual_price[:5000])
plt.legend(['my prediction', 'actual'], loc='upper right')
```




    <matplotlib.legend.Legend at 0x7fab819b5ef0>




    
![png](Feature%20Selection%20%26%20Model_Building%20_files/Feature%20Selection%20%26%20Model_Building%20_25_1.png)
    


## Last 5000  predictions


```python
plt.plot(predictions[-5000:])
plt.plot(actual_price[-5000:])
plt.legend(['my prediction', 'actual'], loc='upper right')
```




    <matplotlib.legend.Legend at 0x7fab819754a8>




    
![png](Feature%20Selection%20%26%20Model_Building%20_files/Feature%20Selection%20%26%20Model_Building%20_27_1.png)
    


**Obsevation :**
The long and short term memory model is doing great job for predicting the price at time stamps which directly after the trainng. However, As I expected, the model is not very good at predicting far future after the last point you trained. Although it follows the same shape, there was some gap between predictions and actual prices in the last 5000 predictions. To solve this problem, we could train model more, if we can train the samples right before the last 5000 timestop, the predictions would be more accurate.


```python

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
      <th>date</th>
      <th>xbtusd</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>2019-04-13 10:56:00</td>
      <td>5053.25</td>
      <td>5053.25</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2019-04-13 10:57:00</td>
      <td>5053.25</td>
      <td>5052.25</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2019-04-13 10:58:00</td>
      <td>5052.25</td>
      <td>5049.25</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2019-04-13 10:59:00</td>
      <td>5049.25</td>
      <td>5047.25</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2019-04-13 11:00:00</td>
      <td>5047.25</td>
      <td>5051.25</td>
    </tr>
  </tbody>
</table>
</div>



### Adding Custome Features 

we will be adding following list of features to see if can help us classify better
1. change from 1 minute ago
2. change from 3 minute ago 
3. change from 5 minute ago 
4. change from 10 minute ago
5. difference from average of last 5 mins
6. difference from average of last 10 mins
7. difference from average of last 60 mins


```python
bitmex_price['1mins_ago'] =  bitmex_price['xbtusd'] - bitmex_price['xbtusd'].shift(1) 
bitmex_price['3mins_ago'] =  bitmex_price['xbtusd'] - bitmex_price['xbtusd'].shift(3) 
bitmex_price['5mins_ago'] = bitmex_price['xbtusd'] - bitmex_price['xbtusd'].shift(5)
bitmex_price['10mins_ago'] = bitmex_price['xbtusd'] - bitmex_price['xbtusd'].shift(10)
```


```python
def every_x_min_avg(df, x):
    sum_series = df['xbtusd'].copy()
    for i in range(x-1):
        sum_series += df['xbtusd'].shift(i)
    return sum_series / x

bitmex_price['5mins_avg'] = every_x_min_avg(bitmex_price, 5)
bitmex_price['10mins_avg'] = every_x_min_avg(bitmex_price, 10)
bitmex_price['60mins_avg'] = every_x_min_avg(bitmex_price, 60)
bitmex_price.tail()
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
      <th>date</th>
      <th>xbtusd</th>
      <th>target</th>
      <th>1mins_ago</th>
      <th>3mins_ago</th>
      <th>5mins_ago</th>
      <th>10mins_ago</th>
      <th>5mins_avg</th>
      <th>10mins_avg</th>
      <th>60mins_avg</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>121916</th>
      <td>2019-07-09 23:46:00</td>
      <td>12573.25</td>
      <td>12562.75</td>
      <td>3.0</td>
      <td>18.0</td>
      <td>-5.0</td>
      <td>-34.0</td>
      <td>12568.45</td>
      <td>12579.40</td>
      <td>12585.975000</td>
    </tr>
    <tr>
      <th>121917</th>
      <td>2019-07-09 23:47:00</td>
      <td>12562.75</td>
      <td>12562.75</td>
      <td>-10.5</td>
      <td>-7.5</td>
      <td>-15.5</td>
      <td>-47.5</td>
      <td>12567.85</td>
      <td>12574.10</td>
      <td>12586.116667</td>
    </tr>
    <tr>
      <th>121918</th>
      <td>2019-07-09 23:48:00</td>
      <td>12562.75</td>
      <td>12535.25</td>
      <td>0.0</td>
      <td>-7.5</td>
      <td>7.5</td>
      <td>-42.5</td>
      <td>12566.35</td>
      <td>12569.85</td>
      <td>12586.425000</td>
    </tr>
    <tr>
      <th>121919</th>
      <td>2019-07-09 23:49:00</td>
      <td>12535.25</td>
      <td>12536.75</td>
      <td>-27.5</td>
      <td>-38.0</td>
      <td>-35.0</td>
      <td>-70.0</td>
      <td>12553.85</td>
      <td>12562.15</td>
      <td>12586.108333</td>
    </tr>
    <tr>
      <th>121920</th>
      <td>2019-07-09 23:50:00</td>
      <td>12536.75</td>
      <td>12536.75</td>
      <td>1.5</td>
      <td>-26.0</td>
      <td>-33.5</td>
      <td>-48.0</td>
      <td>12546.85</td>
      <td>12558.15</td>
      <td>12586.300000</td>
    </tr>
  </tbody>
</table>
</div>




```python
bitmex_price['diff_5avg'] = bitmex_price['xbtusd'] - bitmex_price['5mins_avg']
bitmex_price['diff_10avg'] = bitmex_price['xbtusd'] - bitmex_price['10mins_avg']
bitmex_price['diff_60avg'] = bitmex_price['xbtusd'] - bitmex_price['60mins_avg']
bitmex_price.tail()
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
      <th>date</th>
      <th>xbtusd</th>
      <th>target</th>
      <th>1mins_ago</th>
      <th>3mins_ago</th>
      <th>5mins_ago</th>
      <th>10mins_ago</th>
      <th>5mins_avg</th>
      <th>10mins_avg</th>
      <th>60mins_avg</th>
      <th>diff_5avg</th>
      <th>diff_10avg</th>
      <th>diff_60avg</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>121916</th>
      <td>2019-07-09 23:46:00</td>
      <td>12573.25</td>
      <td>12562.75</td>
      <td>3.0</td>
      <td>18.0</td>
      <td>-5.0</td>
      <td>-34.0</td>
      <td>12568.45</td>
      <td>12579.40</td>
      <td>12585.975000</td>
      <td>4.8</td>
      <td>-6.15</td>
      <td>-12.725000</td>
    </tr>
    <tr>
      <th>121917</th>
      <td>2019-07-09 23:47:00</td>
      <td>12562.75</td>
      <td>12562.75</td>
      <td>-10.5</td>
      <td>-7.5</td>
      <td>-15.5</td>
      <td>-47.5</td>
      <td>12567.85</td>
      <td>12574.10</td>
      <td>12586.116667</td>
      <td>-5.1</td>
      <td>-11.35</td>
      <td>-23.366667</td>
    </tr>
    <tr>
      <th>121918</th>
      <td>2019-07-09 23:48:00</td>
      <td>12562.75</td>
      <td>12535.25</td>
      <td>0.0</td>
      <td>-7.5</td>
      <td>7.5</td>
      <td>-42.5</td>
      <td>12566.35</td>
      <td>12569.85</td>
      <td>12586.425000</td>
      <td>-3.6</td>
      <td>-7.10</td>
      <td>-23.675000</td>
    </tr>
    <tr>
      <th>121919</th>
      <td>2019-07-09 23:49:00</td>
      <td>12535.25</td>
      <td>12536.75</td>
      <td>-27.5</td>
      <td>-38.0</td>
      <td>-35.0</td>
      <td>-70.0</td>
      <td>12553.85</td>
      <td>12562.15</td>
      <td>12586.108333</td>
      <td>-18.6</td>
      <td>-26.90</td>
      <td>-50.858333</td>
    </tr>
    <tr>
      <th>121920</th>
      <td>2019-07-09 23:50:00</td>
      <td>12536.75</td>
      <td>12536.75</td>
      <td>1.5</td>
      <td>-26.0</td>
      <td>-33.5</td>
      <td>-48.0</td>
      <td>12546.85</td>
      <td>12558.15</td>
      <td>12586.300000</td>
      <td>-10.1</td>
      <td>-21.40</td>
      <td>-49.550000</td>
    </tr>
  </tbody>
</table>
</div>




```python
bitmex_price.drop(columns=['5mins_avg', '10mins_avg', '60mins_avg'], inplace=True)
bitmex_price.head()
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
      <th>date</th>
      <th>xbtusd</th>
      <th>target</th>
      <th>1mins_ago</th>
      <th>3mins_ago</th>
      <th>5mins_ago</th>
      <th>10mins_ago</th>
      <th>diff_5avg</th>
      <th>diff_10avg</th>
      <th>diff_60avg</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>2019-04-13 10:56:00</td>
      <td>5053.25</td>
      <td>5053.25</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2019-04-13 10:57:00</td>
      <td>5053.25</td>
      <td>5052.25</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2019-04-13 10:58:00</td>
      <td>5052.25</td>
      <td>5049.25</td>
      <td>-1.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2019-04-13 10:59:00</td>
      <td>5049.25</td>
      <td>5047.25</td>
      <td>-3.0</td>
      <td>-4.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-2.2</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2019-04-13 11:00:00</td>
      <td>5047.25</td>
      <td>5051.25</td>
      <td>-2.0</td>
      <td>-6.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-2.6</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
bitmex_price.drop(bitmex_price.index[0:60], inplace=True)
bitmex_price.reset_index(drop=True, inplace=True)
bitmex_price.head(10)
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
      <th>date</th>
      <th>xbtusd</th>
      <th>target</th>
      <th>1mins_ago</th>
      <th>3mins_ago</th>
      <th>5mins_ago</th>
      <th>10mins_ago</th>
      <th>diff_5avg</th>
      <th>diff_10avg</th>
      <th>diff_60avg</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2019-04-13 11:56:00</td>
      <td>5057.50</td>
      <td>5058.75</td>
      <td>0.75</td>
      <td>3.25</td>
      <td>3.25</td>
      <td>0.75</td>
      <td>1.45</td>
      <td>2.150</td>
      <td>0.341667</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2019-04-13 11:57:00</td>
      <td>5058.75</td>
      <td>5059.75</td>
      <td>1.25</td>
      <td>4.50</td>
      <td>4.50</td>
      <td>3.00</td>
      <td>1.55</td>
      <td>2.975</td>
      <td>1.462500</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2019-04-13 11:58:00</td>
      <td>5059.75</td>
      <td>5061.25</td>
      <td>1.00</td>
      <td>3.00</td>
      <td>5.50</td>
      <td>4.00</td>
      <td>1.25</td>
      <td>3.375</td>
      <td>2.270833</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2019-04-13 11:59:00</td>
      <td>5061.25</td>
      <td>5061.25</td>
      <td>1.50</td>
      <td>3.75</td>
      <td>7.00</td>
      <td>6.50</td>
      <td>1.55</td>
      <td>4.025</td>
      <td>3.512500</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2019-04-13 12:00:00</td>
      <td>5061.25</td>
      <td>5060.25</td>
      <td>0.00</td>
      <td>2.50</td>
      <td>4.50</td>
      <td>7.00</td>
      <td>0.80</td>
      <td>3.325</td>
      <td>3.345833</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2019-04-13 12:01:00</td>
      <td>5060.25</td>
      <td>5058.75</td>
      <td>-1.00</td>
      <td>0.50</td>
      <td>2.75</td>
      <td>6.00</td>
      <td>-0.30</td>
      <td>1.825</td>
      <td>2.270833</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2019-04-13 12:02:00</td>
      <td>5058.75</td>
      <td>5059.25</td>
      <td>-1.50</td>
      <td>-2.50</td>
      <td>0.00</td>
      <td>4.50</td>
      <td>-1.30</td>
      <td>0.025</td>
      <td>0.729167</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2019-04-13 12:03:00</td>
      <td>5059.25</td>
      <td>5059.50</td>
      <td>0.50</td>
      <td>-2.00</td>
      <td>-0.50</td>
      <td>5.00</td>
      <td>-0.50</td>
      <td>-0.025</td>
      <td>1.145833</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2019-04-13 12:04:00</td>
      <td>5059.50</td>
      <td>5059.75</td>
      <td>0.25</td>
      <td>-0.75</td>
      <td>-1.75</td>
      <td>5.25</td>
      <td>0.05</td>
      <td>-0.075</td>
      <td>1.320833</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2019-04-13 12:05:00</td>
      <td>5059.75</td>
      <td>5060.75</td>
      <td>0.25</td>
      <td>1.00</td>
      <td>-1.50</td>
      <td>3.00</td>
      <td>0.35</td>
      <td>-0.075</td>
      <td>1.508333</td>
    </tr>
  </tbody>
</table>
</div>




```python
bitmex_price.isnull().sum()
```




    date          0
    xbtusd        0
    target        0
    1mins_ago     0
    3mins_ago     0
    5mins_ago     0
    10mins_ago    0
    diff_5avg     0
    diff_10avg    0
    diff_60avg    0
    dtype: int64




```python
from sklearn.model_selection import train_test_split
y = bitmex_price.target
bitmex_price.drop(columns=['target', 'date'], inplace=True)
x_train, x_test, y_train, y_test = train_test_split(bitmex_price, y, test_size=0.20, shuffle=False)
```


```python
x_train.head()
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
      <th>xbtusd</th>
      <th>1mins_ago</th>
      <th>3mins_ago</th>
      <th>5mins_ago</th>
      <th>10mins_ago</th>
      <th>diff_5avg</th>
      <th>diff_10avg</th>
      <th>diff_60avg</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5057.50</td>
      <td>0.75</td>
      <td>3.25</td>
      <td>3.25</td>
      <td>0.75</td>
      <td>1.45</td>
      <td>2.150</td>
      <td>0.341667</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5058.75</td>
      <td>1.25</td>
      <td>4.50</td>
      <td>4.50</td>
      <td>3.00</td>
      <td>1.55</td>
      <td>2.975</td>
      <td>1.462500</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5059.75</td>
      <td>1.00</td>
      <td>3.00</td>
      <td>5.50</td>
      <td>4.00</td>
      <td>1.25</td>
      <td>3.375</td>
      <td>2.270833</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5061.25</td>
      <td>1.50</td>
      <td>3.75</td>
      <td>7.00</td>
      <td>6.50</td>
      <td>1.55</td>
      <td>4.025</td>
      <td>3.512500</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5061.25</td>
      <td>0.00</td>
      <td>2.50</td>
      <td>4.50</td>
      <td>7.00</td>
      <td>0.80</td>
      <td>3.325</td>
      <td>3.345833</td>
    </tr>
  </tbody>
</table>
</div>




```python
y_train.head()
```




    0    5058.75
    1    5059.75
    2    5061.25
    3    5061.25
    4    5060.25
    Name: target, dtype: float64




```python
x_train.to_csv('x_train.csv', index=False, header=True)

x_test.to_csv('x_test.csv', index=False, header=True)

y_train.to_csv('y_train.csv', index=False, header=True)

y_test.to_csv('y_test.csv', index=False, header=True)
```
