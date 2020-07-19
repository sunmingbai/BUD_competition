# BUD_competition

# 0718

# XGboost

## 只包含简单时间特征

**特征**
时间特征

**参数列表**
XGBRegressor(alpha=0, base_score=0.5, booster='gbtree', colsample_bylevel=1,
             colsample_bynode=1, colsample_bytree=1, eta=0.3,
             eval_metric=['mae'], gama=0, gamma=0, importance_type='gain',
             lambda=1, learning_rate=0.1, max_delta_step=0, max_depth=4,
             min_child_weight=1, missing=None, n_estimators=100, n_jobs=1,
             nthread=None, objective='reg:linear', random_state=0, reg_alpha=0,
             reg_lambda=1, scale_pos_weight=1, seed=100, silent=1, subsample=1,
             verbosity=1)

**效果**
1. 合并SKU Code wmape=0.55
2. 单独SKU Code wmape=0.47

## 包含更多时间特征

**特征**
补充CPI等时间特征，模型表现变好

**参数列表**
同上

**效果**
1. 合并SKU Code wmape=0.55
2. 单独SKU Code wmape=0.42

## 引入CPI与Temp的指数平滑特征（平滑后按月错位）（环比）

## + 历史同期数据特征（同比）
!(https://github.com/sunmingbai/BUD_compete/blob/master/version3%20put.png)

## 引入销量rolling特征
all sku: drop to 0.48
https://github.com/sunmingbai/BUD_competition/blob/master/Ming/version%204%20output.png
single sku: increase to  0.48


# STL+ETS

## 利用2016年1月到2018年11月的数据，预测2019年1月

wmape=0.2464219
stlf(s, 
     h=horizon,#预测多久 
     s.window=3, 
     method='ets',
     ic='bic', 
     opt.crit='mae')
     
# STL+ARIMA
## 利用2016年1月到2018年11月的数据，预测2019年1月
wmape=0.5920622
stlf(s, 
     h=horizon, #预测多久
     s.window=3, 
     method='arima',
     ic='bic')
     
# 0719

# XGBoost

## modeling with monthly data
wmape=0.27
