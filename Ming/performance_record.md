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
