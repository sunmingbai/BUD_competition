def findhistory(tarset): # tarset 被聚合的列表
    # 创建被聚合的变量列表
    tarcol=[]
    
    # 赋值tarcol，一般是除了时间和地域标签外的变量被聚合
    for x in tarset.columns:
        if x not in ['Year','Month', 'SalesRegion','Date']:
            tarcol.append(x)
    
    #加上聚合所依据的变量创建完整列表
    tarcolfull=tarcol.copy().extend(['Month', 'SalesRegion'])
    tarset=tarset.copy()
    
    # 创建聚合后所用的空DF
    tarhist=pd.DataFrame(columns=tarcolfull,index=range(24))
    
    # 与我的数据集有关，有的有Month，有的只有Date，核心就是开始GROUPBY
    if 'Month' in tarset.columns:
        targrouped=tarset.groupby(['Month','SalesRegion'])
    elif 'Date' in tarset.columns:
        tarset['Month']=tarset['Date'].apply(lambda x: x.month)
        targrouped=tarset.groupby(['Month','SalesRegion'])
    else:
        print('Dont have proper timestamp')
    
    # 空DF赋值标记
    count=0
    
    # GET GROUP 循环赋值
    for m in range(1,13):
        for n in ['Heilongjiang','Jilin']:
            tseries=targrouped.get_group((m,n)).mean()
            tarhist.loc[count,'Month']=m
            tarhist.loc[count,'SalesRegion']=n
            for x in tarcol:
                tarhist.loc[count,x]=tseries[x] #对被聚合后的多个变量循环复制
            count=count+1
    # 时间错位
    tarhist['Month']=tarhist['Month'].apply(lambda x: x-1 if x>1 else x+11)
    # 命名标记
    tarhist.columns=tarhist.columns.map(lambda x: x+'_hist' if x in tarcol else x)
    return tarhist
