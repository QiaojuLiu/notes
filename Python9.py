#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# 数据分析和绘图


# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False   #用来正常显示负号


# In[5]:


# 招聘数据分析
df = pd.read_csv('pl_09/DataAnalyst.csv', encoding='gb2312')
df.head()


# In[6]:


df.info()


# In[7]:


# 拆分工资字符串
arr = df.salary.str.split('-')
arr_df = pd.DataFrame(arr.tolist(), columns=['bottom', 'top'])
arr_df['top'] = arr_df['top'].str[:-1]
arr_df['bottom'] = arr_df['bottom'].str[:-1]
arr_df.head()


# In[8]:


# 删除重复职位
df = df.drop_duplicates(subset='positionId', keep='first')
df.info()


# In[10]:


# 加工salary字段
def get_salary(salary, method='None'):
    arr = salary.split('-')
    if len(arr) == 1:
        bottom = arr[0].upper()
        bottom = bottom[:bottom.find('K')]
        top = bottom
    else:
        bottom = arr[0][:-1]
        top = arr[1].upper()
        top = top[:top.find('K')]
    if method == 'top':
        return top
    elif method == 'bottom':
        return bottom

df['bottomSalary'] = df.salary.apply(get_salary, method='bottom')
df.bottomSalary = df.bottomSalary.astype('int') # 类型转换
df['topSalary'] = df.salary.apply(get_salary, method='top')
df.topSalary = df.topSalary.astype('int') # 类型转换
df['avgSalary'] = (df['bottomSalary'] + df['topSalary']) / 2
df.head()


# In[11]:


# 选择关注的列
df = df[['city', 'companyShortName', 'companySize',
         'education', 'positionName', 'positionLables',
         'workYear', 'avgSalary']]
df.head()


# In[12]:


# 统计各个城市的职位数
df.city.value_counts()


# In[13]:


# 查看统计信息
df.describe()


# In[14]:


# 查看各城市平均工资
df.groupby('city').mean()


# In[15]:


# 各城市按学位分组求平均工资
df_city_education = df.groupby(['city', 'education']).mean()
df_city_education


# In[16]:


# 优化展示
df_city_education = df_city_education.unstack()
df_city_education


# In[17]:


# 查看各公司招聘数量与平均工资
df.groupby('companyShortName').avgSalary.agg(['count', 'mean']).sort_values(by='count', ascending=False)


# In[18]:


# 按城市查看企业招聘需求前5
def topN(df, n=5):
    counts = df.value_counts()
    return counts.sort_values(ascending=False)[:n]
df.groupby('city').companyShortName.apply(topN)


# In[19]:


# 按城市查看岗位招聘需求前5
df.groupby('city').positionName.apply(topN)


# In[22]:


# 美国总统大选数据分析
result_df = pd.read_csv('pl_09/2016-us-ge-by-county.csv')
result_df.drop('StateCode', axis=1, inplace=True)
result_df.drop('CountyFips', axis=1, inplace=True)
result_df.head()


# In[23]:


# 提取Trump数据
trump_df = result_df[result_df['Candidate'] == 'Trump']
trump_df.info()
trump_df.head()


# In[24]:


# Clinton
clinton_df = result_df[result_df['Candidate'] == 'Clinton']
clinton_df.info()
clinton_df.head()


# In[25]:


# 合并成一张表并却却按州分组
result_df = pd.merge(trump_df, clinton_df, 
                     on=['StateName', 'CountyName', 'CountyTotalVote'], 
                     suffixes=['_t', '_c'])
print(result_df.head())
result_df.drop('Party_t', axis=1, inplace=True)
result_df.drop('Party_c', axis=1, inplace=True)
result_df.drop('Candidate_t', axis=1, inplace=True)
result_df.drop('Candidate_c', axis=1, inplace=True)
result_df.columns = ['StateName', 'CountyName', 'TotalVote', 'VoteTrump', 'VoteClinton']
result_df = result_df.groupby(by=result_df['StateName'], as_index=False).sum() # 试下as_index不同值的区别
result_df.drop([8], inplace=True) # 删除district-of-columbia
result_df['T-Ratio'] = result_df['VoteTrump'] / result_df['TotalVote']
result_df['C-Ratio'] = result_df['VoteClinton'] / result_df['TotalVote']
result_df['Winner'] = list(map(lambda x, y: 'Trump' if x > y else 'Clinton', result_df['T-Ratio'], result_df['C-Ratio']))
print(result_df.head())
print(result_df['Winner'].value_counts())


# In[27]:


# 收入数据来源：https://en.wikipedia.org/wiki/List_of_U.S._states_by_income
income_df = pd.read_csv('pl_09/2015-us-income-by-county.csv')
income_df.info()
print(income_df.head())
print(income_df.tail())


# In[28]:


# 合并收入数据
result_df = pd.merge(result_df, income_df, on=['StateName'])
result_df.head()


# In[29]:


plt.rcParams['font.sans-serif']=['SimHei'] # 用来正常显示中文标签
colors = result_df.Winner.map({'Trump':'Red', 'Clinton':'Blue'}) # 根据赢家设置颜色
spot_size = result_df.TotalVote / 4000.0 # 设置点大小
plt.figure(figsize=(17, 12)) 
plt.scatter(x = result_df['Income'], 
            y = result_df['T-Ratio'],
            s=spot_size,
            c=colors,
            alpha=0.25)
# plt.scatter( x = result_df['Income'], y = result_df['T-Ratio'], s=spot_size, alpha=0.25)
plt.xlabel('人均收入（美元）',fontsize=20)
plt.ylabel('各州Trump得票比率',fontsize=20)
plt.grid(True) # 添加网格
plt.show()


# In[30]:


# 画图


# In[31]:


plt.figure(figsize=(8, 6))
# 从1X1的网格创建一个子图片
plt.subplot(1, 1, 1)
X = np.linspace(-np.pi, np.pi, 256, endpoint=True)
C, S = np.cos(X), np.sin(X)
# 用宽度为1（像素）的蓝色连续直线绘制cosine
plt.plot(X, C, color="blue", linewidth=1.0, linestyle="-")
# 用宽度为1（像素）的绿色连续直线绘制sine
plt.plot(X, S, color="green", linewidth=1.0, linestyle="-")
# 设置x轴的极值
plt.xlim(-4.0, 4.0)
# 设置x轴的刻度值
plt.xticks(np.linspace(-4, 4, 9, endpoint=True))
# 设置y轴的极值
plt.ylim(-1.0, 1.0)
# 设置y轴的刻度值
plt.yticks(np.linspace(-1, 1, 5, endpoint=True))
# 在屏幕上显示结果
plt.show()


# In[32]:


# 移动轴和图例
ax = plt.gca()  # gca stands for 'get current axis'
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.spines['bottom'].set_position(('data',0))
ax.yaxis.set_ticks_position('left')
ax.spines['left'].set_position(('data',0))
plt.setp(ax, 
         xticks=[-np.pi, -np.pi/2, 0, np.pi/2, np.pi],
         xticklabels=[r'$-\pi$', r'$-\pi/2$', r'$0$', r'$+\pi/2$', r'$+\pi$'])
plt.setp(ax, 
         yticks=[-1, 0, +1],
         yticklabels=[r'$-1$', r'$0$', r'$+1$'])
ax.plot(X, C, color="blue", linewidth=1.0, linestyle='-', label='cosine')
# 用宽度为1（像素）的绿色连续直线绘制sine
ax.plot(X, S, color="green", linewidth=1.0, linestyle='-', label="sine")
ax.legend(loc='upper left') # 顺序要在plot后面
# 特殊点标注
t = 2 * np.pi / 3
ax.annotate(r'$\sin(\frac{2\pi}{3})=\frac{\sqrt{3}}{2}$',
            xy=(t, np.sin(t)), xycoords='data',
            xytext=(+10, +30), textcoords='offset points', fontsize=16,
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))
ax.annotate(r'$\cos(\frac{2\pi}{3})=-\frac{1}{2}$',
            xy=(t, np.cos(t)), xycoords='data',
            xytext=(-90, -50), textcoords='offset points', fontsize=16,
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))
ax.scatter([t,], [np.sin(t),], 50, color ='green')
ax.scatter([t,], [np.cos(t),], 50, color ='blue')
plt.show()


# In[33]:


# 图嵌套
x = np.linspace(0, 5, 10)
y = x ** 2

fig = plt.figure()

# axes = fig.add_axes([0.1, 0.1, 0.8, 0.8]) # left, bottom, width, height (range 0 to 1)
axes = fig.gca()
axes2 = fig.add_axes([0.2, 0.5, 0.4, 0.3]) # inset axes

axes.plot(x, y, 'r')

axes.set_xlabel('x')
axes.set_ylabel('y')
axes.set_title('title')

# insert
axes2.plot(y, x, 'g')
axes2.set_xlabel('y')
axes2.set_ylabel('x')
axes2.set_title('insert title');

plt.show()


# In[34]:


# 正态分布柱状图
mu =0.0
sigma = 2.0
samples = np.random.normal(loc=mu, scale=sigma, size=2000)
plt.figure(num=1, figsize=(8,6))
plt.title('Normal Distribution', size=10)
plt.xlabel('value', size=10)
plt.ylabel('counts', size=10)
plt.hist(samples, bins=80, range=(-10,10)) # bins=柱子数目


# In[35]:


def radar_plot():
    """
    radar plot
    """
    # 生成测试数据
    labels = np.array(["A", "B", "C", "D", "E", "F"])
    data = np.array([68, 83, 90, 77, 89, 73])
    theta = np.linspace(0, 2*np.pi, len(data), endpoint=False)

    # 数据预处理
    data = np.concatenate((data, [data[0]]))
    print(data)
    theta = np.concatenate((theta, [theta[0]]))

    # 画图方式
    plt.subplot(111, polar=True)
    plt.title("Radar")

    # 设置"theta grid"/"radar grid"
    plt.thetagrids(theta*(180/np.pi), labels=labels)
    plt.rgrids(np.arange(20, 100, 20), labels=np.arange(20, 100, 20), angle=0)
    plt.ylim(0, 100)

    # 画雷达图,并填充雷达图内部区域
    plt.plot(theta, data, "bo-", linewidth=2)
    plt.fill(theta, data, color="red", alpha=0.25)

    # 图形显示
    plt.show()
    return

radar_plot()


# In[36]:


def pie_plot():
    """
    pie plot
    """
    # 生成测试数据
    sizes = [15.88, 29.12, 45, 10]
    labels = ["Frogs", "Chinese", "Dogs", "Logs"]
    colors = ["yellowgreen", "gold", "lightskyblue", "lightcoral"]

    # 设置标题
    plt.title("Pie")

    # 设置突出参数
    explode = [0, 0.1, 0, 0]

    # 画饼状图
    patches, l_text, p_text = plt.pie(sizes, 
                                      explode=explode, 
                                      labels=labels,
                                      colors=colors,
                                      autopct="%.2f%%", 
                                      shadow=True,
                                      startangle=180)
    plt.axis("equal")

    # 图形显示
    plt.show()
    return

pie_plot()


# In[37]:


# 实战：销售数据分析
df=pd.read_excel('pl_09\\朝阳医院2016年销售数据.xlsx',sheet_name='Sheet1')
df.info()


# In[ ]:




