'''
对现有的logistic回归的一个小改造，
所考虑的问题是系数的变化
这是一个系数与时间有关的回归问题
'''
import pymysql.cursors
import numpy as np


def logistic():
    connection = database()
    stock_id = get_id(connection)
    alpha = 0.01 # 学习率
    theta = 1
    while True:
        z = []
        y_test = []
        t = []
        for i in stock_id:
            date_ = []
            money = []
            s_id = i['id']
            info = get_info_by_id(s_id, connection)
            if info:
                for n in info:
                    date_.append(date_to_t(n['date_']))
                    money.append(n['money'])
                for t in range(len(money)):
                    if t != len(money):
                        money[t] = money[t] - money[t + 1]
                    else:
                        y_test.append(0 if money[t] - money[t-1] < 0 else 1)
                        money[t] = 0
                t.append(date_)
                z.append(zs(date_, money, theta))
            else:
                continue
        z = np.array(z)
        t = np.array(t)
        y_test = np.array(y_test)
        y_pred = sigmoid(z)
        theta1 = theta - alpha * 1 / (len(z)) * (y_pred - y_test).dot(z * 1 / t)
        if theta1 - theta < 0.1:
            break
        else:
            theta = theta1
    connection.close()
    print(theta)

def sigmoid(z):
    '''
    逻辑回归的激活函数
    '''
    y_pred = 1 / (1 + np.exp(-z))
    return y_pred

def zs(t, delta_p, theta = 1):
    '''
    此函数用来计算z
    之后代入sigmoid函数即可
    '''
    z = 0
    for i in range(len(t)):
        z += delta_p[i] * f(t[i], theta)
    return z

def date_to_t(date_):
    '''
    将日期转换成为我们所想要的t
    即转换成可计算的数值类
    '''
    date = date_.split('-')
    for i in range(len(date)):
        date[i] = int(date[i])
    t = date[0] -2010 + (date[1] - 1) / 12 + (date[2] - 1) / 31
    return t

def database():
    '''
    搭建数据库的连接
    '''
    connection = pymysql.connect(
        host='localhost',
        user='',
        password='',
        db='stock',
        charset='utf8',
        cursorclass=pymysql.cursors.DictCursor
        )
    return connection

def get_id(connection):
    '''
    得到股票代码
    '''
    cs = connection.cursor()
    sql = 'select id from stock_id'
    cs.execute(sql)
    stock_id = cs.fetchall()
    return stock_id

def get_info_by_id(stock_id, connection):
    '''
    这个函数的目的是通过股票的id
    来对同一只股票的数据进行提取
    '''
    cs = connection.cursor()
    sql = 'select date_, money from stock where id = %s' % stock_id
    cs.execute(sql)
    info = cs.fetchall()
    return info

def f(t, theta=1):
    '''
    所需要系数的形式
    我们就用反比例函数来试验好了
    '''
    ans = 1 / (t ** theta)
    return ans
