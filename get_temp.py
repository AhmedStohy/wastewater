import urllib.request as request
from bs4 import BeautifulSoup
from matplotlib import pyplot as plt
import datetime
import ssl
import os

ssl._create_default_https_context = ssl._create_unverified_context

save_file = 'data/temp/'
skip_tab_head = 1


## 获取某城市某年某个月份区间的温度范围,写入文件
## city：城市的字符串；然后是整数的年，月份区间；然后是保存的文件名
def getData(city, year, month_from, month_to, fileName):
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64; rv:23.0) Gecko/20100101 Firefox/23.0'}
    ## 格式为http://www.tianqihoubao.com/lishi/wuxi/month/201910.html
    prefix = "http://www.tianqihoubao.com/lishi/{}/month/{}".format(city,
                                                                    year)  # 格式为http://www.tianqihoubao.com/lishi/wuxi/month/201910.html
    suffix = ".html"
    if os.path.exists(fileName):
        os.remove(fileName)
    ## 遍历月份
    for i in range(month_from, month_to + 1):
        if i < 10:
            url = prefix + str(0) + str(i) + suffix  # 小于10的时候日期格式为201808
        else:
            url = prefix + str(i) + suffix
        req = request.Request(url=url, headers=headers)  # 添加headers，避免防爬虫
        html = request.urlopen(req).read()  # 获取html
        soup = BeautifulSoup(html, "html.parser", from_encoding='gb2312')  # 解析工具
        table = soup.find('table')
        lines = []
        for i, tr in enumerate(table.findAll('tr')):
            if i < skip_tab_head:
                continue
            else:
                line = ''
                for j, td in enumerate(tr.findAll('td')):
                    if j == 0:
                        title = td.a['title']
                        year = title[:4]
                        month = title[5:7]
                        day = title[8:10]
                        line = line + '{}{}{},'.format(year, month, day)
                    if j == 2:
                        text = td.get_text().strip()
                        former, latter = text[:text.find('℃')], text[text.rfind(' ')+1:-1]
                        if int(latter) < int(former):
                            low_t, high_t = latter, former
                        else:
                            low_t, high_t = former, latter
                        line = line + '{},{}\n'.format(low_t, high_t)
                lines.append(line)
        with open(fileName, 'a') as fw:
            fw.writelines(lines)


# 一次性获取多个城市的多个年份数据
def getMoreData(cities, years, months=None):
    from_year, to_year = years
    for city in cities:
        for year in range(from_year, to_year + 1):
            fileName = 'data/temp/' + city + "_" + str(year) + ".txt"  # 命名方式
            if from_year == to_year and months != None:
                from_month, to_month = months
                getData(city, from_year, from_month, to_month, fileName)
            else:
                if year == datetime.datetime.now().year:
                    getData(city, year, 1, datetime.datetime.now().month, fileName)
                else:
                    getData(city, year, 1, 12, fileName)


if __name__ == '__main__':
    getMoreData(["wuxi"], [2020, 2020], [1, 4])
