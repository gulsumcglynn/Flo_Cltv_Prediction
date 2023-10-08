import pandas as pd
import datetime as dt
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions
from sklearn.preprocessing import MinMaxScaler
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.2f' % x)
pd.set_option("display.width",600)
pd.options.mode.chained_assignment = None
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Qt5Agg')

# GÖREV1:Veriyi Hazırlama

df_ = pd.read_csv("flo_data_20k.csv")
df = df_.copy()
df.head()
df.isnull().sum()
df.describe()

# 2. Aykırı değerleri baskılama
def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = round(quartile3 + 1.5 * interquantile_range)
    low_limit = round(quartile1 - 1.5 * interquantile_range)
    return low_limit, up_limit


def replace_tresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

df["order_num_total_ever_online"].quantile(0.01)
df["order_num_total_ever_online"].quantile(0.99)
interquantile_range = df["order_num_total_ever_online"].quantile(0.99) - df["order_num_total_ever_online"].quantile(
    0.01)
up_limit = round(df["order_num_total_ever_online"].quantile(0.99) + 1.5 * interquantile_range)
low_limit = round(df["order_num_total_ever_online"].quantile(0.01) - 1.5 * interquantile_range)

# 3. "order_num_total_ever_online","order_num_total_ever_offline","customer_value_total_ever_offline","customer_value_total_ever_online" değişkenlerinin
#aykırı değerleri varsa baskılayanız.

replace_tresholds(df, "order_num_total_ever_online")
replace_tresholds(df, "order_num_total_ever_offline")
replace_tresholds(df, "customer_value_total_ever_offline")
replace_tresholds(df, "customer_value_total_ever_online")

#4. Herbir müşterinin toplam alışveriş sayısı ve harcaması için yeni değişkenler oluşturun.
df["order_num_total"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
df["customer_value_total"] = df["customer_value_total_ever_offline"] + df["customer_value_total_ever_online"]

# 5. Değişken tiplerini inceleyiniz. Tarih ifade eden değişkenlerin tipini date'e çeviriniz
df.info()
date = [col for col in df.columns if "date" in col]
df[date] = df[date].apply(pd.to_datetime)

# GÖREV2: CLTV Veri Yapısının Oluşturulması
# 1.Veri setindeki en son alışverişin yapıldığı tarihten 2 gün sonrasını analiz tarihi olarak alınız.
df["last_order_date"].max()
analysis_date = dt.datetime(2021, 6, 1)

# 2.customer_id, recency_cltv_weekly, T_weekly, frequency ve monetary_cltv_avg değerlerinin yer aldığı yeni bir cltv dataframe'i oluşturunuz.
cltv_df = pd.DataFrame()
cltv_df["Customer_id"] = df["master_id"]
cltv_df["recency_cltv_weekly"] = ((df["last_order_date"] - df["first_order_date"]).astype(
    'timedelta64[ns]')).dt.days / 7
cltv_df["T_weekly"] = ((analysis_date - df["first_order_date"]).astype("timedelta64[ns]")).dt.days / 7
cltv_df["frequency"] = df["order_num_total"]
cltv_df["monetary_cltv_avg"] = df["customer_value_total"] / df["order_num_total"]
cltv_df = cltv_df[(cltv_df["frequency"] > 1)]
cltv_df.head()

# GÖREV3: BG/NBD, Gamma-Gamma Modellerinin Kurulması, 6 aylık CLTV'nin hesaplanması
# 1. BG/NBD modelini kurunuz.

bgf = BetaGeoFitter(penalizer_coef=0.001)

bgf.fit(cltv_df["frequency"], cltv_df["recency_cltv_weekly"],
        cltv_df["T_weekly"])

# 3 ay içerisinde müşterilerden beklenen satın almaları tahmin ediniz ve exp_sales_3_month olarak cltv dataframe'ine ekleyiniz.

cltv_df["exp_sales_3_months"] = bgf.predict(4 * 3,
                                            cltv_df["frequency"],
                                            cltv_df["recency_cltv_weekly"], cltv_df["T_weekly"])

# 6 ay içerisinde müşterilerden beklenen satın almaları tahmin ediniz ve exp_sales_6_month olarak cltv dataframe'ine ekleyiniz.
cltv_df["exp_sales_6_months"] = bgf.predict(4 * 6,
                                            cltv_df["frequency"],
                                            cltv_df["recency_cltv_weekly"], cltv_df["T_weekly"])

# 3. ve 6.aydaki en çok satın alım gerçekleştirecek 10 kişiyi inceleyeniz.
cltv_df.sort_values("exp_sales_3_month",ascending=False)[:10]

cltv_df.sort_values("exp_sales_6_month",ascending=False)[:10]

# 2.  Gamma-Gamma modelini fit ediniz. Müşterilerin ortalama bırakacakları değeri tahminleyip exp_average_value olarak cltv dataframe'ine ekleyiniz.
ggf = GammaGammaFitter(penalizer_coef=0.01)
ggf.fit(cltv_df["frequency"], cltv_df["monetary_cltv_avg"])

cltv_df["exp_average_value"] = ggf.conditional_expected_average_profit(cltv_df["frequency"],
                                                                       cltv_df["monetary_cltv_avg"])

# 3. 6 aylık CLTV hesaplayınız ve cltv ismiyle dataframe'e ekleyiniz.
cltv = ggf.customer_lifetime_value(bgf,
                                   cltv_df['frequency'],
                                   cltv_df['recency_cltv_weekly'],
                                   cltv_df['T_weekly'],
                                   cltv_df['monetary_cltv_avg'],
                                   time=6,  # 6 aylık
                                   freq="W",  # T'nin frekans bilgisi.
                                   discount_rate=0.01)

cltv_df["scaled_cltv"] = cltv

# CLTV değeri en yüksek 20 kişiyi gözlemleyiniz.
cltv_df.sort_values("cltv",ascending=False)[:20]

# GÖREV4:CLTV'ye Göre Segmentlerin Oluşturulması
# 1. 6 aylık standartlaştırılmış CLTV'ye göre tüm müşterilerinizi 4 gruba (segmente) ayırınız ve grup isimlerini veri setine ekleyiniz.
# cltv_segment ismi ile atayınız

cltv["segment"] = pd.qcut(cltv_df["scaled_cltv"], 4, labels=["D", "C", "B", "A"])



