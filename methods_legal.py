'''Методы и переменные, используемые в классе OdnModel модуля model
Methods:
    fetch_consumers - извлечение данных по consumers из методов и агрегация
    fetch_join_data2gis - извлечение таблицу data2gis из методов, агрегирует и джойн с consumers по fias
    fetch_join_companies - извлечение таблицу companies из методов и джойн с consumers по inn
    fetch_join_kgis - извлечение таблицу KGISBuildings из методов и джойн с consumers по fias
    fetch_join_bfc - извлечение таблицу BuildFeatClean из методов и джойн с consumers по fias
    fetch_join_passports - извлечение таблицу BuildingPassports из методов и джойн с consumers по fias
    join_tables - объединение таблиц в одну
    fetch_profiles - извлечение профилей потребления по fias-ам
    scale_profiles - применение standart scaling для профилей
    month_cons - извлечение кортежей, содержащих дату и переданное потребление из профиля
    get_date_consumption_targets - расширение исходной таблицы и проставление целевых значений
    clean - отчистка и заполнение данных
    cramers_v - вычисление корреляции для категориальных признаков
    num_binary_corr - вычисление корреляций между численными признаками и целевым значением
Variables:
    num_feats - численные признаки
    cat_feats - категориальные признаки
    binary_feats - бинарные признаки
    train_feats - признаки, необходимые для тренировки
'''
import pandas as pd
import requests
import numpy as np
import scipy.stats as ss
import datetime
from dateutil.relativedelta import relativedelta

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)


def fetch_consumers():
    '''Извлечение данных по consumers из методов и агрегация
    Returns:
         pd.DataFrame, содержащий необходимые данные для consumers
    '''
    consumers = pd.DataFrame(requests.get('http://172.25.170.245:8200/api/DataCompare/GetBuKsUr').json())
    consumers = consumers[consumers['object_id'].notna()]
    consumers = consumers[(consumers['period'] != '') & (consumers['fias'] != '')]
    return consumers


def fetch_join_data2gis(consumers):
    '''Извлекает таблицу data2gis из методов, агрегирует и джойнит с consumers по fias
    Args:
        consumers: pd.DataFrame, содержащий таблицу consumers
    Returns:
        pd.DataFrame, содержащий результат джойна сагрегированной data2gis и consumers по fias
    '''
    data2gis = pd.DataFrame(requests.get('http://172.25.170.245:8201/api/DataCompare/GetData2gis').json())
    data2gis['quantity'] = data2gis['quantity'].apply(lambda x: None if x == 'null' else x)
    data2gis['quantity'] = data2gis['quantity'].apply(lambda x: x.replace('-', '0') if isinstance(x, str) else x)
    data2gis['quantity'] = data2gis['quantity'].apply(
        lambda x: x.replace('\u200bцокольный этаж', '200') if isinstance(x, str) else x)
    data2gis['quantity'] = data2gis['quantity'].apply(lambda x: int(x) if isinstance(x, str) else x)
    data2gis = data2gis[
        ['information', 'floors', 'add_info', 'parkings', 'trnsp_acc', 'room_type', 'quantity', 'coordinates',
         'affiliation',
         'fias_id']].groupby('fias_id').agg(information=('information', 'first'), floors=('floors', 'first'),
                                            add_info=('add_info', 'first'), parkings=('parkings', 'first'),
                                            trnsp_acc=('trnsp_acc', 'first'), room_type=('room_type', set),
                                            quantity=('quantity', 'sum'), coordinates=('coordinates', 'first'),
                                            affiliation=('affiliation', set))
    data2gis['room_type'] = data2gis['room_type'].apply(lambda x: ', '.join([str(el) for el in list(x)]).lower())
    data2gis['affiliation'] = data2gis['affiliation'].apply(lambda x: ', '.join([str(el) for el in list(x)]).lower())
    data2gis = consumers.join(data2gis, on='fias')
    return data2gis


def fetch_join_companies(consumers):
    '''Извлекает таблицу companies из методов и джойнит с consumers по inn
    Args:
        consumers: pd.DataFrame, содержащий таблицу consumers
    Returns:
        pd.DataFrame, содержащий результат джойна companies и consumers по inn
    '''
    companies = pd.DataFrame(requests.get('http://172.25.170.245:8201/api/DataCompare/GetSparkCompanies').json())
    companies = companies[
        ['kpp', 'inn', 'okopf', 'director', 'status', 'register', 'enterpriseSize', 'averageHeadcount',
         'authorizedCapital', 'netProfit_2017', 'netProfit_2018', 'netProfit_2019', 'okfs', 'okogu', 'risk', 'okved']]
    companies = companies.set_index('inn')
    companies = consumers.join(companies, on='inn')
    return companies


def fetch_join_kgis(consumers):
    '''Извлекает таблицу KGISBuildings из методов и джойнит с consumers по fias
    Args:
        consumers: pd.DataFrame, содержащий таблицу consumers
    Returns:
        pd.DataFrame, содержащий результат джойна KGISBuildings и consumers по fias
    '''
    kgis = pd.DataFrame(requests.get('http://172.25.170.245:8201/api/DataCompare/GetKgisBuildings').json())
    kgis = kgis[
        ['shortCharacteristic', 'typeBuilding', 'fias', 'yearConstruction', 'signElectrification', 'isNorthRES']]
    kgis = kgis.set_index('fias')
    kgis = consumers.join(kgis, on='fias')
    return kgis


def fetch_join_bfc(consumers):
    '''Извлекает таблицу BuildFeatClean из методов и джойнит с consumers по fias
    Args:
        consumers: pd.DataFrame, содержащий таблицу consumers
    Returns:
        pd.DataFrame, содержащий результат джойна KGISBuildings и consumers по fias
    '''
    bfc = pd.DataFrame(requests.get('http://172.25.170.245:8201/api/Results/GetBuildFeatClean').json())
    bfc = bfc[['fias_id', 'living_area', 'comm_year', 'elev_num', 'floor_num', 'flat_num', 'is_gas']].set_index(
        'fias_id')
    bfc = consumers.join(bfc, on='fias')
    return bfc


def fetch_join_passports(consumers):
    '''Извлекает таблицу BuildingPassports из методов и джойнит с consumers по fias
    Args:
        consumers:  pd.DataFrame, содержащий таблицу consumers
    :return:
         pd.DataFrame, содержащий результат джойна BuildingPassports и consumers по fias
    '''
    passports = pd.DataFrame(requests.get('http://172.25.170.245:8201/api/DataCompare/GetBuildingPassports').json())
    passports = passports[['addrDistrict', 'commType', 'commNum', 'commRoomNum', 'dataSeries',
                           'dataBuildingdate', 'dataReconstructiondate', 'dataBuildingarea',
                           'dataLivingarea', 'dataNonolivingarea', 'dataStairs', 'dataStoreys',
                           'dataResidents', 'dataMansardarea', 'engHeatingcentral',
                           'engHeatingauto', 'engHeatingfire', 'engHotwater', 'engHotwatergas',
                           'engHotwaterwood', 'engElectro', 'engGascentral', 'engGasnoncentral',
                           'engRefusechute', 'flatType', 'liftExploitfromdate',
                           'liftReconstructiondate', 'liftRepairdate', 'outcleanAll',
                           'paramUkname', 'paramFailure', 'repairYear', 'repairJob',
                           'rfcShaftcount', 'roofMetalarea', 'sengLiftcount',
                           'specialBasementarea', 'fias']].set_index('fias')
    passports = consumers.join(passports, on='fias')
    return passports


def join_tables(data2gis, companies, kgis, bfc, passports):
    '''Объединение пяти таблиц
    Объединение пяти таблиц, полученных при помощи методов выше, в одну. При этом, у таблиц companies, kgis, bfc,
    passports удаляются столбцы 'name', 'date', 'address', 'inn', 'consumption', так как они уже содержатся в data2gis.
    Таблицы джойнятся по fias, удаляются дубликаты, так как fias дублируется (один fias, несколько object_id)
    Args:
        data2gis: pd.DataFrame, содержащий таблицу Data2Gis, заджойненную с consumers по fias
        companies: pd.DataFrame, содержащий таблицу Companies, заджойненную с consumers по inn
        kgis: pd.DataFrame, содержащий таблицу KGISBuildings, заджойненную с consumers по fias
        bfc: pd.DataFrame, содержащий таблицу KGISBuildings, заджойненную с consumers по fias
        passports: pd.DataFrame, содержащий таблицу BuildingPassports, заджойненную с consumers по fias
    Returns:
        pd.DataFrame, содержащий объединенные таблицы
    '''
    companies = companies.set_index('fias')
    kgis = kgis.set_index('fias')
    bfc = bfc.set_index('fias')
    passports = passports.set_index('fias')

    for frame in [companies, kgis, bfc, passports]:
        frame.drop(['consumerName', 'period', 'address', 'inn', 'volume', 'object_id'], axis=1, inplace=True)

    data = data2gis.join(kgis, on='fias').drop_duplicates().join(bfc, on='fias').drop_duplicates().join(passports,
                                                                                                        on='fias').drop_duplicates()
    data['period'] = data['period'].apply(lambda x: x.replace('.', '-'))
    return data


def fetch_inference_data():
    '''Извлекает данные для предсказаний. Сначала извлекает данные для fias, object_id, consumer_name из метода
    GetLegalConsumersExt. Затем извлекает данные из GetData2gis, GetKgisBuildings, GetBuildFeatClean,
    GetBuildingPassports и джойнит их с даанными из GetLegalConsumersExt

    Returns:
        pd.DataFrame, содержащий данные для предсказаний
    '''
    inference_data = pd.DataFrame(requests.get('http://172.25.170.245:8200/api/PSK/GetLegalConsumersExt').json())
    inference_data = inference_data[['fias_code', 'object_id', 'type', 'value']]
    inference_data.columns = ['fias', 'object_id', 'consumerName', 'volume']

    data2gis = pd.DataFrame(requests.get('http://172.25.170.245:8201/api/DataCompare/GetData2gis').json())

    data2gis['quantity'] = data2gis['quantity'].apply(lambda x: None if x == 'null' else x)
    data2gis['quantity'] = data2gis['quantity'].apply(lambda x: x.replace('-', '0') if isinstance(x, str) else x)
    data2gis['quantity'] = data2gis['quantity'].apply(
        lambda x: x.replace('\u200bцокольный этаж', '200') if isinstance(x, str) else x)
    data2gis['quantity'] = data2gis['quantity'].apply(lambda x: int(x) if isinstance(x, str) else x)

    data2gis = data2gis[
        ['information', 'floors', 'add_info', 'parkings', 'trnsp_acc', 'room_type', 'quantity', 'coordinates',
         'affiliation',
         'fias_id']].groupby('fias_id').agg(information=('information', 'first'), floors=('floors', 'first'),
                                            add_info=('add_info', 'first'), parkings=('parkings', 'first'),
                                            trnsp_acc=('trnsp_acc', 'first'), room_type=('room_type', set),
                                            quantity=('quantity', 'sum'), coordinates=('coordinates', 'first'),
                                            affiliation=('affiliation', set))

    data2gis['room_type'] = data2gis['room_type'].apply(lambda x: ', '.join([str(el) for el in list(x)]).lower())
    data2gis['affiliation'] = data2gis['affiliation'].apply(lambda x: ', '.join([str(el) for el in list(x)]).lower())
    data2gis = data2gis[data2gis.index.isin(inference_data['fias'].values)]

    kgis = pd.DataFrame(requests.get('http://172.25.170.245:8201/api/DataCompare/GetKgisBuildings').json())
    kgis = kgis[
        ['shortCharacteristic', 'typeBuilding', 'fias', 'yearConstruction', 'signElectrification', 'isNorthRES']]
    kgis = kgis.set_index('fias')
    kgis = kgis[kgis.index.isin(inference_data['fias'].values)]

    bfc = pd.DataFrame(requests.get('http://172.25.170.245:8201/api/Results/GetBuildFeatClean').json())
    bfc = bfc[['fias_id', 'living_area', 'comm_year', 'elev_num', 'floor_num',
               'flat_num', 'is_gas']].set_index('fias_id')
    bfc = bfc[bfc.index.isin(inference_data['fias'].values)]

    passports = pd.DataFrame(requests.get('http://172.25.170.245:8201/api/DataCompare/GetBuildingPassports').json())
    passports = passports[['addrDistrict', 'commType', 'commNum', 'commRoomNum', 'dataSeries',
                           'dataBuildingdate', 'dataReconstructiondate', 'dataBuildingarea',
                           'dataLivingarea', 'dataNonolivingarea', 'dataStairs', 'dataStoreys',
                           'dataResidents', 'dataMansardarea', 'engHeatingcentral',
                           'engHeatingauto', 'engHeatingfire', 'engHotwater', 'engHotwatergas',
                           'engHotwaterwood', 'engElectro', 'engGascentral', 'engGasnoncentral',
                           'engRefusechute', 'flatType', 'liftExploitfromdate',
                           'liftReconstructiondate', 'liftRepairdate', 'outcleanAll',
                           'paramUkname', 'paramFailure', 'repairYear', 'repairJob',
                           'rfcShaftcount', 'roofMetalarea', 'sengLiftcount',
                           'specialBasementarea', 'fias']].set_index('fias')
    passports = passports[passports.index.isin(inference_data['fias'].values)]

    inf_data = inference_data.join(data2gis, on='fias', how='inner').join(kgis, on='fias',
                                                                          how='inner').join(bfc, on='fias',
                                                                                            how='inner').join(passports,
                                                                                                              on='fias',
                                                                                                              how='inner')
    return inf_data


def get_month(date, prev=False):
    '''Метод, принимающий на вход строку и возвращающий значение datetime либо за текущий, либо за предыдущий месяц

    Args:
        date: str, содержащий дату в формате год-месяц
        prev: bool. Если true, возврашает datetime за предыдущий месяц, если False - за текущий
    Returns:
        datetime, содержаший дату
    '''
    if prev:
        return datetime.datetime.strptime(date, '%Y-%m') - relativedelta(months=+1)
    else:
        return datetime.datetime.strptime(date, '%Y-%m')


def expand_by_date(data: pd.DataFrame):
    '''Метод дублирующий строки в датафрейме, добавляя в графу period дублируемой строки дату за предыдущий месяц

    Args:
        pd.DataFrame, содержащий данные из функции join_tables
    Returns:
        pd.DataFrame, содержащий продублируемую таблицу с добавленной датой
    '''
    matrix = data.values

    matrix_exp = np.empty(shape=(data.shape[0] * 2, matrix.shape[1]), dtype=object)
    targets = np.empty(shape=(data.shape[0] * 2,), dtype=object)

    for (i, j) in zip(range(0, matrix.shape[0] * 2, 2), range(matrix.shape[0])):
        matrix_exp[i, :] = np.concatenate([matrix[j, :4], np.array([get_month(matrix[j, 4])]), matrix[j, 5:]])
        matrix_exp[i + 1, :] = np.concatenate(
            [matrix[j, :4], np.array([get_month(matrix[j, 4], prev=True)]), matrix[j, 5:]])
        targets[i] = 0
        targets[i + 1] = 1

    columns = data.columns
    data_ext = pd.DataFrame(matrix_exp)
    data_ext.columns = columns
    return data_ext, targets


def clean(data, is_train=True):
    '''Отчистка и заполнение данных (наиболее частовстречающееся - для категориальных, среднее - для численных)

    Args:
        data: pd.DataFrame, сожержащий расширенную таблицу
    Returns:
        pd.DataFrame, сожержащий отчищенную таблицу
    '''
    data['information'].fillna(value=np.max(data[data['information'].notna()]['information'].values),
                               inplace=True)
    data['information'] = data['information'].apply(lambda x: x.replace('\u200b', '').strip())

    data['floors'].fillna(value=data[data['floors'].notna()]['floors'].value_counts().index[0], inplace=True)
    data['floors'] = data['floors'].apply(lambda x: x.replace('\u200b', ' ').replace('\xa0', ' ').strip())
    data['floors'] = data['floors'].apply(lambda x: [int(word) for word in x.split() if word.isdigit()][0])

    data['add_info'] = data['add_info'].apply(lambda x: np.nan if x == 'null' else x)
    data['add_info'] = data['add_info'].apply(lambda x: np.nan if x == '-' else x)
    data['add_info'].fillna(value=np.max(data[data['add_info'].notna()]['add_info'].values), inplace=True)
    data['add_info'] = data['add_info'].apply(lambda x: x.replace('\xa0', ' ').strip())
    data['add_info'] = data['add_info'].apply(lambda x: x.split())
    data['add_info'] = data['add_info'].apply(lambda x: [int(word) for word in x if word.isdigit()][0])

    data['parkings'] = data['parkings'].apply(lambda x: np.nan if x == 'null' else x)
    data['parkings'] = data['parkings'].apply(lambda x: np.nan if x == '-' else x)
    data['parkings'].fillna(value=np.max(data[data['parkings'].notna()]['parkings'].values), inplace=True)
    data['parkings'] = data['parkings'].apply(lambda x: x.replace('\xa0', ' ').strip())
    data['parkings'] = data['parkings'].apply(lambda x: x.split())
    data['parkings'] = data['parkings'].apply(lambda x: [int(word) for word in x if word.isdigit()][0])

    data['quantity'].fillna(value=np.max(data[data['quantity'].notna()]['quantity'].values), inplace=True)
    data['quantity'] = data['quantity'].apply(lambda x: x.replace('-', '0') if isinstance(x, str) else x)
    data['quantity'] = data['quantity'].apply(
        lambda x: x.replace('\u200bцокольный этаж', '200') if isinstance(x, str) else x)
    data['quantity'] = data['quantity'].apply(lambda x: int(x) if isinstance(x, str) else x)

    data['coordinates'] = data['coordinates'].apply(lambda x: np.nan if x == '-' or x == 'null' else x)
    data['coordinates'].fillna(value=np.max(data[data['coordinates'].notna()]['coordinates'].values), inplace=True)
    data['coordinates'] = data['coordinates'].apply(lambda x: [float(num) for num in x.replace(',', ' ').split()])
    data['longitude'] = data['coordinates'].apply(lambda x: x[0])
    data['latitude'] = data['coordinates'].apply(lambda x: x[1])
    data.drop('coordinates', axis=1, inplace=True)

    data['affiliation'] = data['affiliation'].apply(lambda x: ', '.join([str(el) for el in list(x)]).lower())
    data['affiliation'] = data['affiliation'].apply(lambda x: np.nan if x == '-' else x)
    data['affiliation'] = data['affiliation'].apply(lambda x: np.nan if x == 'none' else x)
    data['affiliation'].fillna(value=np.max(data[data['affiliation'].notna()]['affiliation'].values), inplace=True)

    data['shortCharacteristic'] = data['shortCharacteristic'].apply(lambda x: np.nan if x == ' ' or x == '' else x)
    data['shortCharacteristic'] = data['shortCharacteristic'].apply(lambda x: np.nan if x == '-' else x)
    data['shortCharacteristic'] = data['shortCharacteristic'].apply(lambda x: np.nan if x == '' else x)
    data['shortCharacteristic'] = data['shortCharacteristic'].apply(lambda x: np.nan if x == 'примечание' else x)
    data['shortCharacteristic'] = data['shortCharacteristic'].apply(lambda x: np.nan if x == '9 крылец п' else x)
    data['shortCharacteristic'].fillna(value='отсутствует', inplace=True)

    data['typeBuilding'] = data['typeBuilding'].apply(lambda x: np.nan if x == '' or x == ' ' else x)
    data['typeBuilding'].fillna(value='отсутствует', inplace=True)

    data['signElectrification'].fillna(
        value=np.max(data[data['signElectrification'].notna()]['signElectrification'].values), inplace=True)
    data['signElectrification'] = data['signElectrification'].apply(lambda x: 1 if x == 'Да' else 0)

    data['living_area'].fillna(value=np.mean(data[data['living_area'].notna()]['living_area'].values), inplace=True)

    # из comm_year to age
    data['comm_year'].fillna(value=data[data['comm_year'].notna()]['comm_year'].value_counts().index[0],
                             inplace=True)
    data['age'] = data['comm_year'].apply(lambda x: datetime.datetime.now().year - x)
    data.drop('comm_year', axis=1, inplace=True)

    data['elev_num'].fillna(value=data[data['elev_num'].notna()]['elev_num'].value_counts().index[0],
                            inplace=True)

    data['floor_num'].fillna(value=data[data['floor_num'].notna()]['floor_num'].value_counts().index[0],
                             inplace=True)

    data['flat_num'].fillna(value=data[data['flat_num'].notna()]['flat_num'].value_counts().index[0],
                            inplace=True)

    data['is_gas'].fillna(value=data[data['is_gas'].notna()]['is_gas'].value_counts().index[0],
                          inplace=True)

    data['addrDistrict'].fillna(value=data[data['addrDistrict'].notna()]['addrDistrict'].value_counts().index[0],
                                inplace=True)
    data['addrDistrict'] = data['addrDistrict'].apply(lambda x: x.lower().strip())

    data['commType'].fillna(value='отсутствует', inplace=True)
    data['commType'] = data['commType'].apply(lambda x: x.strip())

    data['dataSeries'].fillna(value='отсутствует', inplace=True)
    data['dataSeries'] = data['dataSeries'].apply(lambda x: x.replace('-', '').replace(' ', '').lower().strip())
    data['dataSeries'] = data['dataSeries'].apply(lambda x: x.replace('инндивидуальный', 'инд'))
    data['dataSeries'] = data['dataSeries'].apply(lambda x: 'инд' if 'инд' in x else x)

    data['dataBuildingdate'].fillna(
        value=data[data['dataBuildingdate'].notna()]['dataBuildingdate'].value_counts().index[0], inplace=True)
    data['dataBuildingdate'] = data['dataBuildingdate'].apply(lambda x: x.replace('до 1917', '1900'))
    data['dataBuildingdate'] = data['dataBuildingdate'].apply(lambda x: int(x))

    data['dataBuildingarea'] = data['dataBuildingarea'].apply(lambda x: float(x) if isinstance(x, str) else x)
    data['dataBuildingarea'].fillna(
        value=np.mean(data[data['dataBuildingarea'].notna()]['dataBuildingarea'].values), inplace=True)

    data['dataLivingarea'] = data['dataLivingarea'].apply(lambda x: x.replace(' ', '') if isinstance(x, str) else x)
    data['dataLivingarea'] = data['dataLivingarea'].apply(lambda x: float(x) if isinstance(x, str) else x)
    data['dataLivingarea'].fillna(value=np.mean(data[data['dataLivingarea'].notna()]['dataLivingarea'].values),
                                  inplace=True)

    data['dataStairs'].fillna(value=data[data['dataStairs'].notna()]['dataStairs'].value_counts().index[0],
                              inplace=True)
    data['dataStairs'] = data['dataStairs'].apply(lambda x: int(x) if isinstance(x, str) else x)

    data['dataStoreys'].fillna(value=data[data['dataStoreys'].notna()]['dataStoreys'].value_counts().index[0],
                               inplace=True)
    data['dataStoreys'] = data['dataStoreys'].apply(lambda x: int(x))

    data['dataResidents'] = data['dataResidents'].apply(
        lambda x: int(float(x.replace(',', '.'))) if isinstance(x, str) else x)
    data['dataResidents'].fillna(value=data[data['dataResidents'].notna()]['dataResidents'].value_counts().index[0],
                                 inplace=True)
    data['dataResidents'] = data['dataResidents'].apply(lambda x: int(x))

    data['engHeatingcentral'].fillna(
        value=data[data['engHeatingcentral'].notna()]['engHeatingcentral'].value_counts().index[0], inplace=True)
    data['engHeatingcentral'] = data['engHeatingcentral'].apply(lambda x: float(x))

    data['engHeatingauto'].fillna(value=data[data['engHeatingauto'].notna()]['engHeatingauto'].value_counts().index[0],
                                  inplace=True)
    data['engHeatingauto'] = data['engHeatingauto'].apply(lambda x: float(x))

    data['engHotwater'].fillna(value=data[data['engHotwater'].notna()]['engHotwater'].value_counts().index[0],
                               inplace=True)
    data['engHotwater'] = data['engHotwater'].apply(lambda x: float(x))

    data['engHotwatergas'].fillna(value=data[data['engHotwatergas'].notna()]['engHotwatergas'].value_counts().index[0],
                                  inplace=True)
    data['engHotwatergas'] = data['engHotwatergas'].apply(lambda x: float(x))

    data['engGascentral'].fillna(value=data[data['engGascentral'].notna()]['engGascentral'].value_counts().index[0],
                                 inplace=True)
    data['engGascentral'] = data['engGascentral'].apply(lambda x: float(x))

    data['engRefusechute'].fillna(value=data[data['engRefusechute'].notna()]['engRefusechute'].value_counts().index[0],
                                  inplace=True)
    data['engRefusechute'] = data['engRefusechute'].apply(lambda x: float(x))

    data['flatType'].fillna(value='отсутствует', inplace=True)
    data['flatType'] = data['flatType'].apply(lambda x: x.strip())

    data['outcleanAll'] = data['outcleanAll'].apply(lambda x: x.lower() if isinstance(x, str) else x)
    data['outcleanAll'] = data['outcleanAll'].apply(
        lambda x: np.nan if isinstance(x, str) and 'жилкомсервис' in x else x)
    data['outcleanAll'] = data['outcleanAll'].apply(
        lambda x: float(x.replace(' ', '').strip()) if isinstance(x, str) else x)
    data['outcleanAll'].fillna(value=np.mean(data[data['outcleanAll'].notna()]['outcleanAll'].values).round(),
                               inplace=True)

    data['specialBasementarea'] = data['specialBasementarea'].apply(
        lambda x: float(x.replace(' ', '').strip()) if isinstance(x, str) else x)
    data['specialBasementarea'].fillna(
        value=np.mean(data[data['specialBasementarea'].notna()]['specialBasementarea'].values).round(),
        inplace=True)

    data['volume'] = data['volume'].apply(lambda x: '96460.00' if x == '96,460.00' else x)
    data['volume'] = data['volume'].apply(lambda x: float(x))

    if is_train:
        data['period'].fillna(value=data[data['period'].notna()]['period'].value_counts().index[0],
                              inplace=True)
        data['period'] = data['period'].apply(lambda x: int(str(x)[5:7]))
    return data


def cramers_v(x, y):
    '''Вычисление корреляции для категориальных признаков
    Args:
        x: pd.Series, содержащий значения одного категориального признака
        y: pd.Series, содержащий значения другого категориального признака
    Returns:
        float значение корреляции
    '''
    confusion_matrix = pd.crosstab(x, y)
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    rcorr = r - ((r - 1) ** 2) / (n - 1)
    kcorr = k - ((k - 1) ** 2) / (n - 1)
    return np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1)))


def num_binary_corr(data_targ, feat: str):
    '''Вычисление корреляций между численными признаками и целевым значением
    Args:
        data_targ: pd.DataFrame, содержащий таблицу с данными, включая целевое значение (target)
        feat: название признака str
    Returns:
        значение корреляции float
    '''
    pos_mean = data_targ[num_feats + ['targets']][data_targ['targets'] == 1][feat].mean()
    neg_mean = data_targ[num_feats + ['targets']][data_targ['targets'] == 0][feat].mean()
    return pos_mean - neg_mean


num_feats = ['period', 'floors', 'add_info', 'parkings', 'signElectrification', 'quantity', 'longitude', 'latitude',
             'living_area', 'age', 'elev_num', 'floor_num', 'flat_num', 'is_gas', 'dataBuildingdate',
             'dataBuildingarea', 'dataLivingarea',
             'dataStairs', 'dataStoreys', 'dataResidents', 'engHeatingcentral', 'engHeatingauto', 'engHotwater',
             'engHotwatergas',
             'engGascentral', 'engRefusechute', 'outcleanAll', 'specialBasementarea', 'volume']

cat_feats = ['consumerName', 'information', 'affiliation', 'shortCharacteristic', 'typeBuilding', 'addrDistrict',
             'commType', 'dataSeries', 'flatType']

binary_feats = ['signElectrification', 'is_gas', 'engHeatingcentral', 'engHeatingauto', 'engHotwater', 'engHotwatergas',
                'engGascentral',
                'engRefusechute']

train_feats = num_feats + cat_feats