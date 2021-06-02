import seaborn as sns
import matplotlib.pyplot as plt
from catboost import CatBoostClassifier, CatBoostError
from sklearn.model_selection import train_test_split
import os
from methods import *


class LegalModel:
    '''Класс, подготавливающий данные для тренировки, и использующий градиентный бустинг на лесах (catboost) как модель
    бинарной классификации
    Methods:
        __init__ - инициализация данных, разбивка на train и test, объявление директивы
        pairplot -попарная визуализация признаков по данным
        plot_categorical_correlations - визуализация heatmap между категориальными признаками
        cat_correlations_with_target - вычисление корреляции категориальных признаков с целевым
        plot_numerical_correlations - визуализация heatmap между численными признаками
        numerical_correlations_with_target - вычисление корреляции численных признаков с целевым бинарным
        train - тренировка модели
        feature_importances - вычисление важности признаков
        predict - осушествление бинарных предсказаний
        predict_proba - вычисление вероятности воровства
    Usage examples:
        инициализируйте объект класса: model = LegalModel()
        затренируйте модель на данных, полученных во время инициализации объекта класса: model.train()
        сделайте предсказания на новых данных: model.predict(X_new)
    '''
    def __init__(self):
        '''Подготавливает данные, разбивает их на train и test, объявляет директиву, в которой сохраняется модель
        Конструктор класса OdnModel извлекает данные, необходимые для тренировки модели, из методов и объединяет
        эти таблицы в одну. Также извлекает профили потребления по имеющимся фиасам и производит стандартизацию значений
        value. Затем чистит данные, производит разбиение на train и test и объявляет директиву, в которой сохраняется
        затренированная модель
        '''
        consumers = fetch_consumers()
        data2gis = fetch_join_data2gis(consumers)
        companies = fetch_join_companies(consumers)
        kgis = fetch_join_kgis(consumers)
        bfc = fetch_join_bfc(consumers)
        passports = fetch_join_passports(consumers)
        data = join_tables(data2gis, companies, kgis, bfc, passports)

        data, targets = expand_by_date(data)
        data = clean(data)
        data['targets'] = targets

        # Оставляем только необходимые признаки + целевой
        self.data = data[train_feats + ['targets']]

        # Индексы категориальных признаков
        self.cat_indices = [self.data.columns.get_loc(x) for x in cat_feats]

        # Разбиваем на тренировочную и тестовую выборки
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.data.iloc[:, :-1].values,
                                                                self.data.iloc[:, -1].values, train_size=0.9)

        # Путь к сохраненной модели
        self.path = os.path.dirname(__file__) + '/legal_inference_model.cbm'

    def pairplot(self):
        '''Попарная визуализация признаков по данным, объявленным в __init__
        Returns:
            None
        '''
        plt.figure()
        sns.pairplot(self.data)
        plt.show()

    def plot_categorical_correlations(self):
        '''Визуализация heatmap между категориальными признаками, используя данные, объявленные в __init__
        Returns:
            None
        '''
        cat_corrs = np.zeros(shape=(len(cat_feats), len(cat_feats)))
        for i, feat1 in enumerate(cat_feats):
            for j, feat2 in enumerate(cat_feats):
                if j < i:
                    cat_corrs[i, j] = cat_corrs[j, i]
                elif i == j:
                    cat_corrs[i, j] = 1
                else:
                    cat_corrs[i, j] = cramers_v(self.data[feat1], self.data[feat2])

        corrs_frame = pd.DataFrame(cat_corrs)
        corrs_frame.columns = cat_feats
        corrs_frame.index = cat_feats

        plt.figure(figsize=(20, 10), dpi=500)
        x = sns.heatmap(corrs_frame, vmin=-1, vmax=1, annot=True)
        x.set_title('Корреляции между категориальными величинами', fontdict={'fontsize': 12}, pad=12)
        plt.show()

    def cat_correlations_with_target(self):
        '''Вычисление корреляции категориальных признаков с целевым, используя данные, объявленные в __init__
        Returns:
            pd.DataFrame, сожержащий корреляции категориальных признаков с целевым
        '''
        targ_corrs = np.zeros(shape=(len(cat_feats),))

        for i, feat in enumerate(cat_feats):
            targ_corrs[i] = cramers_v(self.data[feat], self.data['targets'])

        targs_frame = pd.Series(targ_corrs)
        targs_frame.columns = cat_feats
        targs_frame.index = cat_feats
        return targs_frame

    def plot_numerical_correlations(self, method='pearson'):
        '''Визуализация heatmap между численными признаками, используя данные, объявленные в __init__
        Args:
            method: параметр типа str, определяющий вид корреляции между непрерывными признаками
        Returns:
            None
        '''
        plot_feats = [feat for feat in num_feats if feat not in binary_feats]
        plt.figure(figsize=(20, 10), dpi=500)
        x = sns.heatmap(self.data[plot_feats].corr(method=method), vmin=-1, vmax=1, annot=True)
        x.set_title('Корреляции Пирсона между вещественными величинами', fontdict={'fontsize': 12}, pad=12)
        plt.show()

    def numerical_correlations_with_target(self):
        '''Вычисление корреляции численных признаков с целевым бинарным
        Returns:
            Dict, соотносящий имя признака с соответствующей корреляцией с целевым бинарным признаком
        '''
        corrs = {feat: num_binary_corr(self.data, feat) for feat in num_feats}
        return corrs

    def train(self, metrics: list):
        '''Тренировка модели
        Метод создает объект класса CatBoostClassifier. Тренирует и производит валидацию модели на данных,
        объявленныйх в конструкторе __init__. Затем сохраняет модель в директиве, объявленной в __init__
        Args:
            metrics: массив типа list, состоящий из элементов типа str с названиями вычисляемых метрик
        Returns:
            None
        '''
        model = CatBoostClassifier(
            loss_function='Logloss',
            learning_rate=0.01,
            custom_loss=metrics,
            early_stopping_rounds=50,
            iterations=3000
        )

        model.fit(
            self.X_train, self.y_train,
            cat_features=self.cat_indices,
            eval_set=(self.X_test, self.y_test),
            verbose=True,
            plot=False
        )

        model.save_model(self.path, format='cbm')

    def feature_importances(self):
        '''Вычисление важности признаков, используемых во время тренировки на основе затренированной можели
        Returns: pd.DataFrame, сожержащий важность признаков, используемых во время тренировки на основе затренированной модели
        Raises:
            CatBoostError: Ошибка, возникающая в случае отсутствия классификатора в директиве
        '''
        from_file = CatBoostClassifier()
        try:
            model = from_file.load_model(self.path)
            name_mapping = {str(idx): name for idx, name in zip(range(39), self.data.iloc[:, :-1].columns)}
            importances = model.get_feature_importance(prettified=True)
            importances['name'] = importances['Feature Id'].apply(lambda x: name_mapping[x])
            return importances
        except CatBoostError:
            print('Модели не существует. Пожалуйста, затренируйте модель')

    def predict(self, X):
        '''Осушествление предсказаний
        Модель выдает предсказание воровства, используя новые данные (порог бинаризации - 0.5)
        Args:
            X: np.array с данными, для которых нужно сделать предсказание (схожие по структуре с self.X_test)
        Returns: np.array размерностью (X.shape[0], ) с бинарными значениями (1 - воровал, 2 - не воровал)
        Raises:
            CatBoostError: Ошибка, возникающая в случае отсутствия классификатора в директиве
        '''
        from_file = CatBoostClassifier()
        try:
            model = from_file.load_model(self.path)
            predictions = model.predict(X)
            return predictions
        except CatBoostError:
            print('Модели не существует. Пожалуйста, затренируйте модель')

    def predict_proba(self, X):
        '''Вычисление вероятности воровства
        Вычисление вероятности воровства на основе входных данных. Элементы второй колонки матрицы - вероятности
        воровства, элементы первой колонки матрицы - вероятности отсутствия воровства
        Args:
            X: np.array с данными, для которых нужно сделать предсказание (схожие с self.X_test)
        Returns:
            np.array размерностью (X.shape[0], 2) с вероятностями отсутствия воровства и воровства
        Raises:
            CatBoostError: Ошибка, возникающая в случае отсутствия классификатора в директиве
        '''
        from_file = CatBoostClassifier()
        try:
            model = from_file.load_model(self.path)
            probabilities = model.predict_proba(X)
            return probabilities
        except CatBoostError:
            print('Модели не существует. Пожалуйста, затренируйте модель')


if __name__ == '__main__':
    model = LegalModel()
    model.train(['Accuracy', 'Precision', 'Recall', 'F1'])

