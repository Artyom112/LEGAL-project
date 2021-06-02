from catboost import CatBoostClassifier
from methods import *
import os


class TrainedModel:
    '''Класс, подготавливающий данные и осущетвляющий предсказания на этих данных
    Methods:
        __init__ - инициализация данных и отчистка данных, объявление директивы, в которой хранится модель
        predict - добавление даты, для которой нужно сделать предсказание и осуществление предсказаний
    Usage examples:
        инициализируйте объект класса: model = TrainedModel()
        сделайте предсказания на июнь на основе даты preds_json = model.predict(6)
    '''
    def __init__(self):
        '''
        Извлечение необходимых данных, отчитстка, трансформация, загрузка модели из директивы
        '''
        data = fetch_inference_data()
        self.data = clean(data, is_train=False)

        path = os.path.dirname(__file__) + '/legal_inference_model.cbm'
        from_file = CatBoostClassifier()
        self.model = from_file.load_model(path)

    def predict(self, date: int):
        '''Добавление даты, для которой необходимо осушествить предсказание и осуществление предсказаний
        Args:
            date: int для необходимого месяца, 1 - январь, 12 - декабрь
        Returns:
             list, содержащий элементы типа dict со следующими парами key-value: 'fias' - str, object_id - str,
             period - int, predictions - int, probs - float
        '''
        self.data['period'] = [date for _ in range(self.data.shape[0])]
        predictions = self.model.predict(self.data[train_feats].values)
        probs = self.model.predict_proba(self.data[train_feats].values)
        self.data['predictions'] = predictions
        self.data['probs'] = probs[:, 1]
        result_json = self.data[['fias', 'object_id', 'period', 'predictions', 'probs']].to_json(orient="records")
        return result_json


if __name__ == '__main__':
    model = TrainedModel()
    result_json = model.predict(6)
    # pos_num = sum(pred for pred in result_json['predictions'])
    # neg_num = len(result_json) - pos_num

