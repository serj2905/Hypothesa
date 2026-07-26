# Как сдавать ДЗ №7

Готовый файл: `hw_7_model_improvement.ipynb`.

## Соответствие требованиям

| Критерий | Что находится в ноутбуке | Баллы |
|---|---|---:|
| Пайплайн предобработки / FE | Leakage-safe очистка, маскирование URL/user, word TF-IDF, сквозные char TF-IDF, 14 структурных признаков, единый `FeatureStack` | 3/3 |
| Улучшенная архитектура | Early fusion трёх блоков + LinearSVC, validation-only подбор параметров, ablation study, прямое сравнение с previous best | 4/4 |
| Постобработка | Margin threshold, выбранный только на validation; `accepted / needs_review`; risk–coverage curve и inference-контракт | 3/3 |
| Анализ качества | Accuracy, Macro-F1, Balanced Accuracy, Top-2, Q, per-class report, confusion matrices, срезы, ошибки, paired bootstrap, sensitivity check | 5/5 |

По структуре ноутбука закрыты все пункты рубрики на 15 возможных баллов; итоговая
оценка остаётся за проверяющим.

## Что получилось

В ноутбуке previous best из ДЗ №5–6 честно зафиксирован до эксперимента. Новая модель
добавляет расширенные word n-граммы, сквозные character n-граммы и структурные
признаки текста. Параметры выбираются на validation из train, официальный test не
используется для настройки.

В воспроизводимом прогоне:

- validation Macro-F1: `0.7393 → 0.7626`;
- официальный test Macro-F1: `0.7559 → 0.7629`;
- официальный test Accuracy: `0.7563 → 0.7633`;
- после validation-only постобработки принимается 90.2% test-ответов с Accuracy
  `0.7953`.

95% paired bootstrap CI для прироста Macro-F1 включает ноль, и это прямо указано в
выводе. Результат нужно формулировать аккуратно: это положительное улучшение на
публичном proxy-корпусе SentiRuEval-2016, а не доказанное production-качество на
интервью Hypothesa.

## Публикация

1. Загрузить `hw_7_model_improvement.ipynb` в Google Colab.
2. Выполнить `Runtime → Run all`.
3. Сохранить копию на Google Drive.
4. Включить доступ «Все, у кого есть ссылка → Читатель».
5. Проверить ссылку в приватном окне: должны быть видны код, таблицы, графики и выводы.
6. Отправить ссылку на Colab-ноутбук.

Текст для поля ответа:

> ДЗ №7 по проекту Hypothesa: улучшение вспомогательного классификатора тональности
> русскоязычных ответов. Реализованы leakage-safe preprocessing и feature engineering,
> улучшенная word+char+style early-fusion архитектура с validation-only выбором
> параметров, post-processing неуверенных ответов и подробный анализ качества в
> разрезе общих и поклассовых метрик, срезов, bootstrap CI и risk–coverage. Для
> offline-оценки использован публичный proxy-корпус SentiRuEval-2016; необходимость
> отдельной domain-валидации на размеченных ответах Hypothesa явно указана.
