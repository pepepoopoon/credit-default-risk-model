# Данные

Вход — CSV с `ID`, `LIMIT_BAL`, `SEX`, `EDUCATION`, `MARRIAGE`, `AGE`, шестью полями
задержек (`PAY_0`, `PAY_2`…`PAY_6`), `BILL_AMT1`…`BILL_AMT6`, `PAY_AMT1`…`PAY_AMT6` и
бинарной целью `default`. Также поддерживаются исходные имена UCI `X1`…`X23` и `Y`.

Источник: [UCI Default of Credit Card Clients](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients),
DOI `10.24432/C55S3H`, лицензия CC BY 4.0, автор I-Cheng Yeh. Исходный XLS не скачивается
автоматически; при преобразовании сохраните атрибуцию и зафиксируйте checksum своей версии.

`make smoke` создаёт локальный CSV с фиксированным seed. Это не симулятор кредитного
портфеля и не основание для содержательных выводов.
