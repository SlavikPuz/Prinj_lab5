import pytest
import pandas as pd
from core import survival_percent, young_old_survival_by_class

# 1) Базовый кейс: смешанная выборка
def test_survival_percent_mixed():
    df = pd.DataFrame({"Survived": [1, 0, 1, 0, 1]})
    assert survival_percent(df) == pytest.approx(60.0)

# 2) Пустая выборка -> 0
def test_survival_percent_empty():
    df = pd.DataFrame({"Survived": []})
    assert survival_percent(df) == 0.0

# 3) Все выжили -> 100
def test_survival_percent_all():
    df = pd.DataFrame({"Survived": [1, 1, 1]})
    assert survival_percent(df) == 100.0

# 4) Интеграционный тест без файлов
def test_young_old_by_class():
    df = pd.DataFrame({
        "Pclass":   [1, 1, 1, 2, 2, 2, 2],
        "Age":      [25, 65, 40, 22, 18, 70, 61],
        "Survived": [1,  0,  1, 0,  1,  1,  0],
    })
    # Класс 1: молодые (<30): [25] -> 100%; старые (>60): [65] -> 0%
    y1, o1 = young_old_survival_by_class(df, 1)
    assert y1 == 100.0
    assert o1 == 0.0

    # Класс 2: молодые (<30): [22,18] -> 50%; старые (>60): [70,61] -> 50%
    y2, o2 = young_old_survival_by_class(df, 2)
    assert y2 == pytest.approx(50.0)
    assert o2 == pytest.approx(50.0)
