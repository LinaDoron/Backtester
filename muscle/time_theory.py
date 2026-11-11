import numpy as np
import pandas as pd


def load_and_prep_data(filepath: str, tz_broker: str = "UTC") -> pd.DataFrame:
    df = pd.read_csv(filepath)
    df['time'] = pd.to_datetime(df['time'], utc=True)  # Yahoo отдает в UTC
    df = df.set_index('time')
    df = df[~df.index.duplicated(keep='last')].sort_index()

    # Переводим в Нью-Йорк. Все окна SMT считаются уже в этом поясе.
    df.index = df.index.tz_convert("America/New_York")

    return add_quarterly_theory_labels(df)


def add_quarterly_theory_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Индекс обязан быть America/New_York.

    1. True day (H1): [18:00, следующие 18:00). Метка — дата закрытия окна.
       18:00 понедельника принадлежит уже следующему true day.
    2. Macro quarter (M15): 18:00–00:00, 00:00–06:00, 06:00–12:00, 12:00–18:00.
    3. Micro quarter (M5): каждый 6-часовой квартал делится на 4 части по 90 минут.
    4. Week (H4): с открытия рынка в воскресенье 17:00 до закрытия в пятницу 17:00.
       Метка — дата понедельника этой недели.

    macro_q и micro_q уникальны для каждого окна. Иначе срез 'Q1' забирает
    все кварталы истории, а не один.
    """
    shifted = df.index + pd.Timedelta(hours=6)
    df['true_date'] = shifted.strftime('%Y-%m-%d')

    hours = df.index.hour
    macro_name = pd.Series(
        np.select(
            [hours >= 18, hours < 6, hours < 12],
            ['Q1', 'Q2', 'Q3'],
            default='Q4',
        ),
        index=df.index,
    )
    df['macro_q'] = df['true_date'] + '_' + macro_name

    minutes_since_macro_start = (df.index.hour % 6) * 60 + df.index.minute
    micro_num = (minutes_since_macro_start // 90) + 1
    df['micro_q'] = df['macro_q'] + '_M' + pd.Series(micro_num, index=df.index).astype(str)

    # Настенное время NY + 7ч: воскресенье 17:00 становится понедельником 00:00.
    week_clock = df.index.tz_localize(None) + pd.Timedelta(hours=7)
    week_start = week_clock.normalize() - pd.to_timedelta(week_clock.weekday, unit='D')
    df['week'] = week_start.strftime('%Y-%m-%d')

    return df
