# muscle/smc_alphabet.py
import pandas as pd
import numpy as np

# Что видит код стратегии через `from muscle.smc_alphabet import *`. pd и np сюда не входят:
# через них можно читать файлы и ходить в сеть.
__all__ = [
    'resample_to_timeframe',
    'find_smt_chain',
    'trade_side_for_asset',
    'sweep_time',
    'check_psp',
    'check_tpd',
    'check_dfr',
    'find_mss',
    'find_fvg',
    'find_ifvg',
    'find_bpr',
    'find_entry_levels',
    'get_order_block',
    'find_order_block_after',
    'find_breaker_block',
    'find_rejection_block',
    'find_propulsion_block',
    'simulate_trade_with_rm',
]

# H4 = недели. H1 = сравнение true day и свечи 1h. M15 = 6-часовые кварталы. M5 = 90 минут.
# D1 — только свечи, кварталов у него нет.
LEVEL_DURATION = {
    'week': pd.Timedelta(days=5),
    'true_date': pd.Timedelta(hours=24),
    'macro_q': pd.Timedelta(hours=6),
    'micro_q': pd.Timedelta(minutes=90),
}
TF_TO_FREQ = {
    'D1': '1D',
    'H4': '4h',
    'H1': '1h',
    'M15': '15min',
    'M5': '5min',
    '1D': '1D',
    '4h': '4h',
    '1h': '1h',
    '15min': '15min',
    '5min': '5min',
}
# H4 и D1 открываются в 17:00 по NY, как у брокера. Считаем по настенному времени,
# иначе после перехода на зимнее время свечи съезжают на час.
WALL_CLOCK_OFFSET = {
    '4h': pd.Timedelta(hours=1),
    '1D': pd.Timedelta(hours=17),
}
# Лимитка OB живет 5 свечей ТФ OB, начиная с третьей: H1 — 5 часов, M15 — 75 минут.
OB_TEST_BARS = 5
# Минимальный свип для SMT: 1 пипс у EURUSD, GBPUSD и USDCHF.
MIN_SWEEP = 0.0001
PRICE_EPS = 1e-9
# Rejection Block: тень не меньше этой доли свечи, допуск на равенство close/open двух свечей.
RB_MIN_WICK_SHARE = 0.5
RB_LEVEL_TOLERANCE = PRICE_EPS


def get_safe_loc(data, time_val):
    """
    Безопасный поиск индекса свечи.
    Гарантированно возвращает целое число или None.
    """
    if data is None or data.empty:
        return None
    time_val = _match_index_tz(data, time_val)
    matches = np.where(data.index == time_val)[0]
    if len(matches) == 0:
        return None
    return int(matches[-1])


def _match_index_tz(data, time_val):
    ts = pd.Timestamp(time_val)
    idx_tz = data.index.tz
    if idx_tz is not None:
        if ts.tzinfo is None:
            return ts.tz_localize(idx_tz)
        return ts.tz_convert(idx_tz)
    if ts.tzinfo is not None:
        return ts.tz_localize(None)
    return ts


def _row_at(data, time_val):
    loc = get_safe_loc(data, time_val)
    if loc is None:
        return None
    return data.iloc[loc]


def _label_at(data, time_val, column):
    row = _row_at(data, time_val)
    if row is None or column not in data.columns:
        return None
    return row[column]


def trade_side_for_asset(smt_type, asset):
    """
    Тип SMT задан в логике EUR/GBP.
    USDCHF — обратная пара: медвежий SMT для евро = лонг по USDCHF.
    """
    if asset == 'CHF' and smt_type == 'bearish':
        return 'bullish'
    if asset == 'CHF' and smt_type == 'bullish':
        return 'bearish'
    return smt_type


def resample_to_timeframe(df, freq='1h'):
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    agg = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'true_date': 'first',
        'macro_q': 'first',
        'micro_q': 'first',
        'week': 'first',
    }
    agg = {col: how for col, how in agg.items() if col in df.columns}
    offset = WALL_CLOCK_OFFSET.get(freq)
    if offset is None:
        return df.resample(freq, closed='left', label='left').agg(agg).dropna(
            subset=['open', 'high', 'low', 'close']
        )

    tz = df.index.tz
    local = df.copy()
    if tz is not None:
        local.index = df.index.tz_localize(None)
    resampled = local.resample(freq, closed='left', label='left', offset=offset).agg(agg).dropna(
        subset=['open', 'high', 'low', 'close']
    )
    if tz is not None:
        resampled.index = resampled.index.tz_localize(tz, ambiguous='NaT', nonexistent='shift_forward')
        resampled = resampled[resampled.index.notna()]
    return resampled


def _period_ids(df, level):
    if df is None or df.empty or level not in df.columns:
        return []
    return sorted(pd.unique(df[level]), key=str)


def _limit_deadline(data, origin, level):
    """
    Последняя свеча следующего квартала после origin.
    Лимитка живет до его закрытия включительно.
    """
    allowed = {origin, _next_period_id(origin, level)}
    deadline = None
    seen = False
    for ts, label in zip(data.index, data[level]):
        if label in allowed:
            deadline = ts
            seen = True
        elif seen:
            break
    return deadline


def _next_period_id(period_id, level):
    """Следующий квартал по календарю, даже если в данных там нет свечей."""
    if level == 'week':
        nxt = pd.Timestamp(period_id) + pd.Timedelta(days=7)
        return nxt.strftime('%Y-%m-%d')
    if level == 'true_date':
        nxt = pd.Timestamp(period_id) + pd.Timedelta(days=1)
        return nxt.strftime('%Y-%m-%d')
    if level == 'macro_q':
        day, quarter = str(period_id).split('_')
        number = int(quarter[1:])
        if number < 4:
            return f'{day}_Q{number + 1}'
        nxt = (pd.Timestamp(day) + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
        return f'{nxt}_Q1'
    if level == 'micro_q':
        parent, micro = str(period_id).rsplit('_', 1)
        number = int(micro[1:])
        if number < 4:
            return f'{parent}_M{number + 1}'
        return _next_period_id(parent, 'macro_q') + '_M1'
    return None


def _slice_period(df, level, period_id):
    if df is None or df.empty:
        return df
    return df[df[level] == period_id]


def find_smt(prev_eur, prev_gbp, prev_chf, curr_eur, curr_gbp, curr_chf, kind=None):
    """
    SMT между двумя уже нарезанными окнами (true day, 6ч или 90мин).
    Франк инвертирован: его лой = хай евро, его хай = лой евро.
    Снял — пробил уровень прошлого окна минимум на MIN_SWEEP.
    Время SMT — самый ранний такой съем внутри текущего окна.
    kind: 'bearish' или 'bullish' — искать только эту сторону. None — сначала медвежий.
    levels в ответе — цена, пробой которой засчитан как свип.
    """
    frames = (prev_eur, prev_gbp, prev_chf, curr_eur, curr_gbp, curr_chf)
    if any(frame is None or len(frame) == 0 for frame in frames):
        return None

    up = MIN_SWEEP - PRICE_EPS
    eur_prev_high, eur_prev_low = prev_eur['high'].max() + up, prev_eur['low'].min() - up
    gbp_prev_high, gbp_prev_low = prev_gbp['high'].max() + up, prev_gbp['low'].min() - up
    chf_prev_high, chf_prev_low = prev_chf['high'].max() + up, prev_chf['low'].min() - up

    eur_curr_high, eur_curr_low = curr_eur['high'].max(), curr_eur['low'].min()
    gbp_curr_high, gbp_curr_low = curr_gbp['high'].max(), curr_gbp['low'].min()
    chf_curr_high, chf_curr_low = curr_chf['high'].max(), curr_chf['low'].min()

    def _first_breach(frame, column, level, above):
        values = frame[column]
        hit = frame.index[values > level] if above else frame.index[values < level]
        if len(hit) == 0:
            return None
        return hit.min()

    def _pack(kind, sweeps, times, levels):
        swept_by = [asset for asset, swept in sweeps.items() if swept]
        failed_by = [asset for asset, swept in sweeps.items() if not swept]
        if not swept_by or not failed_by:
            return None
        formed = [times[asset] for asset in swept_by if times[asset] is not None]
        if not formed:
            return None
        return {
            'type': kind,
            'swept': swept_by,
            'failed': failed_by,
            'time': min(formed),
            'levels': {asset: float(levels[asset]) for asset in swept_by},
        }

    if kind not in (None, 'bearish', 'bullish'):
        return None

    if kind in (None, 'bearish'):
        bearish = _pack(
            'bearish',
            {
                'EUR': eur_curr_high > eur_prev_high,
                'GBP': gbp_curr_high > gbp_prev_high,
                'CHF': chf_curr_low < chf_prev_low,
            },
            {
                'EUR': _first_breach(curr_eur, 'high', eur_prev_high, True),
                'GBP': _first_breach(curr_gbp, 'high', gbp_prev_high, True),
                'CHF': _first_breach(curr_chf, 'low', chf_prev_low, False),
            },
            {'EUR': eur_prev_high, 'GBP': gbp_prev_high, 'CHF': chf_prev_low},
        )
        if bearish is not None or kind == 'bearish':
            return bearish

    return _pack(
        'bullish',
        {
            'EUR': eur_curr_low < eur_prev_low,
            'GBP': gbp_curr_low < gbp_prev_low,
            'CHF': chf_curr_high > chf_prev_high,
        },
        {
            'EUR': _first_breach(curr_eur, 'low', eur_prev_low, False),
            'GBP': _first_breach(curr_gbp, 'low', gbp_prev_low, False),
            'CHF': _first_breach(curr_chf, 'high', chf_prev_high, True),
        },
        {'EUR': eur_prev_low, 'GBP': gbp_prev_low, 'CHF': chf_prev_high},
    )


def find_smt_between_periods(eur, gbp, chf, level, kind=None):
    """
    Сравнивает соседние окна одного уровня.
    level: 'true_date' (18:00–18:00), 'macro_q' (6 часов), 'micro_q' (90 минут).
    kind: искать только эту сторону SMT, None — любую.
    """
    if level not in LEVEL_DURATION:
        return []

    ids = _period_ids(eur, level)
    found = []
    for i in range(1, len(ids)):
        prev_id, curr_id = ids[i - 1], ids[i]
        smt = find_smt(
            _slice_period(eur, level, prev_id),
            _slice_period(gbp, level, prev_id),
            _slice_period(chf, level, prev_id),
            _slice_period(eur, level, curr_id),
            _slice_period(gbp, level, curr_id),
            _slice_period(chf, level, curr_id),
            kind=kind,
        )
        if smt is None:
            continue
        smt['level'] = level
        smt['period'] = curr_id
        smt['prev_period'] = prev_id
        found.append(smt)
    return found


def find_smt_chain(eur, gbp, chf, levels, kind=None):
    """
    Связка SMT. Пример: ['true_date', 'macro_q'] —
    сначала SMT между днями, затем внутри ЭТОГО дня и СТРОГО после него
    SMT между 6-часовыми кварталами в ту же сторону. Нет младшего SMT — вся связка пропускается.
    """
    if not levels:
        return []
    for level in levels:
        if level not in LEVEL_DURATION:
            raise ValueError(f"Неизвестный уровень SMT: {level}")

    if len(levels) == 1:
        return [[smt] for smt in find_smt_between_periods(eur, gbp, chf, levels[0], kind)]

    chains = []
    parent_level = levels[0]
    for parent in find_smt_between_periods(eur, gbp, chf, parent_level, kind):
        subchains = find_smt_chain(
            _slice_period(eur, parent_level, parent['period']),
            _slice_period(gbp, parent_level, parent['period']),
            _slice_period(chf, parent_level, parent['period']),
            levels[1:],
            kind=parent['type'],
        )
        for sub in subchains:
            if sub[0]['time'] > parent['time']:
                chains.append([parent] + sub)
    return chains


def _breaks_up(smt_type, asset):
    """Медвежий SMT: EUR/GBP снимают хай, CHF — лой. Бычий наоборот."""
    up = smt_type == 'bearish'
    return not up if asset == 'CHF' else up


def sweep_time(frames, smt, asset):
    """
    Момент SMT на свечах frames (ТФ шифта или входа).
    smt['time'] — начало свечи ТФ SMT, свип внутри нее мог быть позже.
    Ищем первую свечу frames, где любой из снявших реально пробил уровень.
    На том же ТФ результат равен smt['time'].
    Возвращает время свечи из frames[asset] или None.
    """
    start = None
    for swept in smt['swept']:
        df = frames.get(swept)
        if df is None or df.empty:
            continue
        level = smt['levels'][swept]
        window = df[df.index >= _match_index_tz(df, smt['time'])]
        if _breaks_up(smt['type'], swept):
            hit = window.index[window['high'] > level]
        else:
            hit = window.index[window['low'] < level]
        if len(hit) and (start is None or hit[0] < start):
            start = hit[0]

    target = frames.get(asset)
    if start is None or target is None or target.empty:
        return None
    after = target.index[target.index >= _match_index_tz(target, start)]
    return after[0] if len(after) else None


def check_psp(timestamp, eur_data, gbp_data, chf_data, smt_data, timeframe='M5'):
    """
    PSP на таймфрейме, который просил пользователь.
    timeframe: 'H4', 'H1', 'M15' или 'M5'. Цвет смотрится у свечи этого ТФ,
    внутри которой случился SMT, а не всегда у M5.
    """
    located = _bars_at(timestamp, eur_data, gbp_data, chf_data, timeframe)
    if located is None:
        return False

    greens = {}
    for asset, (df, pos) in located.items():
        candle = df.iloc[pos]
        greens[asset] = bool(candle['close'] >= candle['open'])
    return _diverges_like_psp(greens, smt_data)


def check_tpd(timestamp, eur_data, gbp_data, chf_data, timeframe='D1'):
    """
    TPD — фильтр раскорреляции, не вход и не сторона сделки. 3 свечи ТФ: левая, PSP, правая.
    Правая — свеча ТФ, внутри которой timestamp; от нее нужно только открытие,
    поэтому заглядывания в будущее нет. Для D1 это текущий день, PSP — вчера.
    Пары (CHF инвертирован) делятся на группы по цвету средней свечи и по направлению
    от закрытия левой до открытия правой. TPD — обе разбивки есть и совпадают:
    выбилась одна и та же пара. Открытие правой ровно на закрытии левой — TPD нет.
    timeframe: 'D1', 'H4', 'H1', 'M15'.
    """
    located = _bars_at(timestamp, eur_data, gbp_data, chf_data, timeframe)
    if located is None:
        return False

    greens, ups = {}, {}
    for asset, (df, pos) in located.items():
        if pos < 2:
            return False
        left, middle, right = df.iloc[pos - 2], df.iloc[pos - 1], df.iloc[pos]
        if abs(float(middle['close']) - float(middle['open'])) <= PRICE_EPS:
            return False
        move = float(right['open']) - float(left['close'])
        if abs(move) <= PRICE_EPS:
            return False
        greens[asset] = bool(middle['close'] > middle['open'])
        ups[asset] = move > 0

    color_split = _odd_pairs(greens)
    move_split = _odd_pairs(ups)
    return color_split is not None and color_split == move_split


def _odd_pairs(ups):
    """
    Пары, которые выбились из корреляции. CHF инвертирован.
    Все три согласованы — None. Иначе — меньшая группа.
    """
    aligned = {asset: (not up if asset == 'CHF' else up) for asset, up in ups.items()}
    up_group = frozenset(a for a, v in aligned.items() if v)
    down_group = frozenset(a for a, v in aligned.items() if not v)
    if not up_group or not down_group:
        return None
    return up_group if len(up_group) < len(down_group) else down_group


def _bars_at(timestamp, eur_data, gbp_data, chf_data, timeframe):
    """
    Свечи ТФ по трем парам и позиция свечи, внутри которой timestamp.
    Возвращает {'EUR': (df, pos), ...} или None.
    """
    freq = TF_TO_FREQ.get(timeframe, '5min')
    bar_length = pd.to_timedelta(freq)
    located = {}
    for asset, df in (('EUR', eur_data), ('GBP', gbp_data), ('CHF', chf_data)):
        if df is None or df.empty:
            return None
        if freq != '5min':
            df = resample_to_timeframe(df, freq)
        ts = _match_index_tz(df, timestamp)
        pos = df.index.searchsorted(ts, side='right') - 1
        if pos < 0 or ts - df.index[pos] >= bar_length:
            return None
        located[asset] = (df, pos)
    return located


def _diverges_like_psp(ups, smt_data):
    """
    ups: {'EUR': bool, 'GBP': bool, 'CHF': bool} — свеча или движение вверх.
    Медвежий SMT: снявшие EUR/GBP вниз, не снявшие вверх. Бычий наоборот. CHF инвертирован.
    """
    is_bearish_smt = smt_data['type'] == 'bearish'
    swept = smt_data['swept']
    for asset, went_up in ups.items():
        is_chf = asset == 'CHF'
        is_sweeper = asset in swept
        if is_bearish_smt:
            expected_up = is_chf if is_sweeper else not is_chf
        else:
            expected_up = not is_chf if is_sweeper else is_chf
        if went_up != expected_up:
            return False
    return True


def _prev_quarter_start(data, idx, level):
    """
    Индекс первой свечи квартала, который идет перед кварталом свечи idx.
    Предыдущий — по данным, как в find_smt_between_periods: для понедельника это пятница.
    """
    labels = data[level].to_numpy()
    origin = labels[idx]
    prev = None
    i = idx
    while i > 0:
        label = labels[i - 1]
        if label != origin:
            if prev is None:
                prev = label
            elif label != prev:
                break
        i -= 1
    return i


def get_nearest_swing(data, extreme_idx, swing_type='low', min_idx=1):
    """
    Ближайший свинг из 3 свечей ДО указанного индекса.
    Средняя свеча свинга не раньше min_idx.
    """
    limit = max(1, min_idx)

    for i in range(extreme_idx - 1, limit - 1, -1):
        if swing_type == 'low':
            left = data['low'].iloc[i - 1]
            mid = data['low'].iloc[i]
            right = data['low'].iloc[i + 1]
            if left > mid and mid < right:
                return i, mid
        elif swing_type == 'high':
            left = data['high'].iloc[i - 1]
            mid = data['high'].iloc[i]
            right = data['high'].iloc[i + 1]
            if left < mid and mid > right:
                return i, mid

    return None, None


def find_fvg(data, start_idx, end_idx, fvg_type='bearish'):
    """
    3-свечной FVG. Работает как отдельный поиск, если передать границы индексов.

    Бычий: high свечи 1 < low свечи 3. Зона [high_1, low_3].
    Медвежий: low свечи 1 > high свечи 3. Зона [high_3, low_1].
    confirmed_time — свеча 3, на которой разрыв уже виден.
    Сама функция не ждет тест зоны. Тест делает find_entry_levels / find_mss.
    """
    fvgs = []
    if data is None or data.empty or start_idx is None or end_idx is None:
        return fvgs

    start_idx = int(start_idx)
    end_idx = int(end_idx)
    if start_idx < 0 or end_idx < start_idx:
        return fvgs

    search_end = min(end_idx + 1, len(data) - 1)

    for j in range(start_idx + 2, search_end + 1):
        if fvg_type == 'bearish':
            gap_top = data['low'].iloc[j - 2]
            gap_bottom = data['high'].iloc[j]
            if gap_top > gap_bottom:
                fvgs.append({
                    'confirmed_idx': j,
                    'confirmed_time': data.index[j],
                    'top': gap_top,
                    'bottom': gap_bottom,
                })
        elif fvg_type == 'bullish':
            gap_bottom = data['high'].iloc[j - 2]
            gap_top = data['low'].iloc[j]
            if gap_top > gap_bottom:
                fvgs.append({
                    'confirmed_idx': j,
                    'confirmed_time': data.index[j],
                    'top': gap_top,
                    'bottom': gap_bottom,
                })
    return fvgs


def _wait_for_touch(data, start_time, price, side, deadline):
    """
    Первая свеча СТРОГО после start_time и не позже deadline,
    которая коснулась цены. Лонг: low <= price. Шорт: high >= price.
    """
    start_idx = get_safe_loc(data, start_time)
    if start_idx is None:
        return None

    deadline = _match_index_tz(data, deadline)
    for i in range(start_idx + 1, len(data)):
        ts = data.index[i]
        if ts > deadline:
            break
        candle = data.iloc[i]
        if side == 'bullish' and candle['low'] <= price:
            return ts
        if side == 'bearish' and candle['high'] >= price:
            return ts
    return None


def _fvg_on_break(data, breakout_idx, smt_type, timeframe_label, origin):
    """
    Дисбаланс самого слома, не любая дырка после SMT.
    Свеча пробоя входит в тройку FVG, вторая свеча тройки идет в сторону сделки.
    Подтверждение FVG остается в квартале SMT.
    """
    if breakout_idx < 1 or breakout_idx + 1 >= len(data):
        return None
    # Вторая свеча тройки — сама свеча пробоя. Подтверждение на следующей.
    confirmed = breakout_idx + 1
    if data[timeframe_label].iloc[confirmed] != origin:
        return None
    middle = data.iloc[breakout_idx]
    if smt_type == 'bullish' and middle['close'] <= middle['open']:
        return None
    if smt_type == 'bearish' and middle['close'] >= middle['open']:
        return None
    fvg_type = 'bearish' if smt_type == 'bearish' else 'bullish'
    for fvg in find_fvg(data, breakout_idx - 1, confirmed, fvg_type):
        if fvg['confirmed_idx'] == confirmed:
            return fvg
    return None


def find_mss(data, extreme_time, smt_type, max_bars_to_break=None, timeframe_label='true_date'):
    """
    Shift ищется только в квартале SMT. В следующем квартале слом уже не берем.
    Ломаемый свинг — в квартале SMT или в предыдущем квартале того же уровня.
    FVG — часть этого слома, вторая свеча бычья для лонга и медвежья для шорта.
    Лимитка на FVG живет до закрытия следующего квартала. Теста нет — None.
    """
    if data is None or data.empty or timeframe_label not in LEVEL_DURATION:
        return None
    if timeframe_label not in data.columns:
        return None

    extreme_time = _match_index_tz(data, extreme_time)
    if extreme_time not in data.index:
        return None

    extreme_idx = get_safe_loc(data, extreme_time)
    if extreme_idx is None or extreme_idx >= len(data) - 1:
        return None

    origin = _label_at(data, extreme_time, timeframe_label)
    allowed = {origin}

    target_swing = 'low' if smt_type == 'bearish' else 'high'
    swing_idx, swing_price = get_nearest_swing(
        data, extreme_idx, swing_type=target_swing,
        min_idx=_prev_quarter_start(data, extreme_idx, timeframe_label),
    )
    if swing_idx is None:
        return None

    breakout_idx = None
    search_limit = len(data) if max_bars_to_break is None else min(extreme_idx + max_bars_to_break, len(data))

    for i in range(extreme_idx + 1, search_limit):
        if data[timeframe_label].iloc[i] not in allowed:
            break
        if smt_type == 'bearish' and data['low'].iloc[i] < swing_price:
            breakout_idx = i
            break
        if smt_type == 'bullish' and data['high'].iloc[i] > swing_price:
            breakout_idx = i
            break

    if breakout_idx is None:
        return None

    fvg = _fvg_on_break(data, breakout_idx, smt_type, timeframe_label, origin)
    if fvg is None:
        return None

    trigger_idx = max(breakout_idx, fvg['confirmed_idx'])
    if data[timeframe_label].iloc[trigger_idx] != origin:
        return None
    mss_time = data.index[trigger_idx]
    entry_price = fvg['bottom'] if smt_type == 'bearish' else fvg['top']

    test_deadline = _limit_deadline(data, origin, timeframe_label)
    if test_deadline is None:
        return None
    test_time = _wait_for_touch(data, mss_time, entry_price, smt_type, test_deadline)
    if test_time is None:
        return None

    return {
        'mss_time': mss_time,
        'swing_price': swing_price,
        'fvg_top': fvg['top'],
        'fvg_bottom': fvg['bottom'],
        'top': fvg['top'],
        'bottom': fvg['bottom'],
        'mean_threshold': (fvg['top'] + fvg['bottom']) / 2,
        'entry': entry_price,
        'test_time': test_time,
    }


def _inverted_fvgs(data, start_time, side, timeframe_label):
    """
    FVG против сделки, которые после свипа пробило тело свечи.
    Шорт: бычий FVG, свеча закрылась ниже его низа. Лонг: медвежий FVG, закрытие выше верха.
    Исходный FVG подтвержден до свечи свипа, в квартале SMT или предыдущем (как свинг шифта).
    Пробой — после свипа и внутри квартала SMT (как шифт).
    Возвращает (контекст, список (fvg, индекс свечи пробоя)) или (None, []).
    """
    if data is None or data.empty or timeframe_label not in LEVEL_DURATION:
        return None, []
    if timeframe_label not in data.columns or side not in ('bullish', 'bearish'):
        return None, []
    start_idx = get_safe_loc(data, start_time)
    if start_idx is None:
        return None, []

    labels = data[timeframe_label].to_numpy()
    origin = labels[start_idx]
    quarter_end = start_idx
    while quarter_end + 1 < len(data) and labels[quarter_end + 1] == origin:
        quarter_end += 1
    window_start = _prev_quarter_start(data, start_idx, timeframe_label)

    source_type = 'bullish' if side == 'bearish' else 'bearish'
    closes = data['close'].to_numpy()
    inverted = []
    for fvg in find_fvg(data, window_start, start_idx - 1, source_type):
        if fvg['confirmed_idx'] >= start_idx:
            continue
        for i in range(fvg['confirmed_idx'] + 1, quarter_end + 1):
            pierced = closes[i] < fvg['bottom'] if side == 'bearish' else closes[i] > fvg['top']
            if pierced:
                if i >= start_idx:
                    inverted.append((fvg, i))
                break

    context = {'start_idx': start_idx, 'origin': origin, 'quarter_end': quarter_end}
    return context, inverted


def _zone_result(data, kind, zone, trigger_idx, side, origin, timeframe_label, extra):
    """Тест ближнего края зоны после trigger_idx до конца следующего квартала, как у FVG."""
    entry_price = float(zone['bottom'] if side == 'bearish' else zone['top'])
    deadline = _limit_deadline(data, origin, timeframe_label)
    if deadline is None:
        return None
    trigger_time = data.index[trigger_idx]
    test_time = _wait_for_touch(data, trigger_time, entry_price, side, deadline)
    if test_time is None:
        return None
    result = {
        'type': kind,
        'trigger_time': trigger_time,
        'top': zone['top'],
        'bottom': zone['bottom'],
        'fvg_top': zone['top'],
        'fvg_bottom': zone['bottom'],
        'mean_threshold': (zone['top'] + zone['bottom']) / 2,
        'entry': entry_price,
        'test_time': test_time,
    }
    result.update(extra)
    return result


def find_ifvg(data, start_time, side, timeframe_label='macro_q'):
    """
    iFVG после свипа в start_time. side — сторона сделки.
    Шорт: бычий FVG до свипа, после свипа пробит закрытием свечи ниже его низа —
    теперь это зона для шорта. Тест ищем только после свечи пробоя, вход и лимитка как у FVG.
    Берем самый ранний пробой, при равенстве — самый свежий FVG. Нет теста — следующий iFVG.
    Ключи: trigger_time (свеча пробоя), test_time, entry, top, bottom, source_time.
    """
    context, inverted = _inverted_fvgs(data, start_time, side, timeframe_label)
    if context is None:
        return None
    inverted.sort(key=lambda item: (item[1], -item[0]['confirmed_idx']))
    for fvg, pierce_idx in inverted:
        found = _zone_result(
            data, 'iFVG', fvg, pierce_idx, side, context['origin'], timeframe_label,
            {'source_time': fvg['confirmed_time']},
        )
        if found is not None:
            return found
    return None


def find_bpr(data, start_time, side, timeframe_label='macro_q'):
    """
    BPR после свипа в start_time. side — сторона сделки.
    Шорт: бычий FVG до свипа, после свипа пробит закрытием ниже его низа, и после свипа
    в квартале SMT образовался медвежий FVG, который накладывается на зону бычьего.
    Зона входа — из двух FVG та, что выгоднее: для шорта с более высоким низом,
    для лонга с более низким верхом. Тест — после пробоя первого и подтверждения второго,
    вход и лимитка как у FVG.
    Ключи: trigger_time, test_time, entry, top, bottom, source_top, source_bottom,
    second_top, second_bottom.
    """
    context, inverted = _inverted_fvgs(data, start_time, side, timeframe_label)
    if context is None or not inverted:
        return None

    start_idx, quarter_end = context['start_idx'], context['quarter_end']
    second_fvgs = [
        fvg for fvg in find_fvg(data, start_idx, quarter_end, side)
        if fvg['confirmed_idx'] <= quarter_end
    ]

    candidates = []
    for source, pierce_idx in inverted:
        for second in second_fvgs:
            if second['bottom'] >= source['top'] or second['top'] <= source['bottom']:
                continue
            candidates.append((max(pierce_idx, second['confirmed_idx']), source, second))

    candidates.sort(key=lambda item: (item[0], -item[2]['confirmed_idx']))
    for trigger_idx, source, second in candidates:
        if side == 'bearish':
            zone = source if source['bottom'] > second['bottom'] else second
        else:
            zone = source if source['top'] < second['top'] else second
        found = _zone_result(
            data, 'BPR', zone, trigger_idx, side, context['origin'], timeframe_label,
            {
                'source_time': source['confirmed_time'],
                'source_top': source['top'],
                'source_bottom': source['bottom'],
                'second_top': second['top'],
                'second_bottom': second['bottom'],
            },
        )
        if found is not None:
            return found
    return None


def get_quarter_extremums_for_sl(data, current_time, timeframe_label='macro_q', smt_type='bearish', quarter_id=None):
    """
    Хай и лой конкретного квартала. Это не свинг.
    Берутся только свечи этого квартала с временем <= current_time.
    """
    if data is None or data.empty or timeframe_label not in LEVEL_DURATION:
        return None, None
    if timeframe_label not in data.columns:
        return None, None

    current_time = _match_index_tz(data, current_time)
    if quarter_id is None:
        if current_time not in data.index:
            return None, None
        quarter_id = _label_at(data, current_time, timeframe_label)

    if quarter_id is None:
        return None, None

    quarter_data = data[data[timeframe_label] == quarter_id]
    quarter_data = quarter_data[quarter_data.index <= current_time]
    if quarter_data.empty:
        return None, None

    return quarter_data['high'].max(), quarter_data['low'].min()


def _stop_for_side(q_high, q_low, side, entry_price):
    if q_high is None or q_low is None or entry_price is None:
        return None
    stop = q_low if side == 'bullish' else q_high
    if side == 'bullish' and stop >= entry_price:
        return None
    if side == 'bearish' and stop <= entry_price:
        return None
    return float(stop)


def _ob_candles(c1, c2, side):
    """Лонг: c1 медвежья, c2 бычья, тело c2 перекрывает тело c1. Шорт зеркально."""
    if side == 'bullish':
        colors_ok = c1['close'] < c1['open'] and c2['close'] > c2['open']
        engulfing = c2['close'] >= c1['open'] - PRICE_EPS and c2['open'] <= c1['close'] + PRICE_EPS
    elif side == 'bearish':
        colors_ok = c1['close'] > c1['open'] and c2['close'] < c2['open']
        engulfing = c2['close'] <= c1['open'] + PRICE_EPS and c2['open'] >= c1['close'] - PRICE_EPS
    else:
        return False
    return bool(colors_ok and engulfing)


def _body_mid(candle):
    return (float(candle['open']) + float(candle['close'])) / 2


def _ob_shape(data, idx, side, timeframe_label):
    """
    Свечи OB без теста: c1 = idx, c2 = idx + 1.
    Лонг: c1 медвежья, c2 бычья, тело c2 перекрывает тело c1, c1 или c2 — лой квартала c2.
    Шорт зеркально. Возвращает (экстремум OB, уровень c1: хай для лонга, лой для шорта) или None.
    """
    if idx is None or idx < 0 or idx + 1 >= len(data):
        return None
    c1 = data.iloc[idx]
    c2 = data.iloc[idx + 1]
    if not _ob_candles(c1, c2, side):
        return None

    confirm_time = data.index[idx + 1]
    quarter_id = _label_at(data, confirm_time, timeframe_label)
    q_high, q_low = get_quarter_extremums_for_sl(
        data, confirm_time, timeframe_label, quarter_id=quarter_id
    )
    if q_high is None:
        return None

    if side == 'bullish':
        extreme = float(min(c1['low'], c2['low']))
        if extreme > q_low + PRICE_EPS:
            return None
        return extreme, float(c1['high'])
    extreme = float(max(c1['high'], c2['high']))
    if extreme < q_high - PRICE_EPS:
        return None
    return extreme, float(c1['low'])


def _ob_style_entry(data, c3_idx, level, side):
    """
    Вход по правилам OB от уровня level, третья свеча — c3_idx.
    c3 открылась ниже уровня (лонг) или выше (шорт) — вход сразу по ее открытию.
    Иначе лимитка на уровне, тест с c3 не дольше OB_TEST_BARS свечей.
    Возвращает (цена входа, время входа) или (None, None).
    """
    if c3_idx >= len(data):
        return None, None
    c3_open = float(data['open'].iloc[c3_idx])
    if (side == 'bullish' and c3_open < level) or (side == 'bearish' and c3_open > level):
        return c3_open, data.index[c3_idx]
    for i in range(c3_idx, min(c3_idx + OB_TEST_BARS, len(data))):
        candle = data.iloc[i]
        touched = candle['low'] <= level if side == 'bullish' else candle['high'] >= level
        if touched:
            return float(level), data.index[i]
    return None, None


def get_order_block(data, extreme_time, side, timeframe_label='macro_q'):
    """
    Валидный OB после SMT:
    - лонг: первая свеча медвежья, вторая бычья; шорт наоборот;
    - тело второй свечи полностью перекрывает тело первой;
    - первая или вторая свеча — лой (лонг) или хай (шорт) квартала второй свечи;
    - стоп за этим экстремумом;
    - вход: хай первой (лонг) или лой первой (шорт). Если третья свеча открылась
      ниже хая первой (лонг) или выше лоя первой (шорт) — вход сразу по ее открытию.
      Иначе лимитка, тест с третьей свечи не дольше OB_TEST_BARS свечей.
    """
    if data is None or data.empty or timeframe_label not in LEVEL_DURATION:
        return None
    if side not in ('bullish', 'bearish'):
        return None
    if extreme_time not in data.index and _match_index_tz(data, extreme_time) not in data.index:
        return None

    idx = get_safe_loc(data, extreme_time)
    if idx is None or idx + 2 >= len(data):
        return None

    shape = _ob_shape(data, idx, side, timeframe_label)
    if shape is None:
        return None
    extreme, entry_price = shape

    entry_price, test_time = _ob_style_entry(data, idx + 2, entry_price, side)
    if test_time is None:
        return None

    stop_loss = _stop_for_side(extreme, extreme, side, entry_price)
    if stop_loss is None:
        return None

    return {
        'entry': entry_price,
        'sl': stop_loss,
        'entry_time': test_time,
        'type': 'OB_Bullish' if side == 'bullish' else 'OB_Bearish',
        'ob_time': data.index[idx],
        'mean_threshold': _body_mid(data.iloc[idx]),
    }


def find_order_block_after(data, start_time, side, timeframe_label='macro_q'):
    """
    Первый валидный OB после свипа в start_time.
    Вторая свеча OB — в квартале SMT. Первая может быть свечой перед свипом,
    тогда вторая — сама свеча свипа. Тест не дольше OB_TEST_BARS свечей.
    """
    if data is None or data.empty or timeframe_label not in LEVEL_DURATION:
        return None
    if timeframe_label not in data.columns:
        return None

    start_idx = get_safe_loc(data, start_time)
    if start_idx is None:
        return None
    labels = data[timeframe_label].to_numpy()
    origin = labels[start_idx]
    for i in range(max(0, start_idx - 1), len(data) - 1):
        if labels[i + 1] != origin:
            break
        ob = get_order_block(data, data.index[i], side, timeframe_label)
        if ob is not None:
            return ob
    return None


def find_breaker_block(data, start_time, side, timeframe_label='macro_q'):
    """
    Breaker Block после свипа в start_time. side — сторона сделки.
    Шорт: бычий OB (те же свечи, что у get_order_block) в квартале SMT или предыдущем,
    после свипа и внутри квартала SMT пробит закрытием тела ниже лоя OB.
    Бывший стоп OB (лой) — вход в шорт, бывший вход (хай c1) — стоп. Лонг зеркально.
    Пробой до свипа не считается. Тест после свечи пробоя до конца следующего квартала.
    Берем самый ранний пробой, при равенстве — самый свежий OB. Нет теста — следующий BB.
    Ключи: entry, sl, entry_time, type, ob_time, break_time.
    """
    if data is None or data.empty or timeframe_label not in LEVEL_DURATION:
        return None
    if timeframe_label not in data.columns or side not in ('bullish', 'bearish'):
        return None
    start_idx = get_safe_loc(data, start_time)
    if start_idx is None:
        return None

    labels = data[timeframe_label].to_numpy()
    origin = labels[start_idx]
    quarter_end = start_idx
    while quarter_end + 1 < len(data) and labels[quarter_end + 1] == origin:
        quarter_end += 1
    window_start = _prev_quarter_start(data, start_idx, timeframe_label)

    ob_side = 'bullish' if side == 'bearish' else 'bearish'
    closes = data['close'].to_numpy()
    candidates = []
    for i in range(window_start, quarter_end):
        shape = _ob_shape(data, i, ob_side, timeframe_label)
        if shape is None:
            continue
        extreme, c1_level = shape
        for j in range(i + 2, quarter_end + 1):
            broken = closes[j] < extreme if side == 'bearish' else closes[j] > extreme
            if broken:
                if j >= start_idx:
                    candidates.append((j, i, extreme, c1_level))
                break

    deadline = _limit_deadline(data, origin, timeframe_label)
    if deadline is None:
        return None
    candidates.sort(key=lambda item: (item[0], -item[1]))
    for break_idx, ob_idx, extreme, c1_level in candidates:
        test_time = _wait_for_touch(data, data.index[break_idx], extreme, side, deadline)
        if test_time is None:
            continue
        stop_loss = _stop_for_side(c1_level, c1_level, side, extreme)
        if stop_loss is None:
            continue
        return {
            'entry': extreme,
            'sl': stop_loss,
            'entry_time': test_time,
            'type': 'BB_Bearish' if side == 'bearish' else 'BB_Bullish',
            'ob_time': data.index[ob_idx],
            'break_time': data.index[break_idx],
            'mean_threshold': _body_mid(data.iloc[ob_idx]),
        }
    return None


def _prev_quarter_range(data, idx, labels):
    """Хай и лой квартала, который идет перед кварталом свечи idx (по данным). Нет — (None, None)."""
    q_start = idx
    while q_start > 0 and labels[q_start - 1] == labels[idx]:
        q_start -= 1
    if q_start == 0:
        return None, None
    p_end = q_start - 1
    p_start = p_end
    while p_start > 0 and labels[p_start - 1] == labels[p_end]:
        p_start -= 1
    prev = data.iloc[p_start:p_end + 1]
    return float(prev['high'].max()), float(prev['low'].min())


def _rb_shape(data, idx, side, labels):
    """
    Свечи RB: c1 = idx, c2 = idx + 1, c3 = idx + 2 подтверждает свинг.
    Шорт: c1 бычья, c2 медвежья, close c1 = open c2 — это уровень блока.
    Верхняя тень хотя бы одной из двух — не меньше RB_MIN_WICK_SHARE свечи.
    Хай блока — свинг-хай: выше свечи до c1 и выше c3.
    Хай блока снял хай предыдущего квартала минимум на MIN_SWEEP, тела остались ниже него.
    Лонг зеркально. Возвращает (уровень, экстремум блока) или None.
    """
    if idx < 1 or idx + 2 >= len(data):
        return None
    before, c1, c2, c3 = data.iloc[idx - 1], data.iloc[idx], data.iloc[idx + 1], data.iloc[idx + 2]
    if abs(float(c1['close']) - float(c2['open'])) > RB_LEVEL_TOLERANCE:
        return None
    level = float(c1['close'])

    def wick_share(candle, upper):
        size = float(candle['high']) - float(candle['low'])
        if size <= PRICE_EPS:
            return 0.0
        if upper:
            return (float(candle['high']) - max(candle['open'], candle['close'])) / size
        return (min(candle['open'], candle['close']) - float(candle['low'])) / size

    prev_high, prev_low = _prev_quarter_range(data, idx + 1, labels)
    if prev_high is None:
        return None

    if side == 'bearish':
        if not (c1['close'] > c1['open'] and c2['close'] < c2['open']):
            return None
        if max(wick_share(c1, True), wick_share(c2, True)) < RB_MIN_WICK_SHARE:
            return None
        extreme = float(max(c1['high'], c2['high']))
        if not (extreme > before['high'] and extreme > c3['high']):
            return None
        if extreme < prev_high + MIN_SWEEP - PRICE_EPS or level >= prev_high:
            return None
        return level, extreme

    if side == 'bullish':
        if not (c1['close'] < c1['open'] and c2['close'] > c2['open']):
            return None
        if max(wick_share(c1, False), wick_share(c2, False)) < RB_MIN_WICK_SHARE:
            return None
        extreme = float(min(c1['low'], c2['low']))
        if not (extreme < before['low'] and extreme < c3['low']):
            return None
        if extreme > prev_low - MIN_SWEEP + PRICE_EPS or level <= prev_low:
            return None
        return level, extreme
    return None


def find_rejection_block(data, start_time, side, timeframe_label='macro_q'):
    """
    Rejection Block после свипа в start_time. side — сторона сделки.
    Свечи блока — по правилам _rb_shape. Вторая свеча блока — в квартале SMT,
    первая может быть свечой перед свипом.
    Вход на уровне close/open по правилам OB: сразу по открытию c3 или лимиткой 5 свечей.
    Стоп за самым высоким хаем (шорт) или самым низким лоем (лонг) двух свечей.
    Берем первый RB с входом. Ключи: entry, sl, entry_time, type, rb_time.
    """
    if data is None or data.empty or timeframe_label not in LEVEL_DURATION:
        return None
    if timeframe_label not in data.columns or side not in ('bullish', 'bearish'):
        return None
    start_idx = get_safe_loc(data, start_time)
    if start_idx is None:
        return None

    labels = data[timeframe_label].to_numpy()
    origin = labels[start_idx]
    for i in range(max(1, start_idx - 1), len(data) - 2):
        if labels[i + 1] != origin:
            break
        shape = _rb_shape(data, i, side, labels)
        if shape is None:
            continue
        level, extreme = shape
        entry_price, entry_time = _ob_style_entry(data, i + 2, level, side)
        if entry_time is None:
            continue
        stop_loss = _stop_for_side(extreme, extreme, side, entry_price)
        if stop_loss is None:
            continue
        return {
            'entry': entry_price,
            'sl': stop_loss,
            'entry_time': entry_time,
            'type': 'RB_Bearish' if side == 'bearish' else 'RB_Bullish',
            'rb_time': data.index[i],
            'level': level,
            'mean_threshold': (level + extreme) / 2,
        }
    return None


def find_propulsion_block(data, zone, side, timeframe_label='macro_q'):
    """
    Propulsion Block после касания исходной зоны. side — сторона сделки, та же, что у зоны.
    zone — результат find_order_block_after, find_breaker_block, find_rejection_block,
    find_mss, find_ifvg или find_bpr. Касание — test_time (FVG-зоны) или entry_time (блоки).
    PB — новый OB по свечам (цвета и поглощение), без экстремума квартала и свипа:
    первая свеча не раньше касания, вторая — до конца квартала касания.
    Уровень 50% зоны (mean_threshold): если до PB свеча закрылась ниже него (лонг)
    или выше (шорт) — зона сломана, PB нет.
    Вход по правилам OB от хая (лонг) или лоя (шорт) первой свечи PB, стоп за экстремумом PB.
    invalidation — тот же уровень 50%, его передают в simulate_trade_with_rm.
    Ключи: entry, sl, entry_time, type, pb_time, invalidation.
    """
    if data is None or data.empty or timeframe_label not in LEVEL_DURATION or not zone:
        return None
    if timeframe_label not in data.columns or side not in ('bullish', 'bearish'):
        return None
    mean_threshold = zone.get('mean_threshold')
    touch_time = zone.get('test_time', zone.get('entry_time'))
    if mean_threshold is None or touch_time is None:
        return None
    touch_idx = get_safe_loc(data, touch_time)
    if touch_idx is None:
        return None

    labels = data[timeframe_label].to_numpy()
    origin = labels[touch_idx]
    quarter_end = touch_idx
    while quarter_end + 1 < len(data) and labels[quarter_end + 1] == origin:
        quarter_end += 1

    closes = data['close'].to_numpy()

    def closed_beyond(i):
        return closes[i] < mean_threshold if side == 'bullish' else closes[i] > mean_threshold

    if closed_beyond(touch_idx):
        return None
    for i in range(touch_idx, quarter_end):
        if closed_beyond(i + 1):
            return None
        c1, c2 = data.iloc[i], data.iloc[i + 1]
        if not _ob_candles(c1, c2, side):
            continue
        if side == 'bullish':
            extreme, level = float(min(c1['low'], c2['low'])), float(c1['high'])
        else:
            extreme, level = float(max(c1['high'], c2['high'])), float(c1['low'])
        entry_price, entry_time = _ob_style_entry(data, i + 2, level, side)
        if entry_time is None:
            continue
        if (side == 'bullish' and entry_price <= mean_threshold) or (side == 'bearish' and entry_price >= mean_threshold):
            continue
        stop_loss = _stop_for_side(extreme, extreme, side, entry_price)
        if stop_loss is None:
            continue
        return {
            'entry': entry_price,
            'sl': stop_loss,
            'entry_time': entry_time,
            'type': 'PB_Bullish' if side == 'bullish' else 'PB_Bearish',
            'pb_time': data.index[i],
            'invalidation': float(mean_threshold),
        }
    return None


def check_dfr(data, time, side, entry_price, timeframe='M15'):
    """
    DFR — фильтр направления, не вход. Диапазон — хай и лой второй и третьей трети первого квартала.
    H1: понедельник 02:00–18:00 NY, действует до закрытия рынка в пятницу.
    M15: 20:00–00:00 NY, действует до 18:00 этого true day.
    Цена входа выше середины диапазона (премия) — только шорт, ниже (дискаунт) — только лонг.
    Ровно на середине — ни то, ни другое.
    Диапазон еще не сформирован, его нет в данных или другой ТФ — True, без фильтра.
    data — свечи пары сделки с метками времени (week, true_date).
    """
    if side not in ('bullish', 'bearish'):
        return False
    if data is None or data.empty or timeframe not in ('H1', 'M15'):
        return True
    ts = _match_index_tz(data, time)
    pos = data.index.searchsorted(ts, side='right') - 1
    if pos < 0:
        return True
    row = data.iloc[pos]
    hours = data.index.hour

    if timeframe == 'H1':
        week = row['week']
        formed_at = pd.Timestamp(f"{week} 18:00")
        in_range = (data['week'] == week) & (data.index.weekday == 0) & (hours >= 2) & (hours < 18)
    else:
        day = row['true_date']
        formed_at = pd.Timestamp(day)
        in_range = (data['true_date'] == day) & (hours >= 20)
    if data.index.tz is not None:
        formed_at = formed_at.tz_localize(data.index.tz)
    if ts < formed_at:
        return True

    window = data[in_range]
    if window.empty:
        return True
    middle = (float(window['high'].max()) + float(window['low'].min())) / 2
    if side == 'bullish':
        return entry_price < middle - PRICE_EPS
    return entry_price > middle + PRICE_EPS


def find_entry_levels(data, trigger_time, entry_type, smt_type, fvg_data=None, timeframe_label='macro_q'):
    """
    FVG: вход от теста ближнего края зоны до закрытия квартала после сигнала. Иначе None.
    OB: делегирует в get_order_block (тест и стоп за экстремумом OB уже там).
    entry_time — свеча теста, не свеча сигнала.
    """
    if data is None or data.empty or timeframe_label not in LEVEL_DURATION:
        return None
    if get_safe_loc(data, trigger_time) is None:
        return None

    trigger_time = _match_index_tz(data, trigger_time)

    if entry_type == 'order_block':
        return get_order_block(data, trigger_time, smt_type, timeframe_label)

    if entry_type != 'fvg' or not fvg_data:
        return None

    top = fvg_data.get('top', fvg_data.get('fvg_top'))
    bottom = fvg_data.get('bottom', fvg_data.get('fvg_bottom'))
    if top is None or bottom is None:
        return None

    entry_price = float(bottom if smt_type == 'bearish' else top)
    test_time = fvg_data.get('test_time')
    if test_time is not None:
        test_time = _match_index_tz(data, test_time)
        if test_time not in data.index:
            test_time = None

    origin = _label_at(data, trigger_time, timeframe_label)
    deadline = _limit_deadline(data, origin, timeframe_label)
    if test_time is not None and (deadline is None or test_time > deadline):
        test_time = None
    if test_time is None and deadline is not None:
        test_time = _wait_for_touch(data, trigger_time, entry_price, smt_type, deadline)
    if test_time is None:
        return None

    quarter_id = _label_at(data, trigger_time, timeframe_label)
    q_high, q_low = get_quarter_extremums_for_sl(
        data, test_time, timeframe_label, quarter_id=quarter_id
    )
    stop_loss = _stop_for_side(q_high, q_low, smt_type, entry_price)
    if stop_loss is None:
        return None

    return {'entry': entry_price, 'sl': stop_loss, 'entry_time': test_time}


def simulate_trade_with_rm(data, entry_time, entry_price, sl_price, side, deposit=5000, risk_pct=0.005, tp_ratio=2.0, cancel_after_bars=60, invalidation=None):
    """
    Сделка уже открыта по entry_price на entry_time (свеча теста).
    На свече входа тоже проверяем SL, затем TP: лимитка могла задеть стоп в той же свече.
    invalidation — уровень принудительного закрытия (50% зоны у PB). Свеча закрылась
    ниже него (лонг) или выше (шорт) — выходим по ее закрытию, result 'Closed'.
    Стоп и тейк внутри свечи проверяются раньше закрытия.
    """
    if data is None or data.empty:
        return None
    if side not in ('bullish', 'bearish'):
        return None

    start_idx = get_safe_loc(data, entry_time)
    if start_idx is None:
        return None

    risk_amount = deposit * risk_pct
    risk_distance = abs(entry_price - sl_price)
    if risk_distance == 0:
        return None

    if side == 'bullish':
        tp_price = entry_price + (risk_distance * tp_ratio)
    else:
        tp_price = entry_price - (risk_distance * tp_ratio)

    outcome = {
        'Time': str(data.index[start_idx]),
        'Side': side.upper(),
        'Entry_Price': round(entry_price, 5),
        'Stop_Loss': round(sl_price, 5),
        'Take_Profit': round(tp_price, 5),
        'result': 'Cancelled',
        'profit_usd': 0,
        'balance_change': 0
    }

    last_i = min(start_idx + cancel_after_bars, len(data) - 1)
    for i in range(start_idx, last_i + 1):
        candle = data.iloc[i]
        hit_sl = (side == 'bullish' and candle['low'] <= sl_price) or \
                 (side == 'bearish' and candle['high'] >= sl_price)
        hit_tp = (side == 'bullish' and candle['high'] >= tp_price) or \
                 (side == 'bearish' and candle['low'] <= tp_price)

        if hit_sl:
            outcome['result'] = 'Loss'
            outcome['profit_usd'] = -risk_amount
            outcome['balance_change'] = -risk_amount
            return outcome
        if hit_tp:
            outcome['result'] = 'Win'
            outcome['profit_usd'] = risk_amount * tp_ratio
            outcome['balance_change'] = risk_amount * tp_ratio
            return outcome
        if invalidation is not None:
            close = float(candle['close'])
            closed_beyond = (side == 'bullish' and close < invalidation) or \
                            (side == 'bearish' and close > invalidation)
            if closed_beyond:
                move = close - entry_price if side == 'bullish' else entry_price - close
                pnl = risk_amount * move / risk_distance
                outcome['result'] = 'Closed'
                outcome['Exit_Price'] = round(close, 5)
                outcome['profit_usd'] = pnl
                outcome['balance_change'] = pnl
                return outcome

    return outcome
