import json
import math
import os
from datetime import datetime, timezone

import numpy as np
import yfinance as yf
from scipy.stats import norm


# ============================================================
# MACH1 GAMMA
# MES / ES INTRADAY GEX ENGINE
#
# Source:
#   SPX options via Yahoo Finance
#
# Output:
#   data/gex_es.json
#
# Main levels:
#   CALL WALL
#   GAMMA FLIP
#   PUT WALL
# ============================================================

OUTPUT_FILE = "data/gex_es.json"

RISK_FREE_RATE = 0.04

# ------------------------------------------------------------
# FILTERS
# ------------------------------------------------------------

MAX_DTE = 45
MIN_OPEN_INTEREST = 20
MAX_DISTANCE_FROM_SPOT = 0.15

# Gamma flip search range
FLIP_RANGE = 0.06
FLIP_STEPS = 401

# Profile smoothing
SMOOTHING_WINDOW = 5

# Wall detection
WALL_PERCENTILE = 80
MAX_WALL_DISTANCE = 0.12
WALL_CLUSTER_RADIUS = 2


# ============================================================
# SAFE NUMBER
# ============================================================

def num(value, default=0.0):

    try:

        if value is None:
            return default

        value = float(value)

        if not math.isfinite(value):
            return default

        return value

    except Exception:

        return default


# ============================================================
# SPX SPOT
# ============================================================

def get_spx_spot():

    ticker = yf.Ticker("^GSPC")

    hist = ticker.history(
        period="2d",
        interval="1m",
        prepost=True
    )

    if hist.empty:

        hist = ticker.history(
            period="5d",
            interval="1d"
        )

    if hist.empty:

        raise RuntimeError(
            "Unable to obtain SPX price."
        )

    return float(
        hist["Close"].dropna().iloc[-1]
    )


# ============================================================
# ES FUTURES PRICE
# ============================================================

def get_es_price():

    ticker = yf.Ticker("ES=F")

    hist = ticker.history(
        period="2d",
        interval="5m",
        prepost=True
    )

    if hist.empty:

        hist = ticker.history(
            period="5d",
            interval="1d"
        )

    if hist.empty:

        return None

    return float(
        hist["Close"].dropna().iloc[-1]
    )


# ============================================================
# GET SPX OPTIONS
# ============================================================

def get_options():

    ticker = yf.Ticker("^SPX")

    expirations = ticker.options

    if not expirations:

        raise RuntimeError(
            "No SPX option expirations returned."
        )

    now = datetime.now(timezone.utc)

    chains = []

    for expiration in expirations:

        try:

            exp_dt = datetime.strptime(
                expiration,
                "%Y-%m-%d"
            ).replace(
                hour=21,
                tzinfo=timezone.utc
            )

            dte = (
                exp_dt - now
            ).total_seconds() / 86400.0

            if dte < -0.5:
                continue

            if dte > MAX_DTE:
                continue

            chain = ticker.option_chain(
                expiration
            )

            calls = chain.calls.copy()
            puts = chain.puts.copy()

            calls["type"] = "call"
            puts["type"] = "put"

            calls["expiration"] = expiration
            puts["expiration"] = expiration

            calls["dte"] = max(
                dte,
                0.05
            )

            puts["dte"] = max(
                dte,
                0.05
            )

            chains.append(calls)
            chains.append(puts)

        except Exception as e:

            print(
                f"Skipping expiration {expiration}: {e}"
            )

    if not chains:

        raise RuntimeError(
            "Unable to retrieve SPX option chains."
        )

    return chains


# ============================================================
# PREPARE OPTIONS
# ============================================================

def prepare_options(
    spot,
    chains
):

    strikes = []
    oi_values = []
    iv_values = []
    dte_values = []
    signs = []

    for chain in chains:

        for _, row in chain.iterrows():

            strike = num(
                row.get("strike")
            )

            oi = num(
                row.get("openInterest")
            )

            iv = num(
                row.get("impliedVolatility")
            )

            dte = num(
                row.get("dte")
            )

            option_type = row.get(
                "type"
            )

            if strike <= 0:
                continue

            if oi < MIN_OPEN_INTEREST:
                continue

            if iv <= 0.0001:
                continue

            if abs(
                strike / spot - 1.0
            ) > MAX_DISTANCE_FROM_SPOT:

                continue

            if option_type == "call":

                sign = 1.0

            elif option_type == "put":

                sign = -1.0

            else:

                continue

            strikes.append(strike)
            oi_values.append(oi)
            iv_values.append(iv)
            dte_values.append(
                max(dte, 0.05)
            )
            signs.append(sign)

    if not strikes:

        raise RuntimeError(
            "No usable option contracts after filtering."
        )

    return (
        np.array(strikes, dtype=float),
        np.array(oi_values, dtype=float),
        np.array(iv_values, dtype=float),
        np.array(dte_values, dtype=float),
        np.array(signs, dtype=float)
    )


# ============================================================
# CALCULATE GEX AT PRICE
# ============================================================

def calculate_gex_at_price(
    price,
    strikes,
    oi,
    iv,
    dte,
    signs
):

    if price <= 0:
        return np.nan

    T = np.maximum(
        dte / 365.0,
        1.0 / 365.0
    )

    sqrt_T = np.sqrt(T)

    d1 = (
        np.log(price / strikes)
        + (
            RISK_FREE_RATE
            + 0.5 * iv * iv
        ) * T
    ) / (
        iv * sqrt_T
    )

    gamma = (
        np.exp(
            -0.5 * d1 * d1
        )
        / np.sqrt(2.0 * np.pi)
    ) / (
        price * iv * sqrt_T
    )

    # Moderate weighting toward shorter expirations.
    dte_weight = 1.0 / np.sqrt(
        np.maximum(dte, 1.0)
    )

    dte_weight = np.clip(
        dte_weight,
        0.10,
        1.0
    )

    gex = (
        gamma
        * oi
        * 100.0
        * price
        * price
        * 0.01
        * dte_weight
        * signs
    )

    return float(
        np.nansum(gex)
    )


# ============================================================
# BUILD GEX PROFILE
# ============================================================

def build_gex_profile(
    spot,
    strikes,
    oi,
    iv,
    dte,
    signs
):

    profile = {}

    unique_strikes = np.unique(
        strikes
    )

    for strike in unique_strikes:

        mask = strikes == strike

        local_strikes = strikes[mask]
        local_oi = oi[mask]
        local_iv = iv[mask]
        local_dte = dte[mask]
        local_signs = signs[mask]

        T = np.maximum(
            local_dte / 365.0,
            1.0 / 365.0
        )

        d1 = (
            np.log(
                spot / local_strikes
            )
            + (
                RISK_FREE_RATE
                + 0.5 * local_iv * local_iv
            ) * T
        ) / (
            local_iv * np.sqrt(T)
        )

        gamma = (
            np.exp(
                -0.5 * d1 * d1
            )
            / np.sqrt(2.0 * np.pi)
        ) / (
            spot
            * local_iv
            * np.sqrt(T)
        )

        dte_weight = 1.0 / np.sqrt(
            np.maximum(
                local_dte,
                1.0
            )
        )

        dte_weight = np.clip(
            dte_weight,
            0.10,
            1.0
        )

        gex = (
            gamma
            * local_oi
            * 100.0
            * spot
            * spot
            * 0.01
            * dte_weight
            * local_signs
        )

        profile[
            float(strike)
        ] = float(
            np.nansum(gex)
        )

    return profile


# ============================================================
# SMOOTH PROFILE
# ============================================================

def smooth_profile(profile):

    if len(profile) < 3:
        return profile

    strikes = np.array(
        sorted(profile.keys()),
        dtype=float
    )

    values = np.array(
        [
            profile[s]
            for s in strikes
        ],
        dtype=float
    )

    kernel_size = SMOOTHING_WINDOW

    if kernel_size <= 1:
        return profile

    if kernel_size > len(values):
        kernel_size = len(values)

    kernel = (
        np.ones(kernel_size)
        / kernel_size
    )

    smoothed = np.convolve(
        values,
        kernel,
        mode="same"
    )

    half = kernel_size // 2

    if half > 0:

        smoothed[:half] = values[:half]
        smoothed[-half:] = values[-half:]

    return {
        float(strikes[i]):
        float(smoothed[i])
        for i in range(len(strikes))
    }


# ============================================================
# FIND GAMMA FLIP
#
# Improved method:
# 1. Search for actual zero crossings.
# 2. Rank crossings by strength and distance.
# 3. If no clean crossing exists, return the
#    strongest near-zero point instead of immediately
#    returning null.
# ============================================================

def find_gamma_flip(
    spot,
    strikes,
    oi,
    iv,
    dte,
    signs
):

    low = spot * (1.0 - FLIP_RANGE)
    high = spot * (1.0 + FLIP_RANGE)

    prices = np.linspace(
        low,
        high,
        FLIP_STEPS
    )

    values = []

    for price in prices:

        try:

            value = calculate_gex_at_price(
                float(price),
                strikes,
                oi,
                iv,
                dte,
                signs
            )

            values.append(value)

        except Exception:

            values.append(np.nan)

    values = np.array(
        values,
        dtype=float
    )

    valid = np.isfinite(values)

    if not np.any(valid):
        return None

    finite_values = np.abs(
        values[valid]
    )

    if len(finite_values) == 0:
        return None

    strength_threshold = np.percentile(
        finite_values,
        25
    )

    candidates = []

    for i in range(
        1,
        len(prices)
    ):

        a = values[i - 1]
        b = values[i]

        if not np.isfinite(a):
            continue

        if not np.isfinite(b):
            continue

        # Actual sign change
        if a * b < 0:

            strength = (
                abs(a) + abs(b)
            ) / 2.0

            if strength < strength_threshold:
                continue

            p1 = prices[i - 1]
            p2 = prices[i]

            if b != a:

                flip = (
                    p1
                    + (-a)
                    * (p2 - p1)
                    / (b - a)
                )

                candidates.append(
                    (
                        float(flip),
                        float(strength)
                    )
                )

    # --------------------------------------------------------
    # If a genuine crossing exists, choose the strongest
    # crossing with preference for proximity to spot.
    # --------------------------------------------------------

    if candidates:

        candidates.sort(
            key=lambda x: (
                abs(x[0] - spot),
                -x[1]
            )
        )

        return candidates[0][0]

    # --------------------------------------------------------
    # FALLBACK
    #
    # Sometimes the calculated GEX curve does not cross zero
    # inside the selected range. Instead of producing null,
    # identify the price where absolute GEX is smallest.
    #
    # This gives us a "zero-GEX zone" approximation.
    # --------------------------------------------------------

    abs_values = np.abs(values)

    abs_values[
        ~np.isfinite(abs_values)
    ] = np.inf

    index = int(
        np.argmin(abs_values)
    )

    if not np.isfinite(
        abs_values[index]
    ):

        return None

    fallback_flip = float(
        prices[index]
    )

    # Only accept fallback if it is reasonably close
    # to current SPX.

    if abs(
        fallback_flip / spot - 1.0
    ) <= FLIP_RANGE:

        return fallback_flip

    return None


# ============================================================
# WALL CLUSTER SCORE
# ============================================================

def cluster_score(
    values,
    index
):

    start = max(
        0,
        index - WALL_CLUSTER_RADIUS
    )

    end = min(
        len(values),
        index
        + WALL_CLUSTER_RADIUS
        + 1
    )

    cluster = np.abs(
        values[start:end]
    )

    weights = np.ones(
        len(cluster)
    )

    center = index - start

    if 0 <= center < len(weights):

        weights[center] = 2.0

    return float(
        np.sum(
            cluster * weights
        )
    )


# ============================================================
# FIND WALLS
# ============================================================

def find_walls(
    profile,
    spot
):

    if not profile:

        return None, None

    strikes = np.array(
        sorted(profile.keys()),
        dtype=float
    )

    values = np.array(
        [
            profile[s]
            for s in strikes
        ],
        dtype=float
    )

    call_wall = None
    put_wall = None

    # --------------------------------------------------------
    # CALL WALL
    # --------------------------------------------------------

    positive = values > 0

    if np.any(positive):

        positive_values = values[
            positive
        ]

        threshold = np.percentile(
            positive_values,
            WALL_PERCENTILE
        )

        candidates = []

        for i in range(
            len(strikes)
        ):

            if values[i] <= threshold:
                continue

            if strikes[i] < spot:
                continue

            if abs(
                strikes[i] / spot - 1.0
            ) > MAX_WALL_DISTANCE:

                continue

            score = cluster_score(
                values,
                i
            )

            candidates.append(
                (
                    score,
                    abs(
                        strikes[i] - spot
                    ),
                    strikes[i]
                )
            )

        if candidates:

            candidates.sort(
                key=lambda x: (
                    -x[0],
                    x[1]
                )
            )

            call_wall = candidates[0][2]

    # --------------------------------------------------------
    # PUT WALL
    # --------------------------------------------------------

    negative = values < 0

    if np.any(negative):

        negative_values = np.abs(
            values[negative]
        )

        threshold = np.percentile(
            negative_values,
            WALL_PERCENTILE
        )

        candidates = []

        for i in range(
            len(strikes)
        ):

            if values[i] >= 0:
                continue

            if abs(
                values[i]
            ) < threshold:

                continue

            if strikes[i] > spot:
                continue

            if abs(
                strikes[i] / spot - 1.0
            ) > MAX_WALL_DISTANCE:

                continue

            score = cluster_score(
                values,
                i
            )

            candidates.append(
                (
                    score,
                    abs(
                        strikes[i] - spot
                    ),
                    strikes[i]
                )
            )

        if candidates:

            candidates.sort(
                key=lambda x: (
                    -x[0],
                    x[1]
                )
            )

            put_wall = candidates[0][2]

    return (
        call_wall,
        put_wall
    )


# ============================================================
# REGIME
# ============================================================

def get_regime(
    spot,
    flip
):

    if flip is None:
        return "UNKNOWN"

    if spot > flip:
        return "POSITIVE_GAMMA"

    if spot < flip:
        return "NEGATIVE_GAMMA"

    return "AT_FLIP"


# ============================================================
# SPX → ES
# ============================================================

def convert_to_es(
    spx_level,
    spx_spot,
    es_spot
):

    if spx_level is None:
        return None

    if es_spot is None:
        return spx_level

    basis = (
        es_spot
        - spx_spot
    )

    return (
        spx_level
        + basis
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("===================================")
    print("       MACH1 GAMMA ENGINE")
    print("       MES INTRADAY MODE")
    print("===================================")

    # --------------------------------------------------------
    # SPX
    # --------------------------------------------------------

    print("Getting SPX price...")

    spx_spot = get_spx_spot()

    print(
        f"SPX: {spx_spot:.2f}"
    )

    # --------------------------------------------------------
    # ES
    # --------------------------------------------------------

    print("Getting ES price...")

    es_spot = get_es_price()

    if es_spot is not None:

        print(
            f"ES:  {es_spot:.2f}"
        )

    else:

        print(
            "ES price unavailable."
        )

    # --------------------------------------------------------
    # OPTIONS
    # --------------------------------------------------------

    print(
        "Downloading SPX option chains..."
    )

    chains = get_options()

    print(
        f"Loaded {len(chains)} option-chain datasets."
    )

    # --------------------------------------------------------
    # PREPARE
    # --------------------------------------------------------

    print(
        "Filtering option contracts..."
    )

    (
        strikes,
        oi,
        iv,
        dte,
        signs
    ) = prepare_options(
        spx_spot,
        chains
    )

    print(
        f"Usable contracts: {len(strikes)}"
    )

    print(
        f"DTE filter: 0-{MAX_DTE} days"
    )

    # --------------------------------------------------------
    # PROFILE
    # --------------------------------------------------------

    print(
        "Building GEX profile..."
    )

    profile = build_gex_profile(
        spx_spot,
        strikes,
        oi,
        iv,
        dte,
        signs
    )

    print(
        f"Calculated {len(profile)} strikes."
    )

    # --------------------------------------------------------
    # SMOOTH
    # --------------------------------------------------------

    print(
        "Smoothing GEX profile..."
    )

    profile = smooth_profile(
        profile
    )

    # --------------------------------------------------------
    # GAMMA FLIP
    # --------------------------------------------------------

    print(
        "Finding Gamma Flip..."
    )

    gamma_flip_spx = find_gamma_flip(
        spx_spot,
        strikes,
        oi,
        iv,
        dte,
        signs
    )

    print(
        "Gamma Flip SPX:",
        gamma_flip_spx
    )

    # --------------------------------------------------------
    # WALLS
    # --------------------------------------------------------

    print(
        "Finding Call / Put Wall clusters..."
    )

    (
        call_wall_spx,
        put_wall_spx
    ) = find_walls(
        profile,
        spx_spot
    )

    print(
        "Call Wall SPX:",
        call_wall_spx
    )

    print(
        "Put Wall SPX:",
        put_wall_spx
    )

    # --------------------------------------------------------
    # CONVERT TO ES
    # --------------------------------------------------------

    gamma_flip_es = convert_to_es(
        gamma_flip_spx,
        spx_spot,
        es_spot
    )

    call_wall_es = convert_to_es(
        call_wall_spx,
        spx_spot,
        es_spot
    )

    put_wall_es = convert_to_es(
        put_wall_spx,
        spx_spot,
        es_spot
    )

    # --------------------------------------------------------
    # REGIME
    # --------------------------------------------------------

    regime = get_regime(
        spx_spot,
        gamma_flip_spx
    )

    # --------------------------------------------------------
    # OUTPUT
    # --------------------------------------------------------

    output = {

        "indicator":
            "Mach1 Gamma",

        "symbol":
            "MES",

        "source":
            "SPX options",

        "method":
            "Filtered + smoothed GEX",

        "max_dte":
            MAX_DTE,

        "spx_spot":
            round(
                spx_spot,
                2
            ),

        "es_spot":
            (
                round(
                    es_spot,
                    2
                )
                if es_spot is not None
                else None
            ),

        "call_wall":
            (
                round(
                    call_wall_es,
                    2
                )
                if call_wall_es is not None
                else None
            ),

        "gamma_flip":
            (
                round(
                    gamma_flip_es,
                    2
                )
                if gamma_flip_es is not None
                else None
            ),

        "put_wall":
            (
                round(
                    put_wall_es,
                    2
                )
                if put_wall_es is not None
                else None
            ),

        "regime":
            regime,

        "updated_utc":
            datetime.now(
                timezone.utc
            ).isoformat()
    }

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    os.makedirs(
        os.path.dirname(
            OUTPUT_FILE
        ),
        exist_ok=True
    )

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            output,
            f,
            indent=2
        )

    # --------------------------------------------------------
    # DISPLAY
    # --------------------------------------------------------

    print("")
    print("===================================")
    print("          MACH1 GAMMA")
    print("===================================")

    print(
        f"CALL WALL : {output['call_wall']}"
    )

    print(
        f"GAMMA FLIP: {output['gamma_flip']}"
    )

    print(
        f"PUT WALL  : {output['put_wall']}"
    )

    print(
        f"REGIME    : {output['regime']}"
    )

    print(
        f"UPDATED   : {output['updated_utc']}"
    )

    print("===================================")


if __name__ == "__main__":
    main()
