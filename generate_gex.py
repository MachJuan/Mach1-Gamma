
import json
import math
import os
from datetime import datetime, timezone

import numpy as np
import yfinance as yf
from scipy.stats import norm


# ============================================================
# MACH1 GAMMA
# GEX engine for MES / ES
#
# Data source:
#   SPX options via Yahoo Finance
#
# Output:
#   data/gex_es.json
#
# Main levels:
#   Call Wall
#   Gamma Flip
#   Put Wall
# ============================================================


OUTPUT_FILE = "data/gex_es.json"

RISK_FREE_RATE = 0.04

# Use options with expirations up to this many days away.
MAX_DTE = 90

# Ignore extremely small OI positions.
MIN_OPEN_INTEREST = 5

# Number of price points used when searching for Gamma Flip.
FLIP_STEPS = 241

# Search approximately +/- 8% around spot.
FLIP_RANGE = 0.08

# Require a gamma concentration to be meaningful.
WALL_PERCENTILE = 85


# ------------------------------------------------------------
# Black-Scholes gamma
# ------------------------------------------------------------

def bs_gamma(S, K, T, sigma, r):
    if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
        return 0.0

    try:
        d1 = (
            math.log(S / K)
            + (r + 0.5 * sigma * sigma) * T
        ) / (sigma * math.sqrt(T))

        return (
            norm.pdf(d1)
            / (S * sigma * math.sqrt(T))
        )

    except Exception:
        return 0.0


# ------------------------------------------------------------
# Safe numeric conversion
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# Get SPX spot
# ------------------------------------------------------------

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
        raise RuntimeError("Unable to obtain SPX price.")

    return float(hist["Close"].dropna().iloc[-1])


# ------------------------------------------------------------
# Get ES futures price
# ------------------------------------------------------------

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

    return float(hist["Close"].dropna().iloc[-1])


# ------------------------------------------------------------
# Get options chain
# ------------------------------------------------------------

def get_options():
    ticker = yf.Ticker("^SPX")

    expirations = ticker.options

    if not expirations:
        raise RuntimeError("No SPX option expirations returned.")

    now = datetime.now(timezone.utc)

    chains = []

    for expiration in expirations:

        try:
            exp_dt = datetime.strptime(
                expiration,
                "%Y-%m-%d"
            ).replace(tzinfo=timezone.utc)

            dte = (
                exp_dt - now
            ).total_seconds() / 86400.0

            if dte < -0.5:
                continue

            if dte > MAX_DTE:
                continue

            chain = ticker.option_chain(expiration)

            calls = chain.calls.copy()
            puts = chain.puts.copy()

            calls["type"] = "call"
            puts["type"] = "put"

            calls["expiration"] = expiration
            puts["expiration"] = expiration

            calls["dte"] = max(dte, 0.01)
            puts["dte"] = max(dte, 0.01)

            chains.append(calls)
            chains.append(puts)

        except Exception:
            continue

    if not chains:
        raise RuntimeError("Unable to retrieve SPX option chains.")

    return chains


# ------------------------------------------------------------
# Build GEX profile
# ------------------------------------------------------------

def build_gex_profile(spot, chains):

    profile = {}

    for chain in chains:

        for _, row in chain.iterrows():

            strike = num(row.get("strike"))

            oi = num(row.get("openInterest"))

            iv = num(row.get("impliedVolatility"))

            dte = num(row.get("dte"))

            option_type = row.get("type")

            if strike <= 0:
                continue

            if oi < MIN_OPEN_INTEREST:
                continue

            if iv <= 0.0001:
                continue

            T = max(dte / 365.0, 1.0 / 365.0)

            gamma = bs_gamma(
                spot,
                strike,
                T,
                iv,
                RISK_FREE_RATE
            )

            if gamma <= 0:
                continue

            # Standard dealer-GEX approximation:
            #
            # Calls -> positive gamma
            # Puts  -> negative gamma
            #
            # Contract multiplier = 100
            # GEX expressed approximately per 1% move.

            gex = (
                gamma
                * oi
                * 100
                * spot
                * spot
                * 0.01
            )

            # Give nearer expirations more influence,
            # while preventing 0DTE from completely
            # overwhelming the entire profile.

            dte_weight = 1.0 / math.sqrt(
                max(dte, 1.0)
            )

            gex *= dte_weight

            if option_type == "put":
                gex *= -1.0

            profile[strike] = (
                profile.get(strike, 0.0)
                + gex
            )

    return profile


# ------------------------------------------------------------
# Calculate net GEX at hypothetical price
# ------------------------------------------------------------

def net_gex_at_price(price, chains):

    total = 0.0

    for chain in chains:

        for _, row in chain.iterrows():

            strike = num(row.get("strike"))
            oi = num(row.get("openInterest"))
            iv = num(row.get("impliedVolatility"))
            dte = num(row.get("dte"))

            option_type = row.get("type")

            if strike <= 0:
                continue

            if oi < MIN_OPEN_INTEREST:
                continue

            if iv <= 0.0001:
                continue

            T = max(dte / 365.0, 1.0 / 365.0)

            gamma = bs_gamma(
                price,
                strike,
                T,
                iv,
                RISK_FREE_RATE
            )

            if gamma <= 0:
                continue

            gex = (
                gamma
                * oi
                * 100
                * price
                * price
                * 0.01
            )

            dte_weight = 1.0 / math.sqrt(
                max(dte, 1.0)
            )

            gex *= dte_weight

            if option_type == "put":
                gex *= -1.0

            total += gex

    return total


# ------------------------------------------------------------
# Find Gamma Flip
# ------------------------------------------------------------

def find_gamma_flip(spot, chains):

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
            value = net_gex_at_price(
                float(price),
                chains
            )

            values.append(value)

        except Exception:
            values.append(np.nan)

    crossings = []

    for i in range(1, len(prices)):

        a = values[i - 1]
        b = values[i]

        if not np.isfinite(a) or not np.isfinite(b):
            continue

        if a == 0:
            crossings.append(prices[i - 1])

        elif a * b < 0:

            # Linear interpolation between
            # the two points.

            p1 = prices[i - 1]
            p2 = prices[i]

            if b != a:

                flip = (
                    p1
                    + (0.0 - a)
                    * (p2 - p1)
                    / (b - a)
                )

                crossings.append(flip)

    if not crossings:
        return None

    # Prefer the crossing closest to current SPX.
    return min(
        crossings,
        key=lambda x: abs(x - spot)
    )


# ------------------------------------------------------------
# Find strongest meaningful wall
# ------------------------------------------------------------

def find_walls(profile, spot):

    if not profile:
        return None, None

    strikes = np.array(
        list(profile.keys()),
        dtype=float
    )

    values = np.array(
        list(profile.values()),
        dtype=float
    )

    positive = values > 0
    negative = values < 0

    call_wall = None
    put_wall = None

    # -------------------------
    # Call Wall
    # -------------------------

    if np.any(positive):

        call_values = values[positive]
        call_strikes = strikes[positive]

        threshold = np.percentile(
            call_values,
            WALL_PERCENTILE
        )

        candidates = np.where(
            call_values >= threshold
        )[0]

        if len(candidates):

            # Prefer resistance above current spot.
            above = [
                i for i in candidates
                if call_strikes[i] >= spot
            ]

            if above:

                call_wall = call_strikes[
                    max(
                        above,
                        key=lambda i: call_values[i]
                    )
                ]

            else:

                call_wall = call_strikes[
                    candidates[
                        np.argmax(
                            call_values[candidates]
                        )
                    ]
                ]

    # -------------------------
    # Put Wall
    # -------------------------

    if np.any(negative):

        put_values = np.abs(
            values[negative]
        )

        put_strikes = strikes[negative]

        threshold = np.percentile(
            put_values,
            WALL_PERCENTILE
        )

        candidates = np.where(
            put_values >= threshold
        )[0]

        if len(candidates):

            below = [
                i for i in candidates
                if put_strikes[i] <= spot
            ]

            if below:

                put_wall = put_strikes[
                    max(
                        below,
                        key=lambda i: put_values[i]
                    )
                ]

            else:

                put_wall = put_strikes[
                    candidates[
                        np.argmax(
                            put_values[candidates]
                        )
                    ]
                ]

    return call_wall, put_wall


# ------------------------------------------------------------
# Determine regime
# ------------------------------------------------------------

def get_regime(spot, flip):

    if flip is None:
        return "UNKNOWN"

    if spot > flip:
        return "POSITIVE_GAMMA"

    if spot < flip:
        return "NEGATIVE_GAMMA"

    return "AT_FLIP"


# ------------------------------------------------------------
# Convert SPX level to ES
# ------------------------------------------------------------

def convert_to_es(spx_level, spx_spot, es_spot):

    if spx_level is None:
        return None

    if es_spot is None:
        return spx_level

    basis = es_spot - spx_spot

    return spx_level + basis


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():

    print("===================================")
    print("       MACH1 GAMMA ENGINE")
    print("===================================")

    print("Getting SPX price...")

    spx_spot = get_spx_spot()

    print(f"SPX: {spx_spot:.2f}")

    print("Getting ES price...")

    es_spot = get_es_price()

    if es_spot:
        print(f"ES:  {es_spot:.2f}")
    else:
        print("ES price unavailable.")

    print("Downloading SPX option chains...")

    chains = get_options()

    print(
        f"Loaded {len(chains)} option-chain datasets."
    )

    print("Building GEX profile...")

    profile = build_gex_profile(
        spx_spot,
        chains
    )

    print(
        f"Calculated {len(profile)} strikes."
    )

    print("Finding Gamma Flip...")

    gamma_flip_spx = find_gamma_flip(
        spx_spot,
        chains
    )

    print(
        "Gamma Flip SPX:",
        gamma_flip_spx
    )

    print("Finding Call / Put Walls...")

    call_wall_spx, put_wall_spx = find_walls(
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

    # Convert SPX levels to ES price space.

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

    regime = get_regime(
        spx_spot,
        gamma_flip_spx
    )

    output = {

        "indicator": "Mach1 Gamma",

        "symbol": "MES",

        "source": "SPX options",

        "spx_spot": round(
            spx_spot,
            2
        ),

        "es_spot": (
            round(es_spot, 2)
            if es_spot is not None
            else None
        ),

        "call_wall": (
            round(call_wall_es, 2)
            if call_wall_es is not None
            else None
        ),

        "gamma_flip": (
            round(gamma_flip_es, 2)
            if gamma_flip_es is not None
            else None
        ),

        "put_wall": (
            round(put_wall_es, 2)
            if put_wall_es is not None
            else None
        ),

        "regime": regime,

        "updated_utc": datetime.now(
            timezone.utc
        ).isoformat()

    }

    os.makedirs(
        os.path.dirname(OUTPUT_FILE),
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
